"""MoE 并行布局：TP / EP / TP+EP，token 全 replicated。

设计（镜像 vLLM ``FusedMoE``，去掉底层量化 kernel）：

- 进入 ``FusedMoE.forward`` 的 token 在 MoE 组的每个 rank 上都**完整
  复制**。若 attention 按 DP/CP 切了 token，上游的 ``prepare_mlp`` 会
  先做一次 all-gather。
- EP 纯靠 mask 实现：每个 rank 只算自己拥有的专家
  （``[ep_rank*E/EP, (ep_rank+1)*E/EP)``），partial 输出在 EP 组上
  all-reduce 求和。
- TP 切每个专家的 intermediate 维度（``w13`` 行并行、``w2`` 列并行），
  ``w2`` 之后在 TP 组上 all-reduce。
- TP 与 EP 正交：当前 combine 依次执行 EP 和 TP 两次 all-reduce。

可通过 create_moe_parallel_config 建立正交组，也可显式传入已有 group。
size=1 时 collective 是 no-op；size>1 且 group=None 时使用默认 WORLD，
运行前检查 WORLD 是否确实匹配该并行维度。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class MoEParallelConfig:
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    tp_group: Optional["dist.ProcessGroup"] = None
    ep_group: Optional["dist.ProcessGroup"] = None
    # 若 attention 按 DP/CP 切了 token，给出需要 all-gather 的组。
    dp_group: Optional["dist.ProcessGroup"] = None
    dp_size: int = 1
    dp_rank: int = 0

    def __post_init__(self):
        for axis in ("tp", "ep", "dp"):
            size, rank = getattr(self, f"{axis}_size"), getattr(self, f"{axis}_rank")
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError(f"{axis}_size must be a positive integer")
            if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < size:
                raise ValueError(f"{axis}_rank must be in [0, {size})")

    def expert_range(self, num_experts: int) -> tuple[int, int]:
        if num_experts <= 0 or num_experts % self.ep_size:
            raise ValueError("num_experts must be positive and divisible by ep_size")
        count = num_experts // self.ep_size
        return self.ep_rank * count, (self.ep_rank + 1) * count

    def validate_group(self, axis: str):
        size = getattr(self, f"{axis}_size")
        if size == 1:
            return
        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before a parallel MoE forward")
        group = getattr(self, f"{axis}_group")
        if (dist.get_world_size(group) != size
                or dist.get_rank(group) != getattr(self, f"{axis}_rank")):
            raise ValueError(f"{axis} process group does not match its configured size/rank")


def create_moe_parallel_config(tp_size: int = 1, ep_size: int = 1) -> MoEParallelConfig:
    """Create orthogonal groups on all ranks of an already initialized WORLD.

    Global rank = ep_rank * tp_size + tp_rank. Every rank must call this
    function in the same order, including ranks outside each subgroup.
    """
    MoEParallelConfig(tp_size=tp_size, ep_size=ep_size)
    if not dist.is_initialized():
        raise RuntimeError("Initialize torch.distributed before creating MoE groups")
    if dist.get_world_size() != tp_size * ep_size:
        raise ValueError("WORLD size must equal tp_size * ep_size")
    rank = dist.get_rank()
    config = MoEParallelConfig(
        tp_size=tp_size, tp_rank=rank % tp_size,
        ep_size=ep_size, ep_rank=rank // tp_size,
    )
    for ep_rank in range(ep_size):
        ranks = list(range(ep_rank * tp_size, (ep_rank + 1) * tp_size))
        if tp_size > 1:
            group = dist.new_group(ranks)
            if rank in ranks:
                config.tp_group = group
    for tp_rank in range(tp_size):
        ranks = list(range(tp_rank, tp_size * ep_size, tp_size))
        if ep_size > 1:
            group = dist.new_group(ranks)
            if rank in ranks:
                config.ep_group = group
    return config


class MoEDispatcher:
    """基于 mask 的 dispatcher（无 all-to-all）。

    每个 rank 已持有全部 token。``combine`` 对各 rank 的 partial 专家
    输出做跨 rank all-reduce。
    """

    def __init__(self, parallel: MoEParallelConfig, num_experts: int):
        start, end = parallel.expert_range(num_experts)
        self.parallel = parallel
        self.num_experts = num_experts
        self.experts_per_rank = num_experts // parallel.ep_size
        self.local_expert_start, self.local_expert_end = start, end

    def combine(self, partial: torch.Tensor) -> torch.Tensor:
        """在 EP 与 TP 组上对 partial 专家输出求和。

        ``partial`` 是本 rank ``FusedExperts(...)`` 的输出（已按 topk
        权重加权、并在本地专家上 reduce 完）。TP 与 EP 组独立时需要两次
        all-reduce，顺序不影响求和结果。

        NOTE: 传 ``group=None`` 给 ``dist.all_reduce`` 会用默认（WORLD）
        组——当 ``ep_size==1`` 时它正是正确的 TP 组。因此**不要**用
        ``is not None`` 守卫，否则 ``tp>1, ep==1`` 时会漏掉 TP all-reduce
        导致 MoE 输出错乱。size==1 时下面两个分支都不进，自然 no-op。
        """
        if self.parallel.ep_size > 1:
            self.parallel.validate_group("ep")
            dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=self.parallel.ep_group)
        if self.parallel.tp_size > 1:
            self.parallel.validate_group("tp")
            dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=self.parallel.tp_group)
        return partial


def prepare_mlp(
    hidden: torch.Tensor,
    parallel: MoEParallelConfig,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """若 attention 把 token 按 DP/CP 切了，做一次 all-gather。

    返回 ``(full_hidden, scatter_idx)``。未 gather 时 ``scatter_idx`` 为
    None；否则它是把本 rank shard 映射回去的索引，供调用方在 MoE 后
    re-shard。
    """
    if parallel.dp_size <= 1:
        return hidden, None
    parallel.validate_group("dp")

    T_local, H = hidden.shape
    T_full = T_local * parallel.dp_size
    full = torch.empty(T_full, H, dtype=hidden.dtype, device=hidden.device)
    dist.all_gather_into_tensor(full, hidden.contiguous(), group=parallel.dp_group)
    # 本 rank 的 shard 在 [dp_rank*T_local : (dp_rank+1)*T_local)。
    start = parallel.dp_rank * T_local
    scatter_idx = torch.arange(start, start + T_local, device=hidden.device)
    return full, scatter_idx
