"""逐专家 fused MLP（堆叠权重），可选 TP+EP 切分。

``ep_size > 1`` 时，只有本地拥有的专家在 ``w13_weight`` / ``w2_weight``
中存权重；远程专家靠 ``expert_map``（远程为 -1、本地为本地索引）mask 掉。

堆叠权重布局（每 rank，TP+EP 切分后）：
    w13_weight: (E_local, 2 * I_local, H)        # 行按 TP 切
    w2_weight:  (E_local, H,           I_local)  # 列按 TP 切

``TP > 1`` 时 ``forward`` 返回的 *partial* 输出需在 TP 组上求和；EP partial
同理需 EP all-reduce。``MoEDispatcher.combine`` 负责这两次 reduce。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
# fusedExperts是指将MoE的MLP实现Fused版本
class FusedExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int = 1,
        ep_size: int = 1,
        ep_rank: int = 0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert intermediate_size % tp_size == 0, (
            f"intermediate_size={intermediate_size} not divisible by tp_size={tp_size}"
        )
        assert num_experts % ep_size == 0, (
            f"num_experts={num_experts} not divisible by ep_size={ep_size}"
        )
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        self.local_expert_start = ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.intermediate_size_local = intermediate_size // tp_size
        kw = {"dtype": dtype, "device": device}
        # 仅 BF16/FP16/FP32，无量化：直接用浮点堆叠权重。
        # w13_weight是指up gate的权重矩阵，这里为了减少launch把它们fused在一个矩阵里面了
        self.w13_weight = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * self.intermediate_size_local, hidden_size, **kw),
            requires_grad=False,
        )
        # w2_weight指的是down的权重矩阵
        self.w2_weight = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, self.intermediate_size_local, **kw),
            requires_grad=False,
        )

        # expert_map 是 (num_experts, ep_rank, ep_size) 的纯函数，构造后不变。
        # 注册为 buffer 以便只建一次（非每步 forward 分配）并随 .to()/.cuda()
        # 一起搬运。-1 标远程专家；本地专家持有其在 w13/w2 中的本地行索引。
        # full功能是创建一个维度矩阵，然后矩阵里面填满同一个值，如这里创建一个一维的长度为num_experts且内容全为-1的矩阵
        expert_map = torch.full((self.num_experts,), -1, dtype=torch.int32, device=device)
        local_ids = torch.arange(self.num_local_experts, dtype=torch.int32, device=device)
        # 只填写本地rank拥有的experts的local_ids
        expert_map[self.local_expert_start : self.local_expert_end] = local_ids
        self.register_buffer("expert_map", expert_map, persistent=False)

        # 单块平坦 workspace 供 triton fused 路径使用。三个中间量被切成
        # *连续*的平坦子块（非列切片），保证各自连续（下游 .view 廉价）且
        # GEMM2 的输入/输出占据不相交内存：
        #
        #   flat[0          : TK*I    ] -> c2  (silu·mul out, GEMM2 input A)
        #   flat[TK*I       : TK*I+TK*H] -> c3 (GEMM2 output C, 连续)
        #   flat[0          : TK*2I   ] -> c1  (GEMM1 out; 与 c2 & c3 区域重叠，
        #                                       但 GEMM2 跑前已死——kernel 启动 /
        #                                       stream 顺序使其安全）
        #
        # 唯一的 kernel 内 hazard 是 GEMM2（A=c2, C=c3 被同一次 launch 的并发
        # 列 tile 触碰）；上面 c2 与 c3 不相交，故无读写竞争。c1 vs c2 重叠
        # 安全，因为 silu 在拷进 c2 前已完整算出结果。
        #
        # 只增不减；warmup_model() 按最大 TK 一次分配，故永不重分配、基址在
        # CUDA-graph replay 间稳定。

        # c2：用来存激活函数的输出（给 w2 做输入的）。
        # c3：用来存 w2 算出来的结果。
        # c1：用来存 w13 算出来的中间结果。

        #c1 和 c2 的内存区域是重叠（重叠）的。
        # 因为在 GPU 里，w13 算出结果 c1 后，紧接着经过激活函数，结果就直接覆盖写进 c2 了（此时 c1 的数据没用了，叫“已死”）。
        # 而 c2（输入）和 c3（输出）是绝对不能重叠的，因为 w2 的矩阵乘法是一边读 c2 一边写 c3。
        self._workspace: torch.Tensor | None = None


    def _ensure_workspace(self, tk: int, device, dtype) -> torch.Tensor:
        # 这里tk是token数目与其要路由到的expert数目的乘积
        # intermediate是expert的中间维度
        intermediate = self.intermediate_size_local
        H = self.hidden_size
        # need是workspace需要的空间，这里为什么是max(2*intermediate,intermediate*h)呢
        # 在c1中计算 hidden * w_13即计算hidden和w_13(up和gate)的矩阵乘，需要tk ✖️ 2*intermediate
        # 而在c2或者c3中计算的中间大矩阵split之后就需要tk * intermediate（c2） + tk * H (c3)的空间
        # 且c2可在c1的空间基础上完成，c3需要新的空间；所以二者取max
        need = tk * max(2 * intermediate, intermediate + H)
        ws = self._workspace
        if (
            ws is None
            or ws.numel() < need
            or ws.device != device
            or ws.dtype != dtype
        ):
            ws = torch.empty(need, dtype=dtype, device=device)
            self._workspace = ws
        return ws

    # 在GPU上的fused forward，准备好workspace并用triton进行MoE计算
    def forward(
        self,
        hidden_states: torch.Tensor,        # (T, H)，完整 token 集
        topk_ids: torch.Tensor,             # (T, K)
        topk_weights: torch.Tensor,         # (T, K)
        expert_mask: torch.Tensor | None = None,  # legacy，忽略
    ) -> torch.Tensor:
        emap = self.expert_map
        if emap.device != hidden_states.device:
            emap = emap.to(hidden_states.device)
            self.expert_map = emap
        if hidden_states.is_cuda:
            from .fused_moe_kernel import fused_experts_triton

            intermediate = self.intermediate_size_local
            H = self.hidden_size
            tk = hidden_states.size(0) * topk_ids.size(1)
            ws = self._ensure_workspace(tk, hidden_states.device, hidden_states.dtype)
            c1 = ws[: tk * 2 * intermediate].view(tk, 2 * intermediate)
            c2 = ws[: tk * intermediate].view(tk, intermediate)
            c3 = ws[tk * intermediate : tk * intermediate + tk * H].view(tk, H)
            return fused_experts_triton(
                hidden_states,
                self.w13_weight,
                self.w2_weight,
                topk_ids.to(torch.int32),
                topk_weights,
                expert_map=emap,
                cache1=c1,
                cache2=c2,
                cache3=c3,
            )
        # CPU eager fallback（gloo 多 rank parity 测试用）。
        return self._eager_forward(hidden_states, topk_ids, topk_weights)
    # 在cpu上的即使forward
    def _eager_forward(self, hidden_states, topk_ids, topk_weights):
        T, H = hidden_states.shape
        out = torch.zeros_like(hidden_states)
        E = self.num_experts
        oh = F.one_hot(topk_ids.long(), num_classes=E).permute(2, 1, 0)  # (E, K, T)
        local_present = oh.sum(dim=(-1, -2)) > 0
        # 只算本地拥有的专家。
        for e in range(self.local_expert_start, self.local_expert_end):
            if not local_present[e]:
                continue
            local_e = e - self.local_expert_start
            mat = oh[e]
            k_idx, t_idx = torch.where(mat)
            x = hidden_states.index_select(0, t_idx)
            w13 = self.w13_weight[local_e]
            gate_up = F.linear(x, w13)
            I_local = self.intermediate_size_local
            gate, up = gate_up.split(I_local, dim=-1)
            mid = F.silu(gate) * up
            y = F.linear(mid, self.w2_weight[local_e])
            y = y * topk_weights[t_idx, k_idx, None].to(y.dtype)
            out.index_add_(0, t_idx, y)
        return out
