"""KV Cache Manager - 由 Scheduler 持有"""

import triton.language as tl
import triton
from typing import Optional, Tuple
import torch
from miniinfer.utils import get_logger

from .interface import (
  IKVCacheStorage,
  ITokenAllocator,
  IRequestPool,
  IPrefixCache,
)
from .memory_pool import MHAKVCacheStorage, PagedTokenAllocator, RequestPool
from .radix_cache import RadixCache
from miniinfer.scheduler.scheduler_batch import ScheduledBatch, Req, ForwardMode, ForwardBatch

logger = get_logger(__name__)


class KVCacheManager:
  """
  KV Cache 管理器

  由多个组件构成:
  - CacheStorage: 物理存储 (MHAKVCacheStorage)
  - TokenAllocator: 分配 KV indices (PagedTokenAllocator)
  - RequestPool: 管理不同请求分配
  - PrefixCache: 前缀缓存 (RadixCache)
  - MemoryBudgetManager: 显存预算管理

  初始化流程:
  1. 使用 gpu_memory_utilization 计算 GPU 空闲显存能容纳的 max_total_tokens
  2. 根据 max_total_tokens 初始化物理存储和各组件
  """

  def __init__(
    self,
    size: int,                    # kv_cache的总容量
    max_requests: int = 256,      # 允许并发的最大req数量
    max_context_len: int = 4096,  #  单请求的最大上下文数
    num_layers: int = 32,
    num_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    enable_prefix_cache: bool = True,
    page_size: int = 256,          # 一个page能存储的最大token数
    max_extend_tokens: int = 8192,  # 单次prefill允许的最大token数
  ):
    """
    初始化 KV Cache 管理器

    Args:
        size: KV cache 容量（token 数），None 时自动根据 GPU 显存计算
        max_requests: 最大并发请求数
        max_context_len: 单请求最大上下文长度
        num_layers: 模型层数
        num_heads: KV head 数量
        head_dim: head 维度
        dtype: 数据类型
        device: GPU 设备
        enable_prefix_cache: 是否启用前缀缓存
        page_size: 每页 token 数
        max_extend_tokens: 单次 prefill 最大 token 数
    """
    self.max_requests = max_requests
    self.max_context_len = max_context_len
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.dtype = dtype
    self.device = device
    self.enable_prefix_cache = enable_prefix_cache
    self.page_size = page_size
    self.max_extend_tokens = max_extend_tokens

    # for debug
    self.size = size

    # ============ Pinned Memory Buffer Pool (SGLang 风格优化) ============
    # 预分配 pinned memory buffers 用于异步 CPU→GPU 传输
    # 这避免了每次 step 都动态分配 tensor 和临时 pinned buffer 的开销
    logger.info(
      f"Initializing pinned memory buffer pool (max_requests={max_requests}, max_extend_tokens={max_extend_tokens})..."
    )

    # 固定内存（pinned memory） 缓冲区，用于在 CPU 和 GPU 之间异步传输数据。在推理过程中，调度器会将请求的长度、token IDs 等信息从 CPU 复制到 GPU。使用固定内存可以避免每次动态分配临时缓冲区，并且允许异步复制（non_blocking=True），提升性能。
    #  固定内存是操作系统保证不会被换页到磁盘的内存区域，GPU 可以直接 DMA 访问，传输速度更快。
    # 序列长度相关 (每个 batch 最多 max_requests 个请求)
    '''
    pinned_seq_lens：每个请求的序列长度。
    pinned_prefix_lens：每个请求的前缀长度（可复用的部分）。
    pinned_extend_lens：每个请求需要新计算的 token 数。
    pinned_req_pool_indices：请求在 RequestPool 中的行号。
    pinned_decode_input_ids：decode 阶段每个请求的新 token ID。
    pinned_input_ids：prefill/extend 阶段的所有新 token ID（总长度不超过 max_extend_tokens
    '''

    self.pinned_seq_lens = torch.empty(max_requests, dtype=torch.int64, pin_memory=True)
    self.pinned_prefix_lens = torch.empty(max_requests, dtype=torch.int64, pin_memory=True)
    self.pinned_extend_lens = torch.empty(max_requests, dtype=torch.int64, pin_memory=True)
    self.pinned_req_pool_indices = torch.empty(max_requests, dtype=torch.int64, pin_memory=True)
    # Decode input ids (one token per request)
    self.pinned_decode_input_ids = torch.empty(max_requests, dtype=torch.int64, pin_memory=True)

    # Token IDs (extend 阶段可能有很多 token)
    self.pinned_input_ids = torch.empty(max_extend_tokens, dtype=torch.int64, pin_memory=True)

    logger.info("Pinned memory buffer pool initialized successfully")

    # 初始化物理存储
    self.storage: IKVCacheStorage = MHAKVCacheStorage(
      size=size,
      page_size=page_size,
      num_layers=num_layers,
      num_heads=num_heads,
      head_dim=head_dim,
      dtype=dtype,
      device=device,
    )

    # 初始化 Token 分配器
    '''
    PagedTokenAllocator 就是页式分配器，管理空闲页列表，提供 alloc_extend、alloc_decode、free 等方法。它不直接操作存储张量，只负责分配和回收页号（或 token 索引）。
    '''
    self.token_allocator: ITokenAllocator = PagedTokenAllocator(
      size=size,
      device=device,
      page_size=page_size,
    )

    # 初始化请求池
    '''
    RequestPool 维护一张 req_to_token 表，形状 (max_requests, max_context_len)，用于记录每个请求的每个 token 对应的物理 KV 索引。当调度器为请求分配物理位置后，会写入这张表
    '''
    self.request_pool: IRequestPool = RequestPool(
      max_requests=max_requests,
      max_context_len=max_context_len,
      device=device,
    )

    # 初始化前缀缓存
    if enable_prefix_cache:
      self.prefix_cache: IPrefixCache = RadixCache(
        token_allocator=self.token_allocator,
        page_size=page_size,
      )
    else:
      self.prefix_cache = None

    # prefill max length
    self.max_extend_tokens = max_extend_tokens

    logger.info(
      f"Initialized KVCacheManager with size={size} tokens, "
      f"max_requests={max_requests}, max_context_len={max_context_len}, "
      f"num_layers={num_layers}, num_heads={num_heads}, head_dim={head_dim}, "
      f"dtype={dtype}, device={device}, enable_prefix_cache={enable_prefix_cache}, "
      f"page_size={page_size}, max_extend_tokens={max_extend_tokens}"
    )

  # ============== public methods for scheduler ==============

  def warmup_release_path(self) -> None:
    """
    Warm up CUDA ops used by the deferred release path.

    The first real `drain_pending_releases()` can otherwise pay for lazy module
    loading (e.g. request-pool read cast and grouped free/unique kernels). Use
    a no-op warmup that leaves allocator/request-pool state unchanged.

    在推理引擎中，drain_pending_releases() 通常负责批量释放那些被标记为待回收的 KV 页。它可能会用到：

  request_pool.read：从请求池中读取对应 token 的 KV 索引（用于释放时定位）。
  token_allocator.free：将页归还给分配器。
  某些实现中还有分组释放（begin_free_group / end_free_group）优化，合并多次 free 操作。
  这些操作在第一次调用时可能触发：

  CUDA kernel 的即时编译（JIT）；
  Python 模块的延迟导入；
  张量操作的初始化。
  这会导致第一次真实释放时出现额外的延迟，影响用户体验。因此，warmup_release_path 通过执行一次“无实际影响”的预热操作，把这些一次性开销提前承担掉。
    """
    if not torch.cuda.is_available() or str(self.device) == "cpu":
      return

    # Warm request-pool read/cast used to fetch kv_indices during release.
    warmup_tokens = max(1, min(int(self.page_size), int(self.max_context_len)))
    _ = self.request_pool.read(0, slice(0, warmup_tokens))

    free_pages = getattr(self.token_allocator, "free_pages", None)
    if free_pages is None or free_pages.numel() == 0:
      torch.cuda.synchronize()
      return

    # Warm grouped free/flush kernels using an already-free page so allocator
    # contents stay logically unchanged after dedup.
    dummy_page = free_pages[:1].clone()
    page_offsets = torch.arange(self.page_size, dtype=torch.int64, device=self.device)
    dummy_free = dummy_page * self.page_size + page_offsets

    #  这里是首先用getattr看看TokenAllocator是否实现了批量释放的方法
    #  若实现，则用begin_group开启批量释放模式，随后利用free将要释放的page放到
    #  free_group中，随后再用end_group来释放；否则就直接free
    begin_group = getattr(self.token_allocator, "begin_free_group", None)
    end_group = getattr(self.token_allocator, "end_free_group", None)
    if begin_group is not None and end_group is not None:
      begin_group()
      try:
        self.token_allocator.free(dummy_free)
      finally:
        end_group()
    else:
      self.token_allocator.free(dummy_free)

    torch.cuda.synchronize()

  # prefix_for_waiting_req函数通过接收一个req对象
  # 随后通过调用prefix_cache中的种种内容计算该req中在prefill阶段已经命中的prefix cache
  # 计算出该req真正需要 extend = len(prefill) - len(prefix_cache)
  def prefix_for_waiting_req(self, req: "Req"):
    """
    为一个还在 waiting_queue 里的请求寻找可复用的前缀。

    这是 prefix cache 的入口。
    输入是一个“刚被 Scheduler 拿出来准备 prefill 的 Req”，输出后会在 Req 上写入：
    - fill_ids: 当前完整 token 序列
    - prefix_indices: 命中的 prefix KV 物理位置
    - cache_protected_len: 命中的前缀长度
    - last_node: radix tree 中对应的节点
    - extend_input_len: 本轮真正需要新算的 token 数

    你可以把它理解成：
    原始 prompt -> 和 prefix cache 比对 -> 得到“哪些 token 可以直接复用”
    """

    '''
    req.origin_input_ids：原始输入 prompt 的 token ID 列表。
    req.output_ids：该请求已生成的输出 token ID 列表（可能在多轮对话或自回归中非空）。
    两者拼接得到完整的 token 序列，存储在 req.fill_ids 中。这个序列就是请求希望用于生成下一个 token 的全部上下文。
    '''
    req.fill_ids = req.origin_input_ids + req.output_ids
    '''
    len(fill_ids) - 1，因为最后一个 token 是即将生成的 token（或当前要预测的位置），其 KV 尚未计算，因此不能作为前缀的一部分进行复用。前缀缓存只缓存已经计算过的 KV，所以只匹配到倒数第二个 token。
    '''
    max_prefix_len = max(len(req.fill_ids) - 1, 0)
    token_ids = req.fill_ids[:max_prefix_len]
    if self.prefix_cache is not None:
      #  有缓存的情况下，则命中prefix_cache并返回已经命中的token kvcache物理槽位
      #  前缀长度、并保护已经命中的node不被驱逐
      kv_indices, node = self.prefix_cache.match_prefix(token_ids)
      # A request can stay in waiting_queue across scheduler iterations, so this
      # lookup may run repeatedly while the request already holds a cache lock.
      # Keep exactly one lock per request: acquire a changed path before releasing
      # the old one, so shared ancestors remain protected during the transfer.
      previous_node = req.last_node
      if node is not previous_node:
        self.prefix_cache.inc_lock_ref(node)
        if previous_node is not None:
          self.prefix_cache.dec_lock_ref(previous_node)
      req.prefix_indices = kv_indices
      req.cache_protected_len = kv_indices.size(0)
      req.last_node = node
    else:
      # When prefix cache is disabled, still expose an empty prefix tensor for downstream logic.
      req.prefix_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
      req.cache_protected_len = 0
      req.last_node = None
    # A waiting request may be a resumed request whose output history is kept
    # across retract.  Prefix matching tells us which part of that logical
    # history has a reusable KV entry in the current cache; the rest must be
    # recomputed by this extend.  Do not carry the old committed boundary
    # across an eviction or a partial prefix hit.
    req.kv_committed_len = req.cache_protected_len
    req.extend_input_len = len(req.fill_ids) - req.cache_protected_len
    logger.debug(
      f"prefix_for_waiting_req: req_id={req.req_id}, prefix_len={req.cache_protected_len}, extend_input_len={req.extend_input_len}"
    )

  def prepare_for_extend(
    self,
    batch: "ScheduledBatch",
  ):
    """
    为纯 prefill/extend batch 准备所有 KV cache 元数据。

    这个函数是第二周最重要的一个接口之一。它把 Scheduler 选出来的一批请求，
    变成模型 forward 和 attention backend 真正能使用的 batch 元数据。

    主要做的事：
    1. 计算每个请求的 prefix_len / extend_len / seq_len。
    2. 为每个请求分配 RequestPool 槽位。
    3. 为需要新写入的 token 分配 KV 物理位置。
    4. 把 req_pool_idx -> token_pos -> kv_loc 的映射写入 RequestPool。

    注意：
    batch 是可变对象。这个函数会原地修改 batch.input_ids / batch.seq_lens / batch.out_cache_loc 等字段。
    """

    # 先把 batch 标记成 EXTEND 模式。
    # extend_ids 是每个请求本轮真正需要 forward 的新 token。
    batch.forward_mode = ForwardMode.EXTEND
    # 将每个token需要extend的token_index放到extend_ids中去
    extend_ids = [r.fill_ids[len(r.prefix_indices) :] for r in batch.reqs]
    # 计算batch中所有需要extend的token总数
    extend_num_tokens = sum(len(ids) for ids in extend_ids)
    seq_lens = [len(r.fill_ids) for r in batch.reqs]
    prefix_lens = [len(r.prefix_indices) for r in batch.reqs]
    # 每个req需要extend的总数放到一个list中去
    extend_lens = [r.extend_input_len for r in batch.reqs]
    bs = len(batch.reqs)

    # 优化：使用预分配的 pinned memory buffer 进行零拷贝异步传输
    # seq_lens
    self.pinned_seq_lens[:bs] = torch.as_tensor(seq_lens, dtype=torch.int64)
    seq_lens_tensor = self.pinned_seq_lens[:bs].to(self.device, non_blocking=True)
    seq_lens_cpu = self.pinned_seq_lens[:bs].clone()  # CPU 副本

    # extend_ids (flatten)
    # 将所有req需要extend的token_index都展平到一个一维数组中
    extend_ids_flat = [token_id for ids in extend_ids for token_id in ids]
    num_tokens = len(extend_ids_flat)
    self.pinned_input_ids[:num_tokens] = torch.as_tensor(extend_ids_flat, dtype=torch.int64)
    extend_ids_tensor = self.pinned_input_ids[:num_tokens].to(self.device, non_blocking=True)

    batch.prefix_lens = prefix_lens
    batch.extend_lens = extend_lens
    batch.seq_lens = seq_lens_tensor
    batch.seq_lens_cpu = seq_lens_cpu
    batch.input_ids = extend_ids_tensor

    # KV cache related metadata。
    # 每个请求需要一个 RequestPool 槽位，后续通过这个槽位找到它对应的 token->kv 映射表。
    bs = len(batch.reqs)
    prefix_tensors = [r.prefix_indices for r in batch.reqs]
    # Allocate req slots
    req_pool_indices = self.request_pool.alloc(bs)
    if req_pool_indices is None:
      raise RuntimeError(
        f"Failed to allocate request pool slots: "
        f"requested {bs} slots, but request pool is full. "
        f"This usually indicates a scheduling bug where batch size exceeds max_num_seqs."
      )
    # Persist request-pool indices on the Req objects for correct release/update flows.
    for req, pool_idx in zip(batch.reqs, req_pool_indices):
      req.req_pool_idx = int(pool_idx)
    # 优化：使用预分配的 pinned memory buffer
    self.pinned_req_pool_indices[:bs] = torch.as_tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_tensor = self.pinned_req_pool_indices[:bs].to(self.device, non_blocking=True)

    # Allocate KV cache slots。
    # 如果物理空闲页不够，但 prefix cache 里有可驱逐的节点，就先尝试 evict。
    # 检查物理空闲空间，而非 available_tokens()（包含 evictable）
    if self.token_allocator.available_size() < extend_num_tokens and self.prefix_cache is not None:
      needed_to_evict = extend_num_tokens - self.token_allocator.available_size()
      needed_to_evict = (
        (needed_to_evict + self.page_size - 1) // self.page_size * self.page_size
      )  # 向上对齐到 page_size
      logger.debug(
        "prepare_for_extend: need to evict for extend batch, "
        f"extend_num_tokens={extend_num_tokens}, available_size={self.token_allocator.available_size()}, needed_to_evict={needed_to_evict}"
      )
      self.prefix_cache.evict(needed_to_evict)

    # last_loc[i] 表示第 i 个请求复用前缀最后一个 token 的物理 slot
    # 如果这个 page 还没满，allocator 可以从 last_loc + 1 接着分配
    # 这个值用于 alloc_pages_extend() 计算新 token 应该接在哪一页后面。
    last_loc_cpu = [
      (t[-1:] if len(t) > 0 else torch.tensor([-1], dtype=torch.int64, device=self.device))
      for t in prefix_tensors
    ]
    # torch.cat 需要 list/tuple，产生 shape (bs,)
    last_loc_concat = torch.cat(last_loc_cpu)
    # 如果 prefix_tensors 不在 GPU 上，异步传输
    if last_loc_concat.device != self.device:
      last_loc = last_loc_concat.to(self.device, non_blocking=True)
    else:
      last_loc = last_loc_concat

    # 优化：使用预分配的 pinned memory buffer
    self.pinned_prefix_lens[:bs] = torch.as_tensor(batch.prefix_lens, dtype=torch.int64)
    prefix_lens_device = self.pinned_prefix_lens[:bs].to(self.device, non_blocking=True)
    prefix_lens_cpu = self.pinned_prefix_lens[:bs].clone()

    out_cache_loc = self.token_allocator.alloc_pages_extend(
      prefix_lens=prefix_lens_device,
      prefix_lens_cpu=prefix_lens_cpu,
      seq_lens=batch.seq_lens,
      seq_lens_cpu=batch.seq_lens_cpu,
      last_loc=last_loc,
      extend_num_tokens=extend_num_tokens,
    )

    # 检查分配是否成功
    if out_cache_loc is None:
      physical_available = self.token_allocator.available_size()
      logical_available = self.available_tokens()
      raise RuntimeError(
        f"Failed to allocate KV cache for extend batch: "
        f"need {extend_num_tokens} tokens ({bs} requests), "
        f"physical_available {physical_available} tokens (free pages), "
        f"logical_available {logical_available} tokens (includes evictable radix)"
      )

    # 把 extend 长度写入 pinned buffer，再异步送到 GPU。
    # 优化：使用预分配的 pinned memory buffer
    self.pinned_extend_lens[:bs] = torch.as_tensor(extend_lens, dtype=torch.int64)
    extend_lens_device = self.pinned_extend_lens[:bs].to(self.device, non_blocking=True)

    write_cache_indices(
      out_cache_loc,
      req_pool_indices_tensor,
      prefix_lens_device,
      batch.seq_lens,
      extend_lens_device,
      prefix_tensors,
      self.request_pool,
    )

    batch.req_pool_indices = req_pool_indices_tensor
    batch.out_cache_loc = out_cache_loc

  def prepare_for_mixed(
    self,
    batch: "ScheduledBatch",
    extend_reqs: list,
    decode_reqs: list,
    *,
    decode_input_ids: Optional[torch.Tensor] = None,
    decode_seq_lens: Optional[torch.Tensor] = None,
    decode_seq_lens_cpu: Optional[torch.Tensor] = None,
  ):
    """
    为 mixed batch (prefill + decode) 准备 KV cache 和 forward 元数据。

    先建立这几个字段之间的关系：

      RequestPool.req_to_token[req_pool_idx, token_position]
          = physical_kv_slot

    - RequestPool 是整张二维映射表，记录每个请求的每个 token 对应哪个
      物理 KV slot；它不保存 token 内容，也不保存真正的 K/V tensor。
    - req.req_pool_idx 是单个请求在这张表中的行号。decode 请求在进入
      mixed batch 之前已经拥有这个行号；新的 extend 请求需要在本函数中申请。
    - batch.req_pool_indices 是当前 batch 所有请求的行号 tensor，顺序与
      batch.reqs 一致。
    - batch.input_ids 保存 token ID（词表编号），不是 token position。
    - batch.out_cache_loc 保存本轮新 token 的物理 KV slot，并且与 input_ids
      按同一个扁平 token 顺序一一对应。

    Mixed batch 只是在 forward 元数据层面把 decode 请求表示成
    ``prefix_len=已有历史长度, extend_len=1``，从而可以和 prefill 的多个
    token 一起走 extend attention 路径；decode 的物理地址分配仍然使用
    alloc_pages_decode()，extend 的物理地址分配仍然使用 alloc_pages_extend()。

    这是 serving 场景里很实用的一个优化：当有新请求进入系统时，不要阻塞
    已经在 running 的 decode 请求，而是把两类请求放到一轮里一起处理。

    输入：
    - extend_reqs: 新的 prefill/chunked prefill 请求。
    - decode_reqs: 已经在生成中的请求。

    输出：
    - 原地修改 batch，填好 input_ids / seq_lens / prefix_lens / out_cache_loc 等字段。

    最终不变量：
    - 请求维度字段按 ``[extend requests..., decode requests...]`` 排列：
      seq_lens、prefix_lens、extend_lens、req_pool_indices。
    - token 维度字段也按相同的请求顺序展开：
      input_ids、out_cache_loc 都是 ``[extend tokens..., decode tokens...]``。
    - 对每个新 token，都有 ``input_ids[i] <-> out_cache_loc[i]``；模型算出的
      K/V 会写入 out_cache_loc[i]，而 token 的逻辑位置由 prefix_len + offset
      决定，不是由 token ID 决定。

    CPU/GPU 约定：长度和索引通常在 CPU 上保留一份，供调度和 page 数计算使用；
    GPU tensor 供 Triton kernel、attention backend 和模型 forward 使用。并不是
    所有 batch 字段都会同时复制到 CPU 和 GPU，pinned buffer 只是可复用的传输中转区。

    Args:
        batch: 包含所有请求 (extend_reqs + decode_reqs) 的 ScheduledBatch
        extend_reqs: prefill/chunked prefill extend 请求列表
        decode_reqs: 正在 decode 的请求列表（已有 req_pool_idx 和 KV cache）
        decode_input_ids: (overlap 路径) 预设的 decode input_ids（含 -future_indices 占位符），
            不为 None 时跳过 r.output_ids[-1] 读取
        decode_seq_lens: (overlap 路径) decode 请求的 seq_lens（GPU tensor），
            不为 None 时跳过从 len(r.output_ids) 计算
        decode_seq_lens_cpu: (overlap 路径) decode 请求的 seq_lens（CPU tensor）
    """
    if not decode_reqs:
      # 没有 decode 请求，退化为纯 extend
      self.prepare_for_extend(batch)
      return

    # ================== CPU-side metadata build (no CUDA sync) ==================
    # 先在 CPU 上整理元数据，尽量避免在调度阶段触发 GPU->CPU 同步。
    # extend_bs 是新 prefill 请求数量，decode_bs 是已有 running 请求数量。
    extend_bs = len(extend_reqs)
    decode_bs = len(decode_reqs)
    total_bs = extend_bs + decode_bs

    # Extend requests (prefill tokens)
    # 对 extend 请求，真正要输入模型的是“没有被 prefix cache 命中的那段 token”。
    # 例如 prompt 长度 100，prefix 命中 64，则 ids 是后 36 个 token。

    # 收集 extend 请求的元数据（CPU 上）。这些数组的第 i 项对应第 i 个
    # extend request；extend_ids_flat 则把所有请求本轮的新 token 按请求顺序展开。
    extend_seq_lens: list[int] = []
    extend_prefix_lens: list[int] = []
    extend_extend_lens: list[int] = []
    extend_prefix_tensors: list[torch.Tensor] = []
    extend_ids_flat: list[int] = []
    extend_num_tokens = 0
    if extend_bs > 0:
      for r in extend_reqs:
        prefix_len = len(r.prefix_indices)
        ids = r.fill_ids[prefix_len:]
        extend_ids_flat.extend(ids)
        extend_num_tokens += len(ids)
        extend_seq_lens.append(len(r.fill_ids))
        extend_prefix_lens.append(prefix_len)
        extend_extend_lens.append(int(r.extend_input_len))
        extend_prefix_tensors.append(r.prefix_indices)

    # Decode requests: we treat each decode as extend_len=1.
    # mixed batch 中 decode 请求被伪装成 extend_len=1：
    # 它已经有完整历史 KV，本轮只追加一个新 token。
    total_new_tokens = extend_num_tokens + decode_bs

    # Decode input ids: either placeholder (overlap) or last generated tokens.
    # decode 的输入 token 是每个请求“上一轮生成的 token”。
    # overlap 路径下可能先拿到 placeholder，普通路径下直接从 Req.output_ids 取。
    if decode_input_ids is not None:
      decode_input_ids_dev = decode_input_ids.to(self.device, non_blocking=True)
    else:
      # Synchronous/token-based path: gather last tokens on CPU and H2D via pinned buffer.
      for i, r in enumerate(decode_reqs):
        last_token_id = r.get_last_token()
        if last_token_id is None:
          last_token_id = r.output_ids[-1]
        self.pinned_decode_input_ids[i] = int(last_token_id)
        # decode_input_ids 存储的是每个 decode req 上一轮生成的最新 token ID，
        # 它不是该 token 在请求中的 position，也不是 KV cache 的物理 slot。
      decode_input_ids_dev = self.pinned_decode_input_ids[:decode_bs].to(
        self.device, non_blocking=True
      )

    # Decode seq lens metadata must be CPU-known for control flow and to avoid `.item()` sync.
    # Semantics: these are the *current* sequence lengths BEFORE allocating this step's decode token.
    # This function will increment them in-place after allocating the decode KV slot.
    #
    # 这里的 seq_len 是“分配本轮 decode KV 位置之前”的长度。
    # 分配完成后会 add_(1)，表示新 token 已经纳入序列长度。
    if decode_seq_lens_cpu is None:
      # Fallback: build from Req.current_seq_len (CPU-known).
      tmp = torch.empty(decode_bs, dtype=torch.int64)
      for i, r in enumerate(decode_reqs):
        tmp[i] = int(r.current_seq_len)
      decode_seq_lens_cpu = tmp
    else:
      decode_seq_lens_cpu = decode_seq_lens_cpu[:decode_bs]

    if decode_seq_lens is None:
      # Fallback: build a device tensor from CPU lengths via pinned buffer (async H2D).
      #
      # 数据位置要区分清楚：
      # - decode_seq_lens_cpu 是 CPU tensor。
      # - pinned_seq_lens 也是 CPU tensor，只是它位于 pinned（锁页）内存，不是 GPU 显存。
      # - copy_() 把长度写进预分配的 pinned CPU buffer。
      # - to(self.device, non_blocking=True) 再把数据从 pinned CPU 内存异步复制到 GPU。
      #   这是一次复制，不是“移动”或“零拷贝”；CPU 中的两份数据仍然存在。
      #
      # 这里不需要 clone 出新的 CPU 副本，因为 decode_seq_lens_cpu 本身已经是本轮
      # decode 使用的 CPU tensor，后面还会和 GPU 版本一起 add_(1)。
      self.pinned_seq_lens[:decode_bs].copy_(decode_seq_lens_cpu)
      decode_seq_lens = self.pinned_seq_lens[:decode_bs].to(self.device, non_blocking=True)
    else:
      decode_seq_lens = decode_seq_lens[:decode_bs]

    # ================== KV allocations (runs on current CUDA stream) ==================
    # Evict if needed - check physical free pages
    # mixed batch 一共需要 extend_num_tokens + decode_bs 个新 KV token 位置。
    # 如果物理空闲页不够，就从 prefix cache 驱逐一些可回收页。
    if self.token_allocator.available_size() < total_new_tokens and self.prefix_cache is not None:
      needed_to_evict = total_new_tokens - self.token_allocator.available_size()
      needed_to_evict = (needed_to_evict + self.page_size - 1) // self.page_size * self.page_size
      logger.debug(
        "prepare_for_mixed: need to evict, "
        f"total_new_tokens={total_new_tokens}, available_size={self.token_allocator.available_size()}, needed_to_evict={needed_to_evict}"
      )
      self.prefix_cache.evict(needed_to_evict)

    # ---- Decode reqs: allocate 1 token KV slot each, then increment seq_lens in-place ----
    #
    # 先处理 decode 部分，因为这些请求已经有 req_pool_idx，不需要重新分配
    # RequestPool 行。这里的 req_pool_idx 是“行号”，不是新 token 的 KV slot。
    decode_req_pool_indices = [int(r.req_pool_idx) for r in decode_reqs]
    if any(idx < 0 for idx in decode_req_pool_indices):
      raise RuntimeError("prepare_for_mixed: decode_reqs contains invalid req_pool_idx (<0).")

    # H2D req_pool_indices via pinned buffer
    self.pinned_req_pool_indices[:decode_bs] = torch.as_tensor(
      decode_req_pool_indices, dtype=torch.int64
    )
    decode_rpi = self.pinned_req_pool_indices[:decode_bs].to(self.device, non_blocking=True)

    # last_loc for each decode req: position = (current_seq_len - 1)
    # last_loc 是每个 decode 请求当前最后一个 token 的 KV 位置。
    # alloc_pages_decode 会根据 last_loc 和 seq_len 判断本轮是顺序写入，还是需要新 page。
    #
    # allocator 接收追加后的长度；RequestPool 的 last_loc 和 new_pos 仍按旧长度查询。
    # 例如旧长度 256 时，传入 257 才会为第 257 个 token 申请新 page。
    last_loc = self.request_pool.req_to_token_pool()[decode_rpi, decode_seq_lens - 1]
    decode_out_cache_loc = self.token_allocator.alloc_pages_decode(
      seq_lens=decode_seq_lens + 1,
      seq_lens_cpu=decode_seq_lens_cpu + 1,
      last_loc=last_loc,
    )
    if decode_out_cache_loc is None:
      physical_available = self.token_allocator.available_size()
      logical_available = self.available_tokens()
      raise RuntimeError(
        "Failed to allocate KV cache for mixed batch decode part: "
        f"need {decode_bs} tokens ({decode_bs} decode reqs), "
        f"physical_available {physical_available} tokens (free pages), "
        f"logical_available {logical_available} tokens (includes evictable radix)"
      )

    # Update request->kv_loc mapping for the newly allocated decode token:
    # new_pos = current_seq_len (0-indexed position of the new token)
    # 注意：这里 new_pos 使用 add_(1) 之前的 seq_len。
    # 如果当前已有 token 数是 10，新 token 的 0-based position 就是 10；
    # 不是把 input_ids 中的 token ID 当成列下标。
    new_pos = decode_seq_lens
    self.request_pool.req_to_token_pool()[decode_rpi, new_pos] = decode_out_cache_loc.to(
      torch.int32
    )

    # Increment decode seq_lens for forward (now includes the newly allocated token).
    # 此后 decode 请求在统一的 extend 元数据中表示为：
    #   prefix_len = add_(1) 前的旧长度（已有历史 KV 数量）
    #   extend_len = 1（本轮新 token 数量）
    decode_seq_lens.add_(1)
    decode_seq_lens_cpu.add_(1)

    # After increment: prefix_len = seq_len - 1, extend_len = 1
    decode_prefix_lens = (decode_seq_lens_cpu - 1).tolist()
    decode_extend_lens = [1] * decode_bs

    # ---- Extend reqs: allocate req_pool slot + KV cache for all prefill tokens ----
    #
    # 再处理 extend 部分。新请求还没有 req_pool_idx，所以先分配 RequestPool
    # 行号，再分配本轮 token 的物理 KV slot。两种资源不要混淆：
    #   RequestPool 行号：定位“这个请求的地址表”
    #   KV slot：定位“某个 token 的 K/V 实际存储位置”
    extend_req_pool_indices: list[int] = []
    if extend_bs > 0:
      pool_indices = self.request_pool.alloc(extend_bs)
      if pool_indices is None:
        raise RuntimeError(
          f"Failed to allocate request pool slots for extend requests: requested {extend_bs} slots."
        )
      for req, pool_idx in zip(extend_reqs, pool_indices):
        req.req_pool_idx = int(pool_idx)
      extend_req_pool_indices = list(pool_indices)

      # Prepare extend input_ids via pinned buffer (async H2D)
      if extend_num_tokens > 0:
        self.pinned_input_ids[:extend_num_tokens] = torch.as_tensor(
          extend_ids_flat, dtype=torch.int64
        )
        extend_ids_tensor = self.pinned_input_ids[:extend_num_tokens].to(
          self.device, non_blocking=True
        )
      else:
        extend_ids_tensor = torch.empty((0,), dtype=torch.int64, device=self.device)

      # Prepare seq_lens/prefix_lens/extend_lens via pinned buffers (async H2D).
      #
      # 以 extend_seq_lens 为例，完整转换过程是：
      #   Python list
      #     -> torch.as_tensor(...): 临时的普通 CPU tensor
      #     -> 赋值给 pinned_seq_lens[:extend_bs]: 写入 pinned CPU buffer
      #     -> .to(self.device, non_blocking=True): 异步复制出 GPU tensor
      #     -> .clone(): 另外保留一份拥有独立存储的 CPU tensor
      #
      # pinned buffer 是可复用的中转区，下一批请求会覆盖它。因此不能把
      # self.pinned_seq_lens[:extend_bs] 这个 view 直接当作当前 batch 的长期 CPU
      # 元数据；clone() 后的 extend_seq_lens_cpu 才拥有独立存储。
      # 这些长度 tensor 的 shape 都是 [extend_bs]，第 i 个元素对应第 i 个
      # extend request，而不是保存 token 内容的二维数组。
      #
      # 最终几份数据的用途：
      # - extend_seq_lens_device: GPU 上供 Triton kernel / attention metadata 使用。
      # - extend_seq_lens_cpu: CPU 上计算所需新 page 数，避免 GPU -> CPU 同步。
      # - self.pinned_seq_lens: 仅作为后续 batch 还能重复使用的传输缓冲区。
      # prefix_lens 采用相同流程；extend_lens 仍保留原 Python list，因此这里只需
      # 生成 GPU tensor，不再额外 clone 一个 CPU tensor。
      self.pinned_seq_lens[:extend_bs] = torch.as_tensor(extend_seq_lens, dtype=torch.int64)
      extend_seq_lens_device = self.pinned_seq_lens[:extend_bs].to(self.device, non_blocking=True)
      extend_seq_lens_cpu = self.pinned_seq_lens[:extend_bs].clone()

      self.pinned_prefix_lens[:extend_bs] = torch.as_tensor(extend_prefix_lens, dtype=torch.int64)
      extend_prefix_lens_device = self.pinned_prefix_lens[:extend_bs].to(
        self.device, non_blocking=True
      )
      extend_prefix_lens_cpu = self.pinned_prefix_lens[:extend_bs].clone()

      self.pinned_extend_lens[:extend_bs] = torch.as_tensor(extend_extend_lens, dtype=torch.int64)
      extend_lens_device = self.pinned_extend_lens[:extend_bs].to(self.device, non_blocking=True)

      # last_loc for extend alloc: use CPU-built placeholder and async H2D
      last_loc_cpu = [
        (t[-1:] if len(t) > 0 else torch.tensor([-1], dtype=torch.int64, device=self.device))
        for t in extend_prefix_tensors
      ]
      last_loc_concat = torch.cat(last_loc_cpu)
      last_loc = last_loc_concat.to(self.device, non_blocking=True)

      extend_out_cache_loc = self.token_allocator.alloc_pages_extend(
        prefix_lens=extend_prefix_lens_device,
        prefix_lens_cpu=extend_prefix_lens_cpu,
        seq_lens=extend_seq_lens_device,
        seq_lens_cpu=extend_seq_lens_cpu,
        last_loc=last_loc,
        extend_num_tokens=extend_num_tokens,
      )
      if extend_out_cache_loc is None:
        physical_available = self.token_allocator.available_size()
        logical_available = self.available_tokens()
        raise RuntimeError(
          "Failed to allocate KV cache for mixed batch extend part: "
          f"need {extend_num_tokens} tokens, "
          f"physical_available {physical_available} tokens (free pages), "
          f"logical_available {logical_available} tokens (includes evictable radix)"
        )

      # Write prefix + extend indices to request pool。
      # RequestPool 最终会保存：
      #   req_pool_idx, token_position -> physical_kv_location
      # 它把 prefix cache 复用的旧 KV 位置和本轮新分配的位置都写进去。
      # 这里不会写入 token ID；token ID 只用于模型 embedding，RequestPool 只存 KV 地址。
      self.pinned_req_pool_indices[:extend_bs] = torch.as_tensor(
        extend_req_pool_indices, dtype=torch.int64
      )
      extend_rpi = self.pinned_req_pool_indices[:extend_bs].to(self.device, non_blocking=True)

      write_cache_indices(
        extend_out_cache_loc,
        extend_rpi,
        extend_prefix_lens_device,
        extend_seq_lens_device,
        extend_lens_device,
        extend_prefix_tensors,
        self.request_pool,
      )
    else:
      extend_out_cache_loc = torch.empty((0,), dtype=torch.int64, device=self.device)
      extend_ids_tensor = torch.empty((0,), dtype=torch.int64, device=self.device)

    # ================== Batch tensors (forward inputs/metadata) ==================
    # 最后把 extend 部分和 decode 部分拼成一个 batch。
    # 约定顺序是：
    #   [extend requests..., decode requests...]
    # 所以下面的 input_ids / seq_lens / req_pool_indices / out_cache_loc 都按这个顺序排列。
    # input_ids: [extend_tokens..., decode_tokens...]
    # 这是 token ID 的扁平数组，长度是 total_new_tokens；它不是按 request
    # padding 的二维矩阵，且其中每个值也不是 token 在序列中的 position。
    if extend_num_tokens > 0:
      batch.input_ids = torch.empty((total_new_tokens,), dtype=torch.int64, device=self.device)
      batch.input_ids[:extend_num_tokens].copy_(extend_ids_tensor)
      batch.input_ids[extend_num_tokens:].copy_(decode_input_ids_dev)
    else:
      batch.input_ids = decode_input_ids_dev

    # seq_lens: [extend_seq_lens..., decode_seq_lens(after+1)...]
    # 这是“按请求”的数组，长度是 total_bs，不是每个新 token 一个元素。
    if extend_bs > 0:
      # extend_seq_lens_device is defined only when extend_bs>0
      batch.seq_lens = torch.empty((total_bs,), dtype=torch.int64, device=self.device)
      batch.seq_lens[:extend_bs].copy_(extend_seq_lens_device)
      batch.seq_lens[extend_bs:].copy_(decode_seq_lens)
      seq_lens_cpu = torch.empty((total_bs,), dtype=torch.int64)
      seq_lens_cpu[:extend_bs] = torch.as_tensor(extend_seq_lens, dtype=torch.int64)
      seq_lens_cpu[extend_bs:] = decode_seq_lens_cpu
      batch.seq_lens_cpu = seq_lens_cpu
    else:
      batch.seq_lens = decode_seq_lens
      batch.seq_lens_cpu = decode_seq_lens_cpu

    batch.prefix_lens = extend_prefix_lens + decode_prefix_lens
    batch.extend_lens = extend_extend_lens + decode_extend_lens

    # req_pool_indices: per-seq mapping。
    # 每个请求一个 RequestPool 行号；顺序必须和 batch.reqs、seq_lens 等一致。
    all_req_pool_indices = extend_req_pool_indices + decode_req_pool_indices
    self.pinned_req_pool_indices[:total_bs] = torch.as_tensor(
      all_req_pool_indices, dtype=torch.int64
    )
    batch.req_pool_indices = self.pinned_req_pool_indices[:total_bs].to(
      self.device, non_blocking=True
    )

    # out_cache_loc: per-token mapping aligned with input_ids。
    # 这是本轮每个新 token 的物理 KV slot，和 input_ids 逐元素对应：
    #   input_ids[i]       -> token ID
    #   out_cache_loc[i]   -> 该 token 的 K/V 写入位置
    # 但该 token 的 token_position 仍要根据所属请求的 prefix_len + offset
    # 计算，再写入 RequestPool。
    if extend_num_tokens > 0:
      batch.out_cache_loc = torch.empty((total_new_tokens,), dtype=torch.int64, device=self.device)
      batch.out_cache_loc[:extend_num_tokens].copy_(extend_out_cache_loc)
      batch.out_cache_loc[extend_num_tokens:].copy_(decode_out_cache_loc)
    else:
      batch.out_cache_loc = decode_out_cache_loc

    return

  def prepare_for_decode(self, batch: "ScheduledBatch", skip_input_ids: bool = False):
    """
    为 decode batch 分配下一步要写入的 KV 位置。

    decode 阶段每个请求只输入 1 个 token，这个 token 通常就是“上一步生成的 token”。
    这里要特别注意采样和 KV 写入之间相差一轮：

      第 N-1 轮采样出 token 99
        -> 99 被追加到 req.output_ids，但此时还没有 99 的 K/V
      第 N 轮 prepare_for_decode
        -> input_ids 中放入 99
        -> out_cache_loc 为 99 分配一个物理 KV slot
        -> 本轮模型 forward 才计算并写入 99 的 K/V，同时采样下一个 token

    因此 out_cache_loc[i] 对应的不是“本轮即将采样出的 token”，而是：
    第 i 个请求上一轮已经采样出、本轮作为 input_ids[i] 送入模型的 token。
    prepare_for_decode 结束时，该 slot 只是完成了分配和映射登记；真实 K/V 会在
    后续每一层 attention forward 中写入 MHAKVCacheStorage。

    例：某请求的 prompt 为 [10, 11, 12]，prefill 后采样出 token 99。

      调用前：
        req.output_ids = [99]
        batch.seq_lens = [3]                 # 已经具有 K/V 的 token 数量
        batch.req_pool_indices = [2]         # 该请求占 RequestPool 第 2 行
        req_to_token[2, 0:3] = [4, 5, 6]    # prompt 三个 token 的物理 KV slot

      本轮准备结果（假设旧 page 还有一个空 slot）：
        batch.input_ids = [99]               # token ID，不是物理地址
        batch.out_cache_loc = [7]            # token 99 的物理 KV slot
        batch.seq_lens = [4]
        req_to_token[2, 3] = 7               # token position 3 -> physical slot 7

      随后的第 N 轮 forward 对每一层执行等价操作：
        k_buffer[layer_id][7] = token 99 在该层计算出的 K
        v_buffer[layer_id][7] = token 99 在该层计算出的 V

      第 N 轮 forward 再采样出的 token（例如 108）此时仍没有 K/V；它将在
      第 N+1 轮成为 input_ids，并获得自己的 out_cache_loc。

    函数会：
    1. 准备 input_ids（除非 skip_input_ids=True，用于 overlap placeholder 路径）。
    2. 根据当前 seq_lens 判断本轮是否跨页，需要分配多少新 KV 位置。
    3. 更新 RequestPool 中 token_pos -> kv_loc 的映射。
    4. seq_lens += 1，因为这一轮 decode 后序列长度会多一个 token。

    skip_input_ids=True 时，batch.input_ids 由上层提前准备好，避免 schedule 依赖上一轮
    process 的输出写回。
    """
    batch.forward_mode = ForwardMode.DECODE
    # Decode 阶段的 input_ids 是每个请求最后生成的 token
    # 从每个 req.output_ids[-1] 获取
    bs = len(batch.reqs)
    if bs == 0:
      return

    if not skip_input_ids:
      # 非 overlap 路径：从 CPU 侧收集 last_token，并通过 pinned buffer 异步 H2D。
      # 这避免了 Python 循环逐元素写 GPU tensor（会产生大量 tiny kernel/allocator 交互）。
      for i, req in enumerate(batch.reqs):
        last_token = req.get_last_token()
        if last_token is None:
          # 回退（理论上不应该发生）
          last_token = req.output_ids[-1]
        self.pinned_decode_input_ids[i] = int(last_token)
      batch.input_ids = self.pinned_decode_input_ids[:bs].to(self.device, non_blocking=True)
    else:
      batch.input_ids = batch.output_ids
    batch.output_ids = None
    token_per_req = 1  # decode

    # Allocate KV cache slots - 检查物理空闲空间
    if self.token_allocator.available_size() < bs * token_per_req and self.prefix_cache is not None:
      needed_to_evict = (bs * token_per_req) - self.token_allocator.available_size()
      needed_to_evict = (
        (needed_to_evict + self.page_size - 1) // self.page_size * self.page_size
      )  # 向上对齐到 page_size
      logger.debug(
        "prepare_for_decode: need to evict for decode batch, "
        f"num_tokens={bs * token_per_req}, available_size={self.token_allocator.available_size()}, needed_to_evict={needed_to_evict}"
      )
      self.prefix_cache.evict(needed_to_evict)

    last_loc = self.request_pool.req_to_token_pool()[batch.req_pool_indices, batch.seq_lens - 1]
    out_cache_loc = self.token_allocator.alloc_pages_decode(
      seq_lens=batch.seq_lens + 1,
      seq_lens_cpu=batch.seq_lens_cpu + 1,  # allocator expects the post-append length
      last_loc=last_loc,
    )

    # 检查分配是否成功
    if out_cache_loc is None:
      physical_available = self.token_allocator.available_size()
      logical_available = self.available_tokens()
      raise RuntimeError(
        f"Failed to allocate KV cache for decode batch: "
        f"need {bs} tokens ({bs} requests), "
        f"physical_available {physical_available} tokens (free pages), "
        f"logical_available {logical_available} tokens (includes evictable radix)"
      )

    # out_cache_loc 与 input_ids 按请求一一对应。这里仅登记本轮输入 token 的
    # 物理 KV 写入地址；真实 K/V 要等 attention forward 计算后才写入该地址。
    batch.out_cache_loc = out_cache_loc
    batch.seq_lens.add_(1)
    batch.seq_lens_cpu.add_(1)
    # Update request -> kv_loc mapping for the newly allocated decode token.
    # This is needed to build correct block/page tables for flash-attn backends.
    new_pos = batch.seq_lens - 1
    self.request_pool.req_to_token_pool()[batch.req_pool_indices, new_pos] = out_cache_loc.to(
      torch.int32
    )

  def release_request(
    self,
    req: Req,
    is_insert: bool = True,
  ):
    """
    结束一个请求对 RequestPool 和 KV cache 的占用，并选择性保留 prefix cache。

    这个函数并不是简单地把请求的所有 KV page 都归还 allocator。启用
    RadixCache 时，完整 page 中仍可复用的部分会从“请求持有”转为
    “RadixCache 持有”；只有重复副本和末尾不完整 page 才会立即释放。

    先区分两个时间点的前缀长度：

      C = req.cache_protected_len
        请求开始计算前，prefix_for_waiting_req() 已经从 RadixCache 命中并
        调用 inc_lock_ref() 保护起来的前缀长度。

      M = prefix_cache.insert(...) 返回的 new_prefix_len
        请求结束、执行 insert() 的这一刻，RadixCache 中已经存在并匹配的
        前缀长度。这个名字容易误解，它不是插入后的总长度。

    通常 C <= M。M 可能大于 C，因为当前请求运行期间，另一个请求可能先把
    相同的后续 token/KV 插入 RadixCache；也可能那段 cache 之前就存在，但
    当前请求开始时尚不知道之后会生成哪些 token，因此当时无法匹配。

    设 A = page_aligned_len，N = num_tokens，则 kv_indices 分为四段：

      [0, C):
        请求开始前已经命中的 RadixCache KV。RequestPool 和 RadixCache 指向
        同一批物理 slot，并不存在请求自己的另一份副本。因此不能 free；请求
        结束时只需 dec_lock_ref()，表示不再阻止这些 cache 被驱逐。

      [C, M):
        当前请求自己分配和计算，但释放时发现 RadixCache 已经持有另一份的
        重复 KV。这里要 free 当前请求的物理副本。当前请求从未 lock 释放时
        才发现的那些 RadixCache 节点，所以不能对它们 dec_lock_ref()。

      [M, A):
        RadixCache 中原本不存在，本次 insert() 用当前请求的物理 slot 创建
        新节点。这些 page 的所有权转给 RadixCache，不能 free。

      [A, N):
        末尾不足一个完整 page 的部分。page 模式的 RadixCache 不保存 partial
        page，因此把该 page 归还 allocator。

    例如 page_size=4，C=4、M=8、A=N=12：请求原先共享 [0,4) 的 cache，
    又独立计算了 [4,12)。释放时，[4,8) 已经和别的 cache 重复，所以释放
    当前请求的副本；[8,12) 成为新 RadixCache 节点；最后只解除 [0,4)
    原始命中路径的 lock。

    注意 lock_ref 是“活跃请求使用保护计数”，不是物理内存所有权计数。
    lock_ref 降到 0 只表示节点可以被 LRU 驱逐，KV 仍保留在 RadixCache；
    之后真正执行 prefix_cache.evict() 时才会把对应 page 归还 allocator。

    is_insert=False 时不创建新的 RadixCache 节点，只释放当前请求自己拥有的
    kv_indices[C:]；原先共享的 [0,C) 仍不能释放，最后同样解除其 lock。

    Args:
        req: 要退出运行态的请求，其中保存 RequestPool 行号、完整 token 序列、
            初始命中的 prefix 长度和对应的 RadixCache 节点。
        is_insert: 是否在释放前把完整 page 尽量插入 RadixCache。False 表示
            不保留本次请求新计算出的 cache。
    """
    # 必须先确认 RequestPool 行仍然有效。无效行通常表示请求已经释放过；继续读取
    # 可能取得其他请求的映射，并造成同一物理 page 被重复释放。
    req_pool_idx = int(getattr(req, "req_pool_idx", -1))
    if req_pool_idx < 0:
      raise RuntimeError(
        f"release_request called with invalid req_pool_idx={req_pool_idx} "
        f"for req_id={req.req_id}. This indicates request lifecycle metadata corruption."
      )
    # Only tokens whose KV has completed a forward may be handed to the radix
    # cache.  The latest sampled token is part of the logical history but does
    # not have a KV entry until it is consumed by the next decode forward.
    token_ids = req.committed_token_ids
    # A few low-level legacy tests allocate a request and call release directly
    # without simulating the forward.  They have no generated output, so using
    # the full prompt preserves the old allocator contract without weakening
    # the real retract path (which always has an explicit committed boundary).
    if not token_ids and not req.output_ids and req.req_pool_idx >= 0:
      token_ids = req.origin_input_ids
    num_tokens = len(token_ids)
    # 在释放 RequestPool 行之前，先取出：
    #   kv_indices[token_position] = physical_kv_slot
    # 后面 RadixCache.insert() 和 allocator.free() 都依赖这份地址列表。
    kv_indices = self.request_pool.read(req_pool_idx, slice(0, num_tokens))

    if is_insert and self.prefix_cache is not None:
      # page 模式下 RadixCache 只接管完整 page。向下对齐后的尾部 [A,N)
      # 属于当前请求的 partial page，不能进入共享 cache。
      page_aligned_len = (num_tokens // self.page_size) * self.page_size

      # 插入到前缀缓存（会截断到 page 对齐）
      # insert 返回 M：插入之前，RadixCache 已存在并匹配的 prefix 长度。
      # 对 [M,A)，insert 会把当前请求的 kv_indices 保存到新建节点中。
      new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)

      # 四段所有权：
      # - [0,C):   与原 RadixCache 共享，只在函数末尾解除 lock，不能 free。
      # - [C,M):   当前请求独立计算的重复副本，需要 free。
      # - [M,A):   被新 RadixCache 节点接管，不能 free。
      # - [A,N):   未进入 RadixCache 的 partial page，需要 free。

      # 释放 [C,M) 中“当前请求自己的物理副本”。RadixCache 已有节点保存的是
      # 另一组物理 slot，不会被这次 free 影响。
      if new_prefix_len > req.cache_protected_len:
        self.token_allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])

      # 释放 [A,N) 的尾部。PagedTokenAllocator.free() 会将 slot id 除以
      # page_size 后回收整个 page；partial page 不与其他请求共享，所以这样是安全的。
      if page_aligned_len < num_tokens:
        self.token_allocator.free(kv_indices[page_aligned_len:])
    else:
      # 不插入新 cache 时，只释放请求自己拥有的 [C,N)。[0,C) 是从已有
      # RadixCache 借用的物理 page，仍需保留。
      self.token_allocator.free(kv_indices[req.cache_protected_len :])

    # 这里只释放 RequestPool 的“行号”，不会释放 KV page，也不会清零这一行的旧值。
    # 新请求以后可以复用该行，并在自己的有效 token range 内覆盖旧映射。
    self.request_pool.free([req_pool_idx])  # param is list
    if self.prefix_cache is not None:
      # prefix_for_waiting_req() 曾对 req.last_node 及其祖先 inc_lock_ref()。
      # 此处只撤销当前请求对“开始前命中的 [0,C) 路径”的使用保护。
      # dec_lock_ref() 不释放物理 page；引用降到 0 后节点只是变为可驱逐状态。
      self.prefix_cache.dec_lock_ref(req.last_node)

  def update_finished_req_radix_cache(self, req: Req, is_insert: bool = True):
    """
    处理“自然完成”的请求。

    这里和 release_request 很像，但语义上更偏向“请求正常结束”：
    - 先把 token 序列插入 prefix cache。
    - 再释放不需要的 KV page。
    - 释放 request pool 槽位。

    对初学者可以先把它和 release_request 看成同一类释放接口。
    release_request()
    主要用于 retract、预热清理和直接释放

    update_finished_req_radix_cache()
    用于正常完成请求的 pending_release 批量释放路径
    """
    if int(getattr(req, "req_pool_idx", -1)) < 0:
      raise RuntimeError(
        f"update_finished_req_radix_cache called with invalid req_pool_idx="
        f"{getattr(req, 'req_pool_idx', -1)} for req_id={req.req_id}"
      )
    token_ids = req.committed_token_ids
    if not token_ids and not req.output_ids and req.req_pool_idx >= 0:
      token_ids = req.origin_input_ids
    num_tokens = len(token_ids)
    kv_indices = self.request_pool.read(req.req_pool_idx, slice(0, num_tokens))

    if is_insert and self.prefix_cache is not None:
      # 计算 page 对齐长度
      page_aligned_len = (num_tokens // self.page_size) * self.page_size

      # 插入到前缀缓存
      new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)

      # 同 release_request 的释放逻辑
      # 释放重复部分（如果有）
      if new_prefix_len > req.cache_protected_len:
        self.token_allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])

      # 释放被截断的部分（如果有）
      if page_aligned_len < num_tokens:
        self.token_allocator.free(kv_indices[page_aligned_len:])
    else:
      # 没有 prefix cache，释放所有未保护的部分
      self.token_allocator.free(kv_indices[req.cache_protected_len :])

    self.request_pool.free([req.req_pool_idx])  # param is list
    if self.prefix_cache is not None:
      self.prefix_cache.dec_lock_ref(req.last_node)

  def update_unfinished_req_radix_cache(self, req: Req):
    """
    保存 chunked prefill 的一个中间 chunk，并推进请求的缓存边界。

    调用时机：当前 chunk 已完成 forward，K/V 已写入物理 KV slots，但整个
    prompt 还没有 prefill 完。调用方随后会释放本轮临时 RequestPool 行，因此
    本函数必须先把当前 chunk 可缓存的 KV 交给 RadixCache，并把 Req 改成下一轮
    可以继续使用的新前缀状态。关闭 prefix cache 时，把映射保存在 Req 上，
    KV 仍由当前请求私有持有，最终完成时全部释放。

    进入函数时，令：
      C = req.cache_protected_len             # 原来已命中并锁住的前缀长度
      N = len(req.fill_ids)                   # 截至当前 chunk 的已计算长度
      M = prefix_cache.insert(...) 的返回值   # 插入前树中已存在的前缀长度
      A = len(new_indices)                    # page 对齐后的新缓存长度

    kv_indices[0:N] 的所有权可以分成四段：
      [0, C): 已由 RadixCache 持有并受旧 lock 保护，不能 free。
      [C, M): 本请求新算、但插入时发现树中已有的重复副本，应当 free。
      [M, A): 本次新插入的 KV，生命周期已交给 RadixCache，不能 free。
      [A, N): 不足一个完整 page、没有进入 RadixCache 的尾部。

    中间 chunk 应由 Scheduler 保证 page 对齐，因此正常情况下 A == N；否则
    [A, N) 不能随着 RequestPool 行一起丢失，必须由调用方另行保留或释放。

    ``match_prefix`` 返回的 new_indices 不是新分配的 KV，而是 RadixCache 最终
    采用的 canonical 物理地址：它可能混合其他请求先插入的 slots，以及本请求
    刚转交给树的 slots。共享是安全的，因为缓存前缀只读，并且每个活跃请求都会
    对对应路径增加 lock_ref。一个请求结束只减少自己的引用；只有 lock_ref == 0
    的节点才可能在后续 evict 时真正归还物理 page。

    函数完成后：
      - req.prefix_indices 指向新的 canonical KV 地址；
      - req.cache_protected_len 推进到 A；
      - lock 从旧 last_node 迁移到新 last_node；
      - 后续释放逻辑从新的 A 开始处理私有 KV，不会再次释放 [C, M)。
    """
    if self.prefix_cache is None:
      # 调用方会释放临时 RequestPool 行，因此把已完成 chunk 的地址表保存在 Req。
      # 这些 KV 仍是请求私有的；cache_protected_len 保持 0，结束时必须全部归还。
      req.prefix_indices = self.request_pool.read(req.req_pool_idx, slice(0, len(req.fill_ids)))
      req.mark_kv_committed(len(req.fill_ids))
      return

    # fill_ids 在这里不是完整 prompt，而是“旧缓存前缀 + 当前已完成 chunk”。
    token_ids = req.fill_ids

    # RequestPool 保存 token_position -> physical_kv_slot；这里读取的只是物理
    # slot 编号，不是真正的 K/V tensor 内容。
    kv_indices = self.request_pool.read(req.req_pool_idx, slice(0, len(token_ids)))

    # insert() 返回“插入前已存在的前缀长度 M”。[M:A) 会作为新节点引用
    # 当前请求计算出的 slots，从这一刻起由 RadixCache 管理其生命周期。
    new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)

    # 只释放 [C:M) 中当前请求的重复副本。树中已经存在的 canonical slots
    # 不在这个切片里，因此不会影响仍在使用同一前缀的其他请求。
    self.token_allocator.free(
      kv_indices[req.cache_protected_len : new_prefix_len]
    )

    # 重新查询整段前缀，取得树最终采用的 canonical 地址和末端节点。若另一请求
    # 抢先插入了相同前缀，这里的部分地址会来自那个已存在的 RadixCache 节点。
    new_indices, new_last_node = self.prefix_cache.match_prefix(token_ids)

    # [0:C) 原本就是 canonical 地址，只需把 C 之后的映射改成最新结果。
    # 调用方很快会释放这行；长期状态保存在 RadixCache 和 req.prefix_indices 中。
    self.request_pool.write(
      req.req_pool_idx,
      slice(req.cache_protected_len, len(new_indices)),
      new_indices[req.cache_protected_len :],
    )

    # 迁移的是当前请求持有的那一份引用。若其他请求仍在使用同一节点，
    # 它们自己的 lock_ref 仍然存在，物理 page 不会因此被驱逐。
    self.prefix_cache.dec_lock_ref(req.last_node)
    self.prefix_cache.inc_lock_ref(new_last_node)

    # cache_protected_len 的推进同时也是后续释放的所有权边界：最终释放请求时，
    # [0:A) 不会交给 token_allocator.free()，因此不会发生重复释放。
    req.cache_protected_len = len(new_indices)
    req.prefix_indices = new_indices
    req.last_node = new_last_node
    # The temporary request-pool row is about to be released by the caller,
    # so only the canonical prefix now owned by RadixCache remains reusable.
    req.mark_kv_committed(req.cache_protected_len)

  # =============== Attention forward calling ================
  def get_page_table(self, forward_batch: ForwardBatch, seq_len: torch.Tensor) -> torch.Tensor:
    """
    为 attention backend 取出当前 batch 的 page table。

    page table 可以先理解成：
    “某个请求的第 i 个 token 对应哪个物理 KV 位置”的查表结果。

    FlashAttention2 / FlashInfer 在 decode/paged attention 时都需要这个映射。
    """
    return self.request_pool.req_to_token_pool()[forward_batch.req_pool_indices, :seq_len]

  def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取指定层的 KV buffer"""
    return self.storage.get_kv_buffer(layer_id)

  def set_kv_buffer(
    self,
    layer_id: int,
    loc: torch.Tensor,
    cache_k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    cache_v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
  ):
    """写入 KV cache"""

    self.storage.set_kv_buffer(layer_id, loc, cache_k, cache_v)

  def available_tokens(self) -> int:
    """
    返回逻辑上还能分配多少 token 槽位。

    这里不是简单返回 free_pages 的数量，而是：
    - 当前物理空闲页
    - prefix cache 中可驱逐回收的页

    所以它表示“如果需要，系统理论上还能再容纳多少 token”。
    """
    available = (
      self.token_allocator.available_size() + self.prefix_cache.evictable_size()
      if self.prefix_cache is not None
      else self.token_allocator.available_size()
    )
    # Logical availability can never exceed total KV capacity.
    if available > self.size:
      logger.warning(
        "available_tokens overflow detected: available=%d > size=%d; clamping",
        available,
        self.size,
      )
      return self.size
    return available

  def can_allocate(self, num_tokens: int) -> bool:
    """检查逻辑可用量是否足够分配指定数量的 token。"""
    return self.available_tokens() >= num_tokens

  def get_stats(self) -> dict:
    """获取统计信息"""
    return {
      "size": self.size,
      "available_tokens": self.available_tokens(),
      "used_tokens": self.size - self.available_tokens(),
      "utilization": 1.0 - self.available_tokens() / self.size,
      "kv_cache_bytes": self.storage.get_kv_size_bytes(),
    }

# 维护每个请求的request_to_token二维表
# 维护工作是：将每个请求的RadixCache中命中的prefix slots + 本轮新分配的extend slots
# 维护到二维表中
@triton.jit
def write_req_to_token_pool_triton(
  req_to_token_ptr,  # [max_batch, max_context_len]
  req_pool_indices,  # 每个请求的行号
  prefix_tensors,    # 每个prefix tensor的GPU地址
  pre_lens,          # 每个请求的前缀长度
  seq_lens,          # 每个请求的prefix + 本轮extend后的总长度
  extend_lens,       # 每个请求本轮新增的token数
  out_cache_loc,     # 每个请求本次extend新增的slots
  req_to_token_ptr_stride: tl.constexpr,  # requestpool每一行的长度
):
  # 这个 kernel 只维护映射表，不计算 attention，也不写 K/V tensor。
  #
  # 对每个请求，它会写两段映射：
  # 1. prefix 部分：来自 prefix cache 命中的旧 KV indices。
  # 2. extend 部分：来自本轮 allocator 新分配的 out_cache_loc。
  #
  # 写完后 RequestPool 中就有：
  #   req_to_token[req_pool_index, token_pos] = physical_kv_location
  BLOCK_SIZE: tl.constexpr = 512
  pid = tl.program_id(0)

  req_pool_index = tl.load(req_pool_indices + pid)
  pre_len = tl.load(pre_lens + pid)
  seq_len = tl.load(seq_lens + pid)
  prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

  # 写 prefix 部分：这些 token 不需要重新计算 K/V，但 attention 仍然要能找到它们。
  num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
  for i in range(num_loop):
    offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
    mask = offset < pre_len
    value = tl.load(prefix_tensor + offset, mask=mask)
    tl.store(
      req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
      value,
      mask=mask,
    )

  # NOTE: This can be slow for large bs
  # 计算当前请求的 extend 部分在 out_cache_loc 这个一维数组中的起始偏移。
  # out_cache_loc 的布局是 [req0_extend..., req1_extend..., ...]。
  cumsum_start = tl.cast(0, tl.int64)
  for i in range(pid):
    cumsum_start += tl.load(extend_lens + i)

  num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
  for i in range(num_loop):
    offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
    mask = offset < (seq_len - pre_len)
    value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
    tl.store(
      req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset + pre_len,
      value,
      mask=mask,
    )


def write_cache_indices(
  out_cache_loc: torch.Tensor,
  req_pool_indices_tensor: torch.Tensor,
  prefix_lens_tensor: torch.Tensor,
  seq_lens_tensor: torch.Tensor,
  extend_lens_tensor: torch.Tensor,
  prefix_tensors: list[torch.Tensor],
  req_to_token_pool: RequestPool,
):
  """
  把一批请求的 token_pos -> kv_loc 映射写入 RequestPool。

  这个函数通常由 prepare_for_extend / prepare_for_mixed 调用。
  对每个请求：
  - prefix_lens 之前的 token 使用 prefix_tensors 中的旧 KV 位置。
  - prefix_lens 之后的新增 token 使用 out_cache_loc 中的新 KV 位置。

  初学时可以把它理解成“把逻辑序列长度和物理 KV 地址对齐”的最后一步。
  """
  # Triton kernel 不能直接接收 Python list[Tensor]。
  # 所以这里先把每个 prefix tensor 的 data_ptr() 收集成一个 GPU tensor，
  # kernel 内再按请求 id 取出对应指针。
  prefix_pointers = torch.tensor(
    [t.data_ptr() for t in prefix_tensors],
    device=req_to_token_pool.device,
    dtype=torch.uint64,
  )
  # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
  write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
    req_to_token_pool.req_to_token_pool(),
    req_pool_indices_tensor,
    prefix_pointers,
    prefix_lens_tensor,
    seq_lens_tensor,
    extend_lens_tensor,
    out_cache_loc,
    req_to_token_pool.req_to_token_pool().shape[1],
  )
