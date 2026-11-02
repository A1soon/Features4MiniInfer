import torch
from enum import Enum
import numpy as np
from typing import Optional, Union, List, Tuple, Any
from miniinfer.utils import get_logger
from .interface import IKVCacheStorage, ITokenAllocator, IRequestPool
from kernels.triton.store import store_kv_cache
import triton
import triton.language as tl

logger = get_logger(__name__)
GB = 1024 * 1024 * 1024


def next_power_of_2(n: int):
  return 1 << (n - 1).bit_length() if n > 0 else 1


class MHAKVCacheStorage(IKVCacheStorage):
  """
  物理 KV 存储。

  可以把它理解成“真正装 K/V tensor 的大显存数组”。上层不会用 token id
  直接索引这里，而是先通过 allocator / RequestPool 把“某个 request 的第几个
  token”翻译成“物理 KV slot id”，再用 slot id 读写 K/V。
  """

  def __init__(
    self,
    size: int,
    page_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
  ):
    self.size = size
    self.page_size = page_size
    self.num_layers = num_layers
    self.device = device

    # 为每一层各分配一份 K / V buffer。
    #
    # shape = [size + page_size, num_heads, head_dim]
    # size 是可用 KV token slot 数；额外的 page_size 对应保留的 page 0。
    # PagedTokenAllocator.free_pages 从 page 1 开始，所以合法 page id 是
    # [1, size // page_size]，最后一个 page 会索引到 size + page_size - 1。
    self.k_buffer = [
      torch.zeros((size + self.page_size, num_heads, head_dim), dtype=dtype, device=device)
      for _ in range(num_layers)
    ]
    self.v_buffer = [
      torch.zeros((size + self.page_size, num_heads, head_dim), dtype=dtype, device=device)
      for _ in range(num_layers)
    ]

  def get_kv_buffer(self, layer_id: int):
    # 返回某一层对应的 (K buffer, V buffer)；具体读写哪几个 token slot
    # 由 allocator 给出的 loc 决定。
    return self.k_buffer[layer_id], self.v_buffer[layer_id]

  def set_kv_buffer(
    self,
    layer_id: int,
    loc: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
  ):
    # loc 是物理 KV 位置索引，已经由 allocator 计算好。
    # 这里不再做 request/token 级别判断，只负责把本层新算出的 K/V 写入对应 slot。
    #
    store_kv_cache(
      self.k_buffer[layer_id],
      self.v_buffer[layer_id],
      loc,
      cache_k,
      cache_v,
    )

  def get_kv_size_bytes(self) -> int:
    """统计所有层的 KV cache 总字节数。"""
    total_bytes = 0
    for k, v in zip(self.k_buffer, self.v_buffer):
      total_bytes += k.numel() * k.element_size()
      total_bytes += v.numel() * v.element_size()
    return total_bytes


class TokenAllocator(ITokenAllocator):
  """
  一个最朴素的 token allocator。

  它维护的是空闲 token slot id 列表 free_slots，不是 page id。这个类更像
  参考实现：按 token 直接线性分配，不考虑 page 复用。真正用于 page 化调度的是
  下面的 PagedTokenAllocator。
  """

  def __init__(self, size: int, device: str):
    self.size = size
    self.device = device
    # slot 0 保留，从 1 开始分配；free_slots 中每个元素都是一个物理 KV slot id。
    self.free_slots = torch.arange(1, size + 1, dtype=torch.int64, device=device)

  def alloc(self, num_tokens: int) -> Optional[torch.Tensor]:
    if num_tokens > len(self.free_slots):
      return None
    allocated = self.free_slots[:num_tokens]
    self.free_slots = self.free_slots[num_tokens:]
    return allocated

  def free(self, indices: torch.Tensor):
    if indices.numel() > 0:
      self.free_slots = torch.cat([self.free_slots, indices])

  def available_size(self) -> int:
    return len(self.free_slots)

  def alloc_pages_extend(
    self,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
  ) -> Optional[torch.Tensor]:
    raise NotImplementedError()

  def alloc_pages_decode(
    self,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
  ) -> Optional[torch.Tensor]:
    raise NotImplementedError()


@triton.jit
def alloc_extend_kernel(
  pre_lens_ptr,
  seq_lens_ptr,
  last_loc_ptr,
  free_page_ptr,
  out_indices,
  bs_upper: tl.constexpr,
  page_size: tl.constexpr,
  max_num_extend_tokens: tl.constexpr,
):
  # 这个 kernel 的任务不是写 K/V，而是计算“本轮 extend token 应该写到哪些 KV 位置”。
  # Python 侧传进来的 out_indices 后面会被命名为 out_cache_loc，并交给 attention
  # backend 写真实 K/V：k_buffer[layer][out_cache_loc] = new_k。
  #
  # 每个 program 处理一个请求。假设一个请求已有 prefix_len 个 token，
  # 本轮要把序列扩展到 seq_len，那么新增 token 的物理位置可能分成三段：
  # 1. 旧 partial page 剩余 slot。
  # 2. 新完整 page 展开后的 slot。
  # 3. 新 partial page 展开后的 slot。
  pid = tl.program_id(0)

  # 为了计算当前请求在扁平 out_indices 中的起始偏移，
  # 需要知道它前面所有请求的 extend_len 之和。这里 load_offset <= pid 只加载
  # 当前请求及其之前的请求；后面的 lane 被 mask 掉，不参与 tl.sum 的有效结果。
  load_offset = tl.arange(0, bs_upper)
  seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
  pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
  extend_lens = seq_lens - pre_lens

  seq_len = tl.load(seq_lens_ptr + pid)
  pre_len = tl.load(pre_lens_ptr + pid)
  extend_len = seq_len - pre_len

  # output_start_loc 是当前请求在 out_indices 中写入的开始位置。
  # out_indices 的布局是把所有请求的新增 token 位置拼成一维数组。
  #
  #   out_indices = [req0_extend_locs..., req1_extend_locs..., ...]
  sum_extend_lens = tl.sum(extend_lens)
  output_start_loc = sum_extend_lens - extend_len

  # 计算本轮每个请求需要从 free_page_ptr 中消耗多少新 page。
  # 注意这里是“每个请求各自 ceil(seq_len / page_size)”，不能先把 batch 的
  # seq_lens 求和再除以 page_size；每个 request 都有自己的 page 边界。
  num_pages_after = (seq_lens + page_size - 1) // page_size
  num_pages_before = (pre_lens + page_size - 1) // page_size
  num_new_pages = num_pages_after - num_pages_before

  # 当前请求自己要消耗的新 page 数。
  num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
    pre_len + page_size - 1
  ) // page_size

  # free_page_ptr 是一维 free page id 数组。为了让每个请求拿到不重叠的一段
  # free pages，需要知道“当前请求之前的请求一共消耗了多少新 page”。
  #
  # sum_num_new_pages 是 req0..req_pid 的新 page 数总和；减去当前请求自己的数量，
  # 就得到当前请求在 free_page_ptr 中的起始 page 下标。
  sum_num_new_pages = tl.sum(num_new_pages)
  new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

  # Part 1: fill the old partial page
  # 如果 prefix 的最后一个 token 所在 page 还没满，新增 token 先顺着 last_loc
  # 往后写，不需要从 free_page_ptr 拿新 page。
  last_loc = tl.load(last_loc_ptr + pid)
  num_part1 = min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
  offset_one_page = tl.arange(0, page_size)
  tl.store(
    out_indices + output_start_loc + offset_one_page,
    last_loc + 1 + offset_one_page,
    mask=offset_one_page < num_part1,
  )
  if pre_len + num_part1 == seq_len:
    return

  # Part 2: fill the new full pages
  # num_part2 是落在“完整新 page”里的 token/slot 数；一定是 page_size 的倍数。
  num_part2 = seq_len // page_size * page_size - (pre_len + page_size - 1) // page_size * page_size

  offset_many_page = tl.arange(0, max_num_extend_tokens)

  # offset_many_page // page_size 表示当前 lane 属于第几个新 page：
  #   [0, 1, 2, 3, 4, 5, 6, 7] // 4 = [0, 0, 0, 0, 1, 1, 1, 1]
  # 所以 page_start 是一个向量：每个新增 token lane 对应的 page id。
  page_start = tl.load(
    free_page_ptr + new_page_start_loc + offset_many_page // page_size,
    mask=offset_many_page < num_part2,
  )

  # 把 page id 展开成物理 token slot id：
  #   slot_id = page_id * page_size + offset_in_page
  # 写入 out_indices 的仍然是地址表，不是真正的 K/V 内容。
  tl.store(
    out_indices + output_start_loc + num_part1 + offset_many_page,
    page_start * page_size + offset_many_page % page_size,
    mask=offset_many_page < num_part2,
  )
  if pre_len + num_part1 + num_part2 == seq_len:
    return

  # Part 3: fill the new partial page
  # 如果最后还剩一个不完整 page，从当前请求分到的最后一个新 page 开始展开。
  num_part3 = seq_len - seq_len // page_size * page_size
  start_loc = tl.load(free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1)
  tl.store(
    out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
    start_loc * page_size + offset_one_page,
    mask=offset_one_page < num_part3,
  )


@triton.jit
def alloc_decode_kernel(
  seq_lens_ptr,
  last_loc_ptr,
  free_page_ptr,
  out_indices,
  bs_upper: tl.constexpr,
  page_size: tl.constexpr,
):
  # decode kernel 的输入更简单：每个请求只追加 1 个 token。
  # 如果这个 token 仍在旧 page 内，就写到 last_loc + 1；如果它正好跨到新
  # page 的第一个 slot，就必须从 free_page_ptr 里拿一个新 page。
  pid = tl.program_id(0)

  # 和 extend kernel 一样，这里要知道当前请求之前有多少请求跨 page，
  # 这样当前请求才能从 free_page_ptr 中拿到不重叠的新 page。
  load_offset = tl.arange(0, bs_upper)
  seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
  pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

  seq_len = tl.load(seq_lens_ptr + pid)
  pre_len = seq_len - 1

  num_pages_after = (seq_lens + page_size - 1) // page_size
  num_pages_before = (pre_lens + page_size - 1) // page_size
  num_new_pages = num_pages_after - num_pages_before

  num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
    pre_len + page_size - 1
  ) // page_size
  sum_num_new_pages = tl.sum(num_new_pages)
  new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

  if num_page_start_loc_self == 0:
    last_loc = tl.load(last_loc_ptr + pid)
    # 不跨 page：当前 decode token 紧跟在上一 token 的物理 slot 后面。
    tl.store(out_indices + pid, last_loc + 1)
  else:
    # 跨 page：从 free_pages 中取当前请求对应的新 page，并返回该 page 的第 0 个 slot。
    page = tl.load(free_page_ptr + new_page_start_loc)
    tl.store(out_indices + pid, page * page_size)


def get_num_new_pages(
  seq_lens: torch.Tensor,
  page_size: int,
  prefix_lens: Optional[torch.Tensor] = None,
  decode: bool = False,
) -> torch.Tensor:
  """
  估算一批请求这一步总共需要多少“新 page”。

  这里用 CPU tensor，是因为 Python 调度逻辑需要一个普通 int：
  - 在 launch Triton kernel 前检查 free_pages 是否足够，避免 kernel 读越界。
  - 分配完成后用同一个数量从 free_pages 头部切掉已经消耗的 page。
  如果在 GPU 上算完再 .item()，会触发一次 GPU->CPU 同步；这里本来就已有
  seq_lens_cpu / prefix_lens_cpu，所以直接在 CPU 上做这笔容量账。

  逻辑：
  - extend 场景：对每个 request 分别比较 prefix 前后的 page 数变化，再求和。
  - decode 场景：每个 request 只追加 1 个 token，只需判断哪些 request 跨 page。
  """
  cpu_device = torch.device("cpu")
  assert seq_lens.device == cpu_device

  if prefix_lens is None or decode:
    # NOTE: Special case for handling decode, which prefix lens is `seq_lens - 1`.
    # 这个 helper/kernel 约定：decode 分支里的 seq_lens 表示“追加本轮 token 后”
    # 的序列长度；因此 prefix lens 可视为 seq_lens - 1。
    # 如果 seq_lens % page_size == 1，说明本轮新 token 是某个新 page 的第 1
    # 个 token，因此需要分配新 page。
    #
    # 例：page_size = 4, seq_lens = [4, 5, 1]
    #   [4, 5, 1] % 4 == 1 -> [False, True, True]
    #   .int().sum().item() -> 2
    assert decode
    return (seq_lens % page_size == 1).int().sum().item()

  assert prefix_lens.device == cpu_device
  # 注意不能用 seq_lens.sum() 先合并 batch 再除 page_size。
  # 每个 request 都有独立的 page 边界，所以必须先 per-request ceil，再求和。
  num_pages_after = (seq_lens + page_size - 1) // page_size
  num_pages_before = (prefix_lens + page_size - 1) // page_size
  num_new_pages = num_pages_after - num_pages_before
  sum_num_new_pages = torch.sum(num_new_pages).to(torch.int64)
  return sum_num_new_pages.item()


class PagedTokenAllocator(ITokenAllocator):
  """
  page 级 KV 位置分配器。

  它维护的是空闲 page id 列表 free_pages，不是空闲 token slot 列表。
  当 scheduler 需要为一批新 token 写 KV 时，这里会把 page id 展开成
  具体物理 slot id，返回给 KVCacheManager 作为 out_cache_loc。

  调用链：
    KVCacheManager.prepare_for_extend/decode
      -> token_allocator.alloc_pages_extend/decode
      -> alloc_extend/alloc_decode
      -> Triton kernel 生成 out_indices/out_cache_loc
      -> write_cache_indices 写 RequestPool
      -> attention backend 用 out_cache_loc 写真实 K/V
  """

  def __init__(self, size: int, page_size: int, device: str):
    self.size = size
    self.page_size = page_size
    self.device = device
    self.num_pages = size // page_size
    self.seen_max_num_extend_tokens_next_power_of_2 = 1
    # keep allocator in a usable default state
    self.debug_mode = False
    self.need_sort = False
    self.clear()

  def alloc(self, need_size: int):
    # 这是最基础的 page 对齐分配接口：
    # 先拿到若干页，再把每页展开成连续 token 索引。
    assert need_size % self.page_size == 0, "The allocation size should be page-aligned"

    num_pages = need_size // self.page_size
    if num_pages > len(self.free_pages):
      return None

    out_pages = self.free_pages[:num_pages]
    self.free_pages = self.free_pages[num_pages:]

    # out_pages 是一维 page id 数组。例：out_pages = [2, 5], page_size = 4
    #   out_pages[:, None] -> [[2], [5]]
    #   out_pages[:, None] * 4 + [0, 1, 2, 3]
    #     -> [[8, 9, 10, 11], [20, 21, 22, 23]]
    # reshape 后得到要分配出去的物理 token slot id。
    out_indices = (
      out_pages[:, None] * self.page_size + torch.arange(self.page_size, device=self.device)
    ).reshape(-1)

    return out_indices

  def alloc_extend(
    self,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
  ):
    # extend = prompt/prefill 阶段。
    # 一个请求可能一次写入很多 token，而且前后 token 可能跨 page。
    # 这个函数的输出是一维物理 slot id 数组，布局为：
    #   [req0_extend_slots..., req1_extend_slots..., ...]
    # KVCacheManager 会把它保存为 batch.out_cache_loc。
    if self.debug_mode:
      # last_loc 是 prefix 最后一个 token 的物理 slot；last_loc + 1 应该和
      # prefix_lens 的页内偏移对齐，表示新增 token 可以从正确位置继续写。
      assert torch.all((last_loc + 1) % self.page_size == prefix_lens % self.page_size)

    # 先检查是否有足够的 free pages，避免 Triton kernel 读越界。
    # 这里数的是“新 page 数”，不是“新 token 数”；旧 partial page 的剩余 slot
    # 不需要从 free_pages 中再拿 page。
    num_new_pages = get_num_new_pages(
      seq_lens=seq_lens_cpu,
      page_size=self.page_size,
      prefix_lens=prefix_lens_cpu,
    )
    if num_new_pages > len(self.free_pages):
      return None

    self.seen_max_num_extend_tokens_next_power_of_2 = max(
      self.seen_max_num_extend_tokens_next_power_of_2,
      next_power_of_2(extend_num_tokens),
    )

    bs = len(prefix_lens)
    out_indices = torch.empty((extend_num_tokens,), dtype=torch.int64, device=self.device)

    # Kernel 会把三段位置写到 out_indices：
    # 1. 旧 page 的尾巴（如果 prefix 没写满一个 page）
    # 2. 中间完整的新 page
    # 3. 最后的新 partial page
    # Triton kernel 调用语法：kernel[grid](args...)。
    # grid=(bs,) 表示启动 bs 个 program，每个 program 处理一个请求。
    alloc_extend_kernel[(bs,)](
      prefix_lens,
      seq_lens,
      last_loc,
      self.free_pages,
      out_indices,
      next_power_of_2(bs),
      self.page_size,
      self.seen_max_num_extend_tokens_next_power_of_2,
    )
    # torch.unique 返回去重后的张量；长度不变说明这批 token 没有拿到重复 slot。
    assert len(torch.unique(out_indices)) == len(out_indices)

    # kernel 只读取 free_pages 并生成地址表；真正从 allocator 中“消费”page 是这里。
    self.free_pages = self.free_pages[num_new_pages:]
    return out_indices

  def alloc_decode(
    self,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
  ):
    # decode = 每个请求只追加 1 个 token，输出是一维物理 slot id 数组。
    # kernel 会为每个请求判断：新 token 是接在旧 page 里，还是要从 free_pages
    # 领取一个新 page。
    # 注意：alloc_decode_kernel 内部把 pre_len 写成 seq_len - 1，所以这里的
    # seq_lens 按 allocator 约定应表示“追加本轮 token 后”的长度。
    if self.debug_mode:
      # 这个断言检查的是页内偏移是否一致：
      #   当前 decode token 的候选 slot 是 last_loc + 1。
      #   写完后，下一次追加 token 的页内偏移应与当前序列长度 seq_lens 对齐。
      # 如果这里失败，通常说明 RequestPool 里的 last_loc 和 seq_lens 不匹配。
      assert torch.all((last_loc + 2) % self.page_size == seq_lens % self.page_size)

    bs = len(seq_lens)
    if self.need_sort and bs > len(self.free_pages):
      self.merge_and_sort_free()

    # 先做 page 数量检查，避免 kernel 读取空 free_pages。
    num_new_pages = get_num_new_pages(
      seq_lens=seq_lens_cpu,
      page_size=self.page_size,
      decode=True,
    )
    if num_new_pages > len(self.free_pages):
      return None

    out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)

    # Kernel 输出的是每个请求新 token 的物理位置。
    alloc_decode_kernel[(bs,)](
      seq_lens,
      last_loc,
      self.free_pages,
      out_indices,
      next_power_of_2(bs),
      self.page_size,
    )

    if self.debug_mode:
      assert len(torch.unique(out_indices)) == len(out_indices)

    # 和 extend 一样，kernel 只生成地址表；这里才真正消费新 page。
    self.free_pages = self.free_pages[num_new_pages:]
    return out_indices

  def free(self, free_index: torch.Tensor):
    # free_index 是 token 级物理 slot id。PagedTokenAllocator 真正回收的是 page，
    # 所以先用 slot_id // page_size 折叠成 page id。
    if free_index.numel() == 0:
      return

    if self.is_not_in_free_group:
      free_page_indices = torch.unique(free_index // self.page_size)
      # Page 0 is a reserved/padded slot and must never enter free pages.
      valid_mask = (free_page_indices >= 1) & (free_page_indices <= self.num_pages)
      if self.debug_mode and not bool(torch.all(valid_mask)):
        invalid = free_page_indices[~valid_mask]
        logger.warning("Dropping invalid free page ids: %s", invalid.detach().cpu().tolist())
      free_page_indices = free_page_indices[valid_mask]
      if free_page_indices.numel() == 0:
        return
      # 先把释放的 page 拼回 free_pages，再统一去重。
      # 注意 torch.unique 会排序，因此这里维护的是“可用 page 集合”，不是严格 LIFO。
      self.free_pages = torch.cat((free_page_indices, self.free_pages))
      # 去重：防止 radix tree split 节点时同一个 page 被多个节点持有。
      # 这里的“重复”往往来自前缀树节点拆分，不是 bug。
      self.free_pages = torch.unique(self.free_pages)
    else:
      self.free_group.append(free_index)

    if self.debug_mode:
      assert len(torch.unique(self.free_pages)) == len(self.free_pages)
      assert torch.all(self.free_pages >= 1) and torch.all(self.free_pages <= self.num_pages)
      assert len(self.free_pages) <= self.num_pages

  def begin_free_group(self):
    """进入批量释放模式：先攒起来，最后统一去重合并。"""
    self.is_not_in_free_group = False

  def end_free_group(self):
    """退出批量释放模式，并把暂存的释放内容真正写回 free_pages。"""
    self.flush_free_group()
    self.is_not_in_free_group = True

  def flush_free_group(self):
    """把 free_group 里的 token 位置统一折叠成 page，并合并进 free_pages。"""
    if not self.free_group:
      return
    # free_group 是多次 free() 暂存的 slot id tensor 列表。
    # torch.cat 后得到一个大的一维 slot id 数组，再统一折叠成 page id。
    free_index = torch.cat(self.free_group)
    self.free_group = []
    if free_index.numel() == 0:
      return

    free_page_indices = torch.unique(free_index // self.page_size)
    # Page 0 is a reserved/padded slot and must never enter free pages.
    valid_mask = (free_page_indices >= 1) & (free_page_indices <= self.num_pages)
    if self.debug_mode and not bool(torch.all(valid_mask)):
      invalid = free_page_indices[~valid_mask]
      logger.warning("Dropping invalid free page ids: %s", invalid.detach().cpu().tolist())
    free_page_indices = free_page_indices[valid_mask]
    if free_page_indices.numel() == 0:
      return

    # 一次性合并并去重，摊薄多次 free 的 concat/unique 开销。
    self.free_pages = torch.unique(torch.cat((free_page_indices, self.free_pages)))

    if self.debug_mode:
      assert len(torch.unique(self.free_pages)) == len(self.free_pages)
      assert torch.all(self.free_pages >= 1) and torch.all(self.free_pages <= self.num_pages)
      assert len(self.free_pages) <= self.num_pages

  def clear(self):
    # page 0 保留为 padding / dummy 写回槽位。
    self.free_pages = torch.arange(1, self.num_pages + 1, dtype=torch.int64, device=self.device)
    self.is_not_in_free_group = True
    self.free_group = []

  # ======= ITokenAllocator compatibility helpers =======
  def alloc_pages_extend(
    self,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
  ) -> Optional[torch.Tensor]:
    return self.alloc_extend(
      prefix_lens,
      prefix_lens_cpu,
      seq_lens,
      seq_lens_cpu,
      last_loc,
      extend_num_tokens,
    )

  def alloc_pages_decode(
    self,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
  ) -> Optional[torch.Tensor]:
    return self.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

  def available_size(self) -> int:
    # 只统计当前已经 materialize 到 free_pages 里的 page。
    # free_group 是批量 free 的暂存区，只有 flush 后才会真正生效。
    free_pages = len(self.free_pages)
    return free_pages * self.page_size


class RequestPool(IRequestPool):
  """
  请求池。

  作用：记录“某个 request 的第 i 个 token”对应 KV cache 的物理位置。

  你可以把它理解成一个二维表：
  - 行：req_idx
  - 列：token_pos
  - 值：kv_loc

  attention backend 做 paged attention / flash attention 时，
  就是靠这张表找到每个 token 的真实 KV 地址。

  Args:
      max_requests: max running batch size
      max_context_len: from model config, max context length
  """

  def __init__(self, max_requests: int, max_context_len: int, device: str):
    self.max_requests = max_requests
    self.max_context_len = max_context_len
    self.device = device

    # req_to_token[req_idx, token_pos] = kv_loc
    # int32 足够表示物理位置，且能省显存。
    self.req_to_token = torch.zeros(
      (max_requests, max_context_len), dtype=torch.int32, device=device
    )
    self.free_slots = list(range(max_requests))

  def req_to_token_pool(self):
    return self.req_to_token

  def alloc(self, num_reqs: int) -> Optional[List[int]]:
    # 分配 request 行号。
    if num_reqs > len(self.free_slots):
      return None
    allocated = self.free_slots[:num_reqs]
    self.free_slots = self.free_slots[num_reqs:]
    return allocated

  def free(self, indices: List[int]):
    # 释放 request 行号，供后续新请求复用。
    # 这里不会清空 req_to_token 里的旧值；后续读取会受新的 seq_len/token_range 限制，
    # 新请求复用这一行时也会覆盖需要的范围。
    self.free_slots.extend(indices)

  # TODO(lzy): use cuda kernel to accelerate
  def write(self, req_idx: int, token_range: slice, kv_indices: torch.Tensor):
    # 把一段 token_pos -> kv_loc 的映射写进表里。
    self.req_to_token[req_idx, token_range] = kv_indices.to(torch.int32)

  def read(self, req_idx: int, token_range: slice) -> torch.Tensor:
    # 读取映射，统一转成 int64 方便后续索引运算。
    return self.req_to_token[req_idx, token_range].to(torch.int64)
