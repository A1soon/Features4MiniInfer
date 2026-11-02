from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, List, ClassVar

import torch
from itertools import count
from copy import copy
from miniinfer.layers.attention_backend.base_backend import AttentionBackend
from miniinfer.utils import get_logger
from miniinfer.utils.sampling_params import SamplingParams

logger = get_logger(__name__)


class ForwardMode(IntEnum):
  """
  描述一次模型 forward 的类型。

  这是贯穿 Scheduler、ForwardBatch、ModelRunner、AttentionBackend 的核心枚举。
  你可以先记住：
  - EXTEND: 处理一段新 token，常见于 prompt prefill。
  - DECODE: 每个请求只处理 1 个新 token，常见于逐 token 生成阶段。
  - MIXED: 同一批里既有 prefill，也有 decode。
  - IDLE: 当前没有可跑请求。
  """

  # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
  # It is also called "prefill" in common terminology.
  EXTEND = auto()
  # Decode one token.
  DECODE = auto()
  # Contains both EXTEND and DECODE when doing chunked prefill.
  MIXED = auto()
  # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
  IDLE = auto()

  def is_extend(self) -> bool:
    return self == ForwardMode.EXTEND or self == ForwardMode.MIXED

  def is_decode(self) -> bool:
    return self == ForwardMode.DECODE

  def is_mixed(self) -> bool:
    return self == ForwardMode.MIXED


class BatchType(IntEnum):
  """Batch 类型"""

  PREFILL_ONLY = auto()
  DECODE_ONLY = auto()
  MIXED = auto()


class Req:
  """
  单个用户请求在推理引擎内部的状态对象。

  初学者可以把 Req 理解成“请求档案袋”：
  - 用户原始输入的 token ids 放在 origin_input_ids。
  - 模型已经生成的 token ids 放在 output_ids。
  - 是否完成、为什么完成、是否被 retract，也都记录在这里。
  - KV cache / prefix cache 相关的位置和命中信息，也挂在同一个 Req 上。

  Req 会在多个模块之间流动：
    LLMEngine.add_requests()
      -> Scheduler.waiting_queue
      -> ScheduledBatch
      -> ForwardBatch
      -> ModelRunner.forward()/sample()
      -> LLMEngine._process_step_result()

  注意：
  这里的 req_id 是进程内自增 id，用来区分请求，不是用户输入文本。
  """

  # counter 是一个无限递增计数器。next(Req.counter) 每调用一次就返回下一个整数：
  # 0, 1, 2, 3, ...
  # 这样每个 Req 都能拿到唯一的 req_id。
  counter = count()

  def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
    """
    创建一个请求对象。

    Args:
        token_ids:
            prompt 已经经过 tokenizer 编码后的整数列表。模型不直接处理字符串，
            所以进入 Scheduler 前必须先变成 token ids。
        sampling_params:
            控制生成行为的参数，比如 temperature、max_tokens、top_k/top_p。
    """
    # 给这个请求分配一个唯一 id，后续结果返回、调试日志、detokenizer 状态都会用它。
    self.req_id = next(Req.counter)

    # origin_input_ids 是原始 prompt token。copy(token_ids) 的作用是复制一份，
    # 避免外部传进来的 list 后续被修改，导致 Req 内部状态被意外改变。
    self.origin_input_ids = copy(token_ids)

    # output_ids 保存模型已经生成的新 token，不包含 prompt token。
    # decode 阶段每生成一个 token，就会 append 到这里。
    self.output_ids = []

    # fill_ids 表示当前需要被 prefill/extend 关注的完整序列：
    #   origin_input_ids + output_ids
    # 在普通 prefill 中，它基本等于 prompt。
    # 在 retract/chunked prefill/prefix cache 场景中，它帮助 Scheduler 判断哪些 token
    # 已经有 KV cache，哪些 token 还需要继续计算。
    self.fill_ids = []

    # 保存原始 SamplingParams 对象，Sampler 会读取其中的 temperature/top_p/top_k。
    self.sampling_params = sampling_params

    # ============ sampling related ============
    # 为了后续访问方便，把常用采样字段也平铺到 Req 上。
    # max_tokens 是“最多生成多少输出 token”，不是 prompt 最大长度。
    self.temperature = sampling_params.temperature
    self.max_tokens = sampling_params.max_tokens
    self.ignore_eos = sampling_params.ignore_eos
    self.top_k = sampling_params.top_k
    self.top_p = sampling_params.top_p

    # ============ finish related ============
    # is_retracted 表示这个请求本轮因为 KV cache 空间不足等原因被临时移出 running batch，
    # 后续会重新进入 waiting_queue，重新 prefill。
    self.is_retracted = False

    # ever_retracted 用于记录历史上是否发生过 retract，方便调试/统计。
    self.ever_retracted = False

    # finished 表示请求是否已经完成。完成条件通常是：
    # - 生成到 EOS token。
    # - 或者 output_ids 长度达到 max_tokens。
    self.finished = False

    # finished_reason 常见值是 "eos" 或 "max_tokens"。
    self.finished_reason = ""

    # ============ kv cache related ============
    # req_pool_idx 是这个请求在 RequestPool 里的行号。
    # RequestPool 可以先粗略理解成二维表：
    #   req_pool_idx + token_position -> kv_cache_physical_location
    # 即：某个请求的第几个 token 对应物理 KV cache 中哪个位置。
    #
    # -1 表示当前还没有分配 RequestPool 槽位。
    self.req_pool_idx: int = (
      -1
    )  # The index in the request pool, for radix cache and kv cache management

    # kv_committed_len 是“已经完成 forward、KV 内容有效”的逻辑前缀长度。
    # 它和 len(origin_input_ids) + len(output_ids) 不总是相等：当前轮采样出的
    # 最新 output token 只有在下一轮作为 input 经过 forward 后，才拥有自己的 KV。
    # retract/resume 只能把这个边界以内的 KV 交给 RadixCache。
    self.kv_committed_len: int = 0

    # extend_input_len 表示本轮还需要真正跑模型的 token 数。
    # 如果 prefix cache 命中了前 N 个 token，那么这部分不用重复计算，
    # extend_input_len 就会小于完整 prompt 长度。
    self.extend_input_len = 0

    # prefix_indices 保存 prefix cache 命中的 KV cache 物理位置。
    # 如果一个请求的前缀已经算过，这里就会记录可复用的 KV indices。
    # 关闭 prefix cache 的 chunked prefill 也会在这里保留请求私有的已计算 KV 映射。
    self.prefix_indices: torch.Tensor = None

    # ============ radix cache related ============
    # RadixCache 用一棵压缩前缀树保存可复用的 prompt prefix。
    # last_node 表示这个请求当前命中的 radix tree 节点，后续释放/解锁会用到。
    self.last_node = None

    # cache_protected_len 表示已经被 prefix cache 命中并保护起来的 token 数。
    # “保护”意味着这些 KV 暂时不能被驱逐，因为当前请求正在使用它们。
    self.cache_protected_len = 0  # 已经保护的 KV cache 长度，防止被驱逐

    # ============ chunked prefill related ============
    # 如果 prompt 很长，一次 prefill 可能放不进预算。chunked prefill 会把 prompt
    # 切成多段处理。is_chunked=True 表示这个请求还处在分块 prefill 过程中。
    self.is_chunked = False  # 是否是 chunked 请求

    # 已经完成 prefill 的 token 数。下一轮 chunk 会从这个位置继续。
    self.chunked_prefill_len = 0  # 当前 chunk 已经处理的长度

    # prompt 的总长度，只包含输入 token，不包含后续生成的 output_ids。
    self.total_input_len = len(token_ids)  # 总输入长度

    # ============ cached metadata for performance ============
    # 缓存当前序列长度，避免重复计算 len(origin_input_ids) + len(output_ids)
    self._cached_seq_len = len(token_ids)
    # 缓存最后的 output token，避免重复访问 output_ids[-1]
    self._cached_last_token = None

  @property
  def remaining_prefill_len(self) -> int:
    """
    剩余需要 prefill 的长度。

    total_input_len:
        prompt 总长度。
    chunked_prefill_len:
        已经通过 chunked prefill 处理过的长度。
    cache_protected_len:
        prefix cache 已经命中的长度。

    三者相减后，就是还需要实际跑模型补 KV cache 的 token 数。
    """
    return self.total_input_len - self.chunked_prefill_len - self.cache_protected_len

  def is_prefill_complete(self) -> bool:
    """检查这个请求的 prompt 是否已经全部完成 prefill。"""
    return self.chunked_prefill_len + self.cache_protected_len >= self.total_input_len

  @property
  def current_seq_len(self) -> int:
    """
    获取当前序列长度。

    当前序列长度 = prompt token 数 + 已生成 output token 数。
    这里直接返回缓存值，避免在 decode 热路径里反复做 len(...) 加法。
    """
    return self._cached_seq_len

  def append_output_token(self, token_id: int):
    """
    追加一个新生成的 token，并同步更新缓存字段。

    这个函数通常在 LLMEngine._process_step_result() 中被调用。
    每次 decode/prefill 采样出 next token 后，请求状态就在这里向前推进一步。
    """
    self.output_ids.append(token_id)
    self._cached_seq_len += 1
    self._cached_last_token = token_id

  def mark_kv_committed(self, length: int) -> None:
    """Advance the committed-KV boundary without changing logical output state."""
    logical_len = self.current_seq_len
    self.kv_committed_len = max(0, min(int(length), logical_len))

  def reset_cache_state_after_retract(self) -> None:
    """Drop transient KV metadata while preserving the request's logical history."""
    self.fill_ids = []
    self.prefix_indices = None
    self.cache_protected_len = 0
    self.last_node = None
    self.extend_input_len = 0
    self.req_pool_idx = -1
    self.chunked_prefill_len = 0
    self.kv_committed_len = 0
    self._cached_seq_len = len(self.origin_input_ids) + len(self.output_ids)
    self._cached_last_token = self.output_ids[-1] if self.output_ids else None

  @property
  def committed_token_ids(self) -> list[int]:
    """Return exactly the logical token prefix whose KV has been committed."""
    origin_len = len(self.origin_input_ids)
    committed_len = max(0, min(self.kv_committed_len, self.current_seq_len))
    if committed_len == 0:
      return []
    prompt_len = min(committed_len, origin_len)
    output_len = max(committed_len - origin_len, 0)
    return self.origin_input_ids[:prompt_len] + self.output_ids[:output_len]

  def get_last_token(self) -> Optional[int]:
    """
    获取最后一个已生成 token。

    decode 阶段下一轮的输入通常就是“上一轮刚生成的 token”。
    如果还没有生成任何 token，返回 None。
    """
    if self._cached_last_token is not None:
      return self._cached_last_token
    if len(self.output_ids) > 0:
      self._cached_last_token = self.output_ids[-1]
      return self._cached_last_token
    return None


class ChunkedReq:
  """
  Chunked Prefill 请求的状态追踪

  当一个请求太大无法在一个 batch 中完成 prefill 时，
  会被拆分成多个 chunk 处理。这个类用于追踪 chunked 状态。

  注意：ChunkedReq 不会进入 decode 阶段，直到所有 chunk 都处理完成。
  """

  def __init__(
    self,
    req: Req,
    cached_len: int,
    chunk_size: int,
  ):
    """
    初始化 ChunkedReq

    Args:
        req: 原始请求
        cached_len: 已缓存的长度（prefix cache 命中）
        chunk_size: 当前 chunk 的大小
    """
    self.req = req
    self.cached_len = cached_len  # 包括 prefix cache + 已处理的 chunk
    self.chunk_size = chunk_size  # 当前 chunk 要处理的 token 数

    # 不要在这里修改 req.is_chunked！
    # add_chunked_req / _do_chunked_prefill 已经正确设置了 is_chunked 状态。
    # 最后一个 chunk 时 is_chunked = False，此时不应该被改回 True。

  @property
  def input_len(self) -> int:
    """原始输入总长度"""
    return self.req.total_input_len

  @property
  def remaining_len(self) -> int:
    """处理完当前 chunk 后，还剩余的长度"""
    return self.input_len - self.cached_len - self.chunk_size

  @property
  def is_last_chunk(self) -> bool:
    """是否是最后一个 chunk"""
    return self.remaining_len <= 0

  def get_chunk_input_ids(self) -> list[int]:
    """获取当前 chunk 的 input_ids"""
    start = self.cached_len
    end = start + self.chunk_size
    return self.req.origin_input_ids[start:end]

  def update_after_chunk(self):
    """
    处理完当前 chunk 后更新状态

    如果还有剩余，更新 cached_len 以便下一个 chunk
    """
    self.req.chunked_prefill_len = self.cached_len + self.chunk_size
    if self.is_last_chunk:
      self.req.is_chunked = False  # 完成所有 chunk


@dataclass
class ScheduledBatch:
  """
  Scheduler 输出的一批请求。

  你可以把 ScheduledBatch 理解成“调度器给执行器的一张工单”：
  - 这轮要跑哪些 Req？
  - 是 prefill/extend，还是 decode，还是 mixed？
  - 本轮输入 token ids 是什么？
  - 每个请求当前序列长度是多少？
  - 新生成/新 prefill 的 KV cache 应该写到哪些物理位置？

  它是 Scheduler/KVCacheManager 的视角，还不是模型最终使用的输入。
  ModelRunner 真正使用前，会通过 ForwardBatch.init_new(batch, ...)
  把它转换成 ForwardBatch。
  """

  # 本轮参与调度的请求列表。顺序很重要：后续 input_ids、seq_lens、
  # out_cache_loc 等张量都按这个顺序对应每个 Req。
  reqs: List[Req]

  # forward_mode 描述本轮计算类型：
  # - EXTEND: 处理新 prompt / prefill。
  # - DECODE: 每个 running 请求生成下一个 token。
  # - MIXED: 同一轮里同时处理 prefill 和 decode。
  # - IDLE: 没有请求。
  forward_mode: ForwardMode = ForwardMode.IDLE

  # 当前 batch 相关 tensor 所在设备，通常是 cuda。
  device: torch.device = None

  # ============ model forward related ============
  # 本轮要喂给模型的 token ids。
  # EXTEND/MIXED 时通常是多个请求的新 token 拼接在一起；
  # DECODE 时通常是每个请求 1 个 token。
  input_ids: torch.Tensor = None

  # output_ids 在 overlap 路径里可能保存当前 batch 采样得到的 token，
  # 或者保存 FutureMap placeholder。初学阶段可以先把它理解成“本轮输出 token 容器”。
  output_ids: torch.Tensor = None

  # True when this batch's input_ids contains FutureMap placeholders (negative ids)
  # and must be resolved on the forward stream before model forward.
  # placeholder 是 overlap 优化里的概念：下一轮 schedule 时上一轮 token 可能还没
  # 从 GPU 结果处理回来，于是先用占位符表示，等 forward 前再解析。
  uses_placeholder: bool = False

  # =========== kv cache related ============
  # 每个请求在 RequestPool 中的槽位 id。
  req_pool_indices: torch.Tensor = None

  # 本轮新 token 的 KV cache 物理写入位置。
  # attention backend 会用这个位置把当前层的 K/V 写进 KV cache。
  out_cache_loc: torch.Tensor = None

  # ============ some metadata ============
  # 每个请求当前完整序列长度：
  # prompt tokens + generated tokens。
  # attention backend 需要知道每个请求可以 attend 到多长的历史。
  seq_lens: torch.Tensor = (
    None  # from req.fill_ids = origin_input_ids + output_ids, for forward batch k_cache len
  )

  # CPU 侧 seq_lens 副本。很多调度判断在 CPU 上做，保留 CPU 版本可以减少 GPU->CPU 同步。
  seq_lens_cpu: Optional[torch.Tensor] = None

  # ======== extend related ========
  # prefix_lens[i] 表示第 i 个请求命中了多少 prefix cache token。
  prefix_lens: List[int] = None

  # extend_lens[i] 表示第 i 个请求本轮实际需要 forward 的 token 数。
  # 例如 prompt 长度 100，prefix cache 命中 60，则 extend_len 可能是 40。
  extend_lens: List[int] = None

  # ======== chunked prefill related ========
  # MIXED batch 中，哪些请求是 decode 请求。其余请求就是 extend/prefill 请求。
  decoding_reqs: List[Req] = None

  @classmethod
  def init_new(cls, reqs: List[Req], device: torch.device):
    """
    创建一个最基础的 ScheduledBatch。

    这里只填 reqs 和 device，其他字段会由 KVCacheManager.prepare_for_*()
    根据 prefill/decode/mixed 模式继续补齐。
    """
    return cls(
      reqs=reqs,
      device=device,
    )

  def debug_metadata(self):
    logger.debug("ScheduledBatch metadata:")
    logger.debug(f"forward_mode: {self.forward_mode}")
    logger.debug(f"input_ids: {self.input_ids}")
    logger.debug(f"output_ids: {self.output_ids}")
    logger.debug(f"req_pool_indices: {self.req_pool_indices}")
    logger.debug(f"out_cache_loc: {self.out_cache_loc}")
    logger.debug(f"seq_lens: {self.seq_lens}")
    logger.debug(f"prefix_lens: {self.prefix_lens}")
    logger.debug(f"extend_lens: {self.extend_lens}")


@dataclass
class ForwardBatch:
  """
  ModelRunner/模型前向真正使用的 batch。

  ScheduledBatch 是调度器产物，ForwardBatch 是模型执行产物。
  转换发生在 LLMEngine.step() 或 step_overlap() 里：

    forward_batch = ForwardBatch.init_new(batch, attn_backend)

  ForwardBatch 会把模型真正需要的元数据准备好，例如：
  - input_ids: 当前要算的 token。
  - positions: 每个 token 的位置，用于 RoPE。
  - seq_lens: 每个请求当前总长度，用于 attention 读历史 KV。
  - extend_prefix_lens/extend_seq_lens: prefill/extend attention 需要的长度信息。
  - sampling_*: Sampler 需要的 temperature/top_p/top_k。

  初学时先记住一句话：
  ScheduledBatch 负责“这轮跑谁”，ForwardBatch 负责“模型怎么跑这一轮”。
  """

  # 本轮模型前向的模式，和 ScheduledBatch.forward_mode 对应。
  forward_mode: ForwardMode

  # ============ model forward related ============
  # 请求数量。注意 EXTEND 时 input_ids 的 token 总数可能大于 batch_size，
  # 因为一个请求的 prompt 可能有很多 token。
  batch_size: int

  # 本轮真正输入模型的 token ids。
  input_ids: torch.Tensor

  # 每个 input token 的位置编号。RoPE 会用 positions 给 q/k 加位置信息。
  # - EXTEND: positions 长度等于本轮总 token 数。
  # - DECODE: positions 长度等于 batch_size，每个请求一个位置。
  positions: torch.Tensor = None

  # ============ kv cache related ============
  # 每个请求在 RequestPool 中的槽位 id。
  req_pool_indices: torch.Tensor = None

  # 本轮 K/V 要写入的 KV cache 物理位置。
  out_cache_loc: torch.Tensor = None

  # 当前使用的 attention backend，例如 FlashAttention2Backend 或 FlashInferBackend。
  attn_backend: AttentionBackend = None

  # 本轮参与 forward 的 Req 列表。Sampler 会从这里读取每个请求的采样参数。
  all_seqs: List[Req] = None

  # ============ sampling related ============
  # 下面这些 sampling_* tensor 是把每个请求的采样参数批量放到 GPU 上。
  # 这样 ModelRunner.sample() 不需要每一步重新从 Python list 创建 tensor。
  sampling_temperatures: Optional[torch.Tensor] = None
  sampling_top_ps: Optional[torch.Tensor] = None
  sampling_top_ks: Optional[torch.Tensor] = None

  # CPU-side metadata to avoid GPU->CPU sync in sampling.
  # sampling_max_top_k: Optional[int] = None

  # Optional device-side indices of the last token per sequence in an EXTEND/MIXED
  # logits tensor of shape [total_tokens, vocab_size]. Built on CPU and transferred
  # via pinned buffer to avoid per-step GPU allocations/copies.
  #
  # EXTEND/prefill 时，一个请求可能输入多个 token，模型会对每个位置都输出 logits。
  # 但采样下一个 token 只需要每个请求“最后一个位置”的 logits。
  # last_token_indices 就是从 [total_tokens, vocab_size] 中取这些最后位置的索引。
  last_token_indices: Optional[torch.Tensor] = None

  # ============ Pinned Memory Buffer Pool (类级别，所有实例共享) ============
  # 用于高效的 sampling params 异步传输（SGLang 风格优化）
  #
  # pinned memory 是一种固定在物理内存里的 CPU 内存，CPU->GPU 拷贝更快，
  # 且可以配合 non_blocking=True 异步传输。这里用 ClassVar 表示这些 buffer
  # 属于 ForwardBatch 类本身，所有 ForwardBatch 实例共享，避免每步重复分配。
  _pinned_temperatures: ClassVar[Optional[torch.Tensor]] = None
  _pinned_top_ps: ClassVar[Optional[torch.Tensor]] = None
  _pinned_top_ks: ClassVar[Optional[torch.Tensor]] = None
  _pinned_extend_lens: ClassVar[Optional[torch.Tensor]] = None
  _pinned_extend_prefix_lens: ClassVar[Optional[torch.Tensor]] = None
  _pinned_last_token_indices: ClassVar[Optional[torch.Tensor]] = None
  _max_buffer_size: ClassVar[int] = 256  # 默认最大 batch size

  # ============ some metadata for k ============
  # 每个请求当前总序列长度，attention backend 会用它决定能看到多少历史 KV。
  seq_lens: torch.Tensor = None

  # Pre-converted int32 version to avoid GPU dtype conversion overhead
  # 一些底层 kernel 更希望 int32。提前转好可以避免 backend 里反复转换。
  seq_lens_int32: Optional[torch.Tensor] = None

  # =========== TODO: Maybe remove ===========
  # seq_lens_sum 是历史遗留/兼容字段，表示所有 seq_lens 的和。
  seq_lens_sum: int = 0

  # CPU 侧 seq_lens。调度和一些 shape 计算在 CPU 上完成，避免 GPU 同步。
  seq_lens_cpu: Optional[torch.Tensor] = None

  # Max sequence length (computed on CPU to avoid GPU sync)
  # 当前 batch 中最长序列长度。attention backend 可能用它做 buffer/kernel 参数。
  max_seq_len: Optional[int] = None

  # ============ some metadata for q
  # extend_num_tokens: Optional[int] = None

  # EXTEND/MIXED 模式下，每个请求本轮实际输入多少 token。
  # 例如两个 prompt 本轮分别 extend 3 和 5 个 token，则 extend_seq_lens=[3,5]。
  extend_seq_lens: Optional[torch.Tensor] = None

  # EXTEND/MIXED 模式下，每个请求已经复用/保护了多少 prefix token。
  extend_prefix_lens: Optional[torch.Tensor] = None
  # extend_start_loc: Optional[torch.Tensor] = None

  # CPU 侧副本。用于 Python 逻辑或避免 GPU->CPU sync。
  extend_prefix_lens_cpu: Optional[List[int]] = None
  extend_seq_lens_cpu: Optional[List[int]] = None

  @classmethod
  def _ensure_pinned_buffers(cls, max_bs: int):
    """
    确保 pinned memory buffers 已初始化且容量足够。

    这里是类方法，所以第一个参数是 cls，不是 self。
    cls 指 ForwardBatch 这个类本身。

    为什么要提前分配？
    decode 热路径会被调用很多次，如果每一步都临时分配 CPU tensor、
    再拷贝到 GPU，会产生明显开销。预分配 buffer 后，每步只填前 bs 个位置。
    """
    if cls._pinned_temperatures is None or max_bs > cls._max_buffer_size:
      cls._max_buffer_size = max(max_bs, cls._max_buffer_size)
      cls._pinned_temperatures = torch.empty(
        cls._max_buffer_size, dtype=torch.float32, pin_memory=True
      )
      cls._pinned_top_ps = torch.empty(cls._max_buffer_size, dtype=torch.float32, pin_memory=True)
      cls._pinned_top_ks = torch.empty(cls._max_buffer_size, dtype=torch.int64, pin_memory=True)
      cls._pinned_extend_lens = torch.empty(
        cls._max_buffer_size, dtype=torch.int64, pin_memory=True
      )
      cls._pinned_extend_prefix_lens = torch.empty(
        cls._max_buffer_size, dtype=torch.int64, pin_memory=True
      )
      cls._pinned_last_token_indices = torch.empty(
        cls._max_buffer_size, dtype=torch.int64, pin_memory=True
      )

  @classmethod
  def init_new(cls, batch: ScheduledBatch, attn_backend: Optional[AttentionBackend] = None):
    """
    把 ScheduledBatch 转换成 ForwardBatch。

    这是 step() 中从“调度”进入“模型执行”的桥：
      ScheduledBatch:
        调度器/KV cache manager 填好的工单。
      ForwardBatch:
        ModelRunner、模型层、attention backend、sampler 真正读取的输入。

    这个函数主要做四件事：
    1. 复制容易被 overlap 路径修改的长度元数据。
    2. 计算 attention backend 需要的 seq_lens/max_seq_len/int32 版本。
    3. 把每个请求的 sampling params 批量搬到 GPU。
    4. 根据 EXTEND 或 DECODE 模式构造 positions。
    """
    # Snapshot seq_lens because overlap scheduling mutates running batch lengths in-place
    # for the next step while the current step may still be executing on the GPU.
    #
    # 为什么要 clone？
    # step_overlap() 里 CPU 可能已经开始调度下一轮 batch，而上一轮 GPU 还在跑。
    # 如果两个 batch 共享同一个 seq_lens tensor，下一轮修改长度可能影响上一轮 forward。
    # clone 后，当前 ForwardBatch 拥有自己的快照。
    seq_lens = batch.seq_lens.clone() if batch.seq_lens is not None else None
    seq_lens_cpu = batch.seq_lens_cpu.clone() if batch.seq_lens_cpu is not None else None

    # Compute max_seq_len from CPU tensor to avoid GPU sync.
    # 直接对 GPU tensor 调 .max().item() 会让 CPU 等 GPU 完成，破坏性能。
    # 这里用 CPU 副本计算 max_seq_len，避免隐式同步。
    max_seq_len = None
    if seq_lens_cpu is not None:
      max_seq_len = int(seq_lens_cpu.max())

    # Pre-convert seq_lens to int32 to avoid repeated conversions in FA backends.
    # FlashAttention/FlashInfer 等 backend 常用 int32 长度数组。
    seq_lens_int32 = seq_lens.to(torch.int32, non_blocking=True) if seq_lens is not None else None

    # 先创建 ForwardBatch 基础对象，把 ScheduledBatch 已经准备好的字段搬过来。
    # positions、sampling params、extend 相关字段会在下面继续补齐。
    forward_batch = cls(
      forward_mode=batch.forward_mode,
      batch_size=len(batch.reqs),
      input_ids=batch.input_ids,
      req_pool_indices=batch.req_pool_indices,
      out_cache_loc=batch.out_cache_loc,
      seq_lens=seq_lens,
      seq_lens_int32=seq_lens_int32,
      seq_lens_cpu=seq_lens_cpu,
      max_seq_len=max_seq_len,
      attn_backend=attn_backend,
      all_seqs=batch.reqs,
    )

    # Pre-build per-sequence sampling params on the target device.
    # This avoids repeatedly converting Python lists to tensors inside ModelRunner.sample().
    # 优化：使用预分配的 pinned memory buffer 进行零拷贝异步传输（SGLang 风格）
    #
    # 对每个请求来说，temperature/top_p/top_k 可能不同。
    # Sampler 希望一次性处理整个 batch，所以这里把这些 Python 标量整理成 tensor。
    if batch.reqs:
      dev = batch.device if batch.device is not None else batch.input_ids.device
      bs = len(batch.reqs)

      # 确保 pinned buffer 已初始化
      cls._ensure_pinned_buffers(bs)

      # Compute max_top_k on CPU to avoid `.item()` sync in batched top-k.
      # Ensure it's always >= 1 so top-k path can safely run even when empty.
      # forward_batch.sampling_max_top_k = max(
      #   1, max((int(r.sampling_params.top_k) for r in batch.reqs), default=0)
      # )

      # 提取 sampling params 到 pinned buffer（在 CPU 上操作，无 GPU sync）
      temps = [r.sampling_params.temperature for r in batch.reqs]
      top_ps = [r.sampling_params.top_p for r in batch.reqs]
      top_ks = [r.sampling_params.top_k for r in batch.reqs]

      # 先写入 CPU pinned buffer。
      cls._pinned_temperatures[:bs] = torch.as_tensor(temps, dtype=torch.float32)
      cls._pinned_top_ps[:bs] = torch.as_tensor(top_ps, dtype=torch.float32)
      cls._pinned_top_ks[:bs] = torch.as_tensor(top_ks, dtype=torch.int64)

      # CPU-side flag: 是否任一序列请求了 top-k(>0) 或 top-p(<1.0) 过滤。
      # 供 Sampler 选择快速 greedy/temperature 路径 vs flashinfer top-k/top-p 路径，
      # 避免在 GPU 上做 .any() 触发 sync。
      forward_batch.sampling_enable_top_k_top_p = any(
        int(k) > 0 for k in top_ks
      ) or any(float(p) < 1.0 for p in top_ps)
      # CPU-side temperature 路由 flag：
      #   all_greedy = 全部 temperature<=0；all_non_greedy = 全部 temperature>0。
      #   让 Sampler 对纯 greedy 走 argmax、对纯 temp>0 走 flashinfer 融合采样，
      #   仅 mixed 才回落到 float32 Gumbel-max（约 10× 慢）。
      forward_batch.sampling_all_greedy = all(float(t) <= 0 for t in temps)
      forward_batch.sampling_all_non_greedy = all(float(t) > 0 for t in temps)

      # 异步传输到 GPU（真正的零拷贝，无临时 buffer）
      # non_blocking=True 表示如果条件允许，CPU 不必等待拷贝完成。
      forward_batch.sampling_temperatures = cls._pinned_temperatures[:bs].to(dev, non_blocking=True)
      forward_batch.sampling_top_ps = cls._pinned_top_ps[:bs].to(dev, non_blocking=True)
      forward_batch.sampling_top_ks = cls._pinned_top_ks[:bs].to(dev, non_blocking=True)

    # EXTEND 包括普通 prefill 和 MIXED。
    # 这类 batch 的 input_ids 可能是多个请求的多个 token 拼接起来的：
    #   req0: 3 tokens, req1: 5 tokens -> input_ids 长度 8
    # 所以需要额外记录每个请求各自的 extend_len/prefix_len，backend 才能拆回去。
    if batch.forward_mode.is_extend():
      # forward_batch.extend_num_tokens = batch.input_ids.shape[0]
      extend_lens_cpu = batch.extend_lens or []
      total_extend_tokens = int(sum(extend_lens_cpu)) if extend_lens_cpu else 0

      dev = batch.device if batch.device is not None else batch.input_ids.device
      bs = len(batch.reqs)
      cls._ensure_pinned_buffers(bs)

      # H2D extend lens/prefix lens via pinned buffers on the current CUDA stream
      cls._pinned_extend_lens[:bs] = torch.as_tensor(extend_lens_cpu, dtype=torch.int64)
      cls._pinned_extend_prefix_lens[:bs] = torch.as_tensor(
        batch.prefix_lens or [], dtype=torch.int64
      )

      # 把每个请求的 extend_len/prefix_len 拷到 GPU。
      forward_batch.extend_seq_lens = cls._pinned_extend_lens[:bs].to(dev, non_blocking=True)
      forward_batch.extend_prefix_lens = cls._pinned_extend_prefix_lens[:bs].to(
        dev, non_blocking=True
      )

      # Precompute last token indices for sampling from [total_tokens, vocab] logits
      # without per-step GPU tensor allocations.
      #
      # 举例：
      #   extend_lens = [3, 5]
      #   拼接后的 logits 行号为 req0: 0,1,2；req1: 3,4,5,6,7
      #   每个请求用于采样的是最后一行，所以 last_token_indices = [2, 7]
      if total_extend_tokens > 0:
        cumsum = torch.cumsum(cls._pinned_extend_lens[:bs], dim=0, dtype=torch.int64)
        cls._pinned_last_token_indices[:bs] = torch.clamp(cumsum - 1, min=0)
        forward_batch.last_token_indices = cls._pinned_last_token_indices[:bs].to(
          dev, non_blocking=True
        )
      forward_batch.extend_prefix_lens_cpu = batch.prefix_lens
      forward_batch.extend_seq_lens_cpu = extend_lens_cpu

      # 构造每个输入 token 的 position。
      # 如果某个请求 prefix_len=10，extend_len=3，则本轮新 token 的 position 是：
      # [10, 11, 12]。
      forward_batch.positions = compute_positions_extend(
        extend_prefix_lens=forward_batch.extend_prefix_lens,
        extend_seq_lens=forward_batch.extend_seq_lens,
        total_tokens=total_extend_tokens,
      )
    else:
      # DECODE 模式下，每个请求只输入 1 个 token。
      # 这个 token 的位置就是当前序列最后一个位置：seq_len - 1。
      forward_batch.positions = clamp_position(batch.seq_lens)

    return forward_batch

  def debug_metadata(self):
    logger.debug("ForwardBatch metadata:")
    logger.debug(f"forward_mode: {self.forward_mode}")
    logger.debug(f"attn_backend type: {self.attn_backend.type() if self.attn_backend else None}")
    logger.debug(f"batch_size: {self.batch_size}")
    logger.debug(f"input_ids: {self.input_ids}")
    logger.debug(f"req_pool_indices: {self.req_pool_indices}")
    logger.debug(f"out_cache_loc: {self.out_cache_loc}")
    logger.debug(f"seq_lens: {self.seq_lens}")
    logger.debug(f"extend_seq_lens: {self.extend_seq_lens}")
    logger.debug(f"extend_prefix_lens: {self.extend_prefix_lens}")


@dataclass
class BatchResult:
  """
  一次 batch forward + sample 后的结果容器。

  logits:
      模型输出的词表打分。可以先理解为“每个候选 token 的分数”。
  next_token_ids:
      Sampler 根据 logits 和 SamplingParams 选出来的下一个 token。

  LLMEngine._process_step_result() 会读取 BatchResult，把 next_token_ids
  写回对应 Req.output_ids，并判断请求是否完成。
  """

  logits: torch.Tensor
  next_token_ids: torch.Tensor


def compute_positions_extend(
  extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, total_tokens: int
) -> torch.Tensor:
  """
  为 EXTEND/MIXED batch 构造 positions。

  对每个请求 i:
    positions_i = [prefix_len_i, prefix_len_i+1, ..., prefix_len_i+extend_len_i-1]

  然后把所有请求的 positions 拼接起来，和拼接后的 input_ids 一一对应。

  举例:
    prefix_lens = [2, 5]
    extend_lens = [3, 2]

    第 0 个请求的新 token 位置是 [2, 3, 4]
    第 1 个请求的新 token 位置是 [5, 6]
    最终返回 [2, 3, 4, 5, 6]

  这些 position 会被 RoPE 使用，让模型知道每个 token 在序列中的位置。
  """
  if total_tokens <= 0:
    return torch.empty((0,), dtype=torch.int64, device=extend_prefix_lens.device)

  # Exclusive start offsets in the concatenated extend token array.
  start_offsets = torch.cumsum(extend_seq_lens, dim=0) - extend_seq_lens  # [bs]

  prefix_rep = torch.repeat_interleave(extend_prefix_lens, extend_seq_lens)  # [total]
  start_rep = torch.repeat_interleave(start_offsets, extend_seq_lens)  # [total]
  token_idx = torch.arange(total_tokens, device=extend_prefix_lens.device, dtype=torch.int64)
  within = token_idx - start_rep
  return (prefix_rep + within).to(torch.int64)


# Temporarily disable torch.compile to debug batch issues
# @torch.compile(dynamic=True)
def clamp_position(seq_lens):
  """
  为 DECODE batch 构造 positions。

  DECODE 阶段每个请求只输入 1 个 token。这个 token 的位置是当前序列最后一位：
    position = seq_len - 1

  clamp(min=0) 是防御性处理，避免 seq_len 意外为 0 时得到 -1。
  """
  return torch.clamp((seq_lens - 1), min=0).to(torch.int64)
