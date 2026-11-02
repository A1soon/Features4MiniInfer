from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoConfig
import torch


@dataclass
class EngineConfig:
  """
  推理引擎配置

  显存管理相关参数:
  - gpu_memory_utilization: GPU 显存利用率 (0.0 ~ 1.0)，用于计算 KV cache 可用容量
  - max_total_tokens: 手动指定 KV cache 可容纳的最大 token 数（覆盖自动计算）
  - max_num_batched_tokens: 单次推理的最大 token 数（用于 prefill 调度）
  - num_pages: 手动指定 KV cache 页面数（覆盖自动计算）
  - page_size: 每页包含的 token 数
  """

  model_path: str
  dtype: torch.dtype = torch.float16
  # 默认 flashinfer：无 flash_attn 的 page_size-必须被 256 整除约束，且是 vLLM-grade
  # paged decode attention。如需切回 FA2，传 attention_backend="flash_attn"。
  # 注：benchmark 显示二者在 page_size=256 下吞吐接近（见
  # docs/性能优化_FA2对比_与长序列attention瓶颈.md §8）。
  attention_backend: str = "flashinfer"
  # V100: select attention_backend="flash_attn_v100" explicitly (FP16, page_size % 16 == 0).

  # ============ 显存管理配置 ============
  # GPU 显存利用率，用于自动计算 KV cache 容量
  # 默认 0.9（与 vLLM 一致）：长序列/大 batch 下避免 KV 不足导致 retraction 或
  # 分配失败。0.6 在 256 seq × 长序列下会 KV 耗尽。如需保守可下调。
  gpu_memory_utilization: float = 0.9
  # 每页 token 数
  page_size: int = 256

  # ============ 调度配置 ============
  max_num_seqs: int = 256
  max_context_len: int = 4096
  max_extend_len: int = 8192  # for chunked prefill
  chunked_prefill_size: int = 4096
  # ``fixed`` preserves the current scheduler behavior. ``aggregate`` adds
  # a per-iteration compute budget shared by prefill and decode.
  prefill_budget_policy: str = "fixed"
  max_num_batched_tokens: Optional[int] = None
  enable_prefix_cache: bool = True
  enable_chunked_prefill: bool = False
  # Preserve output history across KV-driven retracts.  The first correctness
  # implementation is intentionally restricted to the non-overlap path.
  enable_retract_state_preservation: bool = False

  # ============ 并行配置 ============
  tensor_parallel_size: int = 1
  expert_parallel_size: int = 1
  ep_timeout_seconds: int = 120
  ep_startup_timeout_seconds: int = 1800
  max_total_tokens: Optional[int] = None

  # ============ 其他配置 ============
  enforce_eager: bool = False
  hf_config: Optional[AutoConfig] = field(default=None, repr=False)
  eos: int = -1

  # ============ CUDA Graph 配置 ============
  # Whether to use decode CUDA Graphs; EP remains eager unless explicitly opted in below.
  # cuda_graph_max_bs: 0 表示使用 max_num_seqs
  cuda_graph_max_bs: int = 0
  # Experimental EP decode-graph path; keeps EP eager/overlap guards in force.
  enable_ep_cuda_graph: bool = False

  # ============ Overlap 配置 ============
  # 是否启用双 Batch 交替执行 (overlap scheduling)
  # 启用后 GPU 执行当前 batch 时 CPU 并行处理上一 batch
  enable_overlap: bool = True

  # 别名支持（兼容旧配置）
  @property
  def model(self) -> str:
    return self.model_path

  @property
  def memory_ratio(self) -> float:
    """兼容旧的 memory_ratio 参数名"""
    return self.gpu_memory_utilization

  def __post_init__(self):
    if self.attention_backend == "flash_attn_v100":
      if (
        not isinstance(self.page_size, int)
        or isinstance(self.page_size, bool)
        or self.page_size <= 0
        or self.page_size % 16
      ):
        raise ValueError("flash_attn_v100 page_size must be a positive multiple of 16")
    elif (
      not isinstance(self.page_size, int) or isinstance(self.page_size, bool) or self.page_size <= 0
    ):
      raise ValueError("page_size must be a positive integer")
    for name in (
      "tensor_parallel_size",
      "expert_parallel_size",
      "ep_timeout_seconds",
      "ep_startup_timeout_seconds",
    ):
      value = getattr(self, name)
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if self.tensor_parallel_size != 1:
      raise ValueError("Engine Attention TP is not implemented; tensor_parallel_size must be 1")
    if self.max_total_tokens is not None:
      if (
        not isinstance(self.max_total_tokens, int)
        or isinstance(self.max_total_tokens, bool)
        or self.max_total_tokens < 2 * self.page_size
      ):
        raise ValueError("max_total_tokens must be an integer covering at least two pages")
    if self.expert_parallel_size > 1 and (not self.enforce_eager or self.enable_overlap):
      raise ValueError("EP currently requires enforce_eager=True and enable_overlap=False")
    if self.enable_ep_cuda_graph and self.expert_parallel_size <= 1:
      raise ValueError("enable_ep_cuda_graph requires expert_parallel_size > 1")
    if self.expert_parallel_size > 1:
      # EP histories must survive KV pressure so every replica resumes the same request.
      self.enable_retract_state_preservation = True
    if self.attention_backend == "flash_attn_v100":
      if self.dtype != torch.float16:
        raise ValueError("flash_attn_v100 requires dtype=torch.float16")
      if self.page_size % 16:
        raise ValueError("flash_attn_v100 page_size must be a positive multiple of 16")
      if self.enable_chunked_prefill and self.chunked_prefill_size < self.page_size:
        raise ValueError("flash_attn_v100 chunked_prefill_size must cover at least one page")
    # 加载 HuggingFace 配置
    self.hf_config = AutoConfig.from_pretrained(self.model_path)
    if self.expert_parallel_size > 1:
      if getattr(self.hf_config, "model_type", "") != "qwen3_moe":
        raise ValueError("EP currently supports Qwen3-MoE only")
      if self.hf_config.num_experts % self.expert_parallel_size:
        raise ValueError("num_experts must be divisible by expert_parallel_size")

    # 限制 max_context_len 不超过模型支持的最大位置编码
    self.max_context_len = min(self.max_context_len, self.hf_config.max_position_embeddings)

    if self.prefill_budget_policy not in {"fixed", "aggregate"}:
      raise ValueError(
        "prefill_budget_policy must be either 'fixed' or 'aggregate', "
        f"got {self.prefill_budget_policy!r}"
      )
    if self.max_num_batched_tokens is not None and self.max_num_batched_tokens <= 0:
      raise ValueError("max_num_batched_tokens must be positive when provided")
    if self.prefill_budget_policy == "aggregate":
      if self.max_num_batched_tokens is None:
        raise ValueError(
          "max_num_batched_tokens is required when prefill_budget_policy='aggregate'"
        )
      if not self.enable_chunked_prefill:
        raise ValueError("aggregate prefill budget currently requires enable_chunked_prefill=True")
      if self.max_num_batched_tokens < self.max_num_seqs:
        raise ValueError(
          "aggregate max_num_batched_tokens must be at least max_num_seqs "
          "so a decode-only batch can satisfy P_t + D_t <= B"
        )
    if self.enable_retract_state_preservation and self.enable_overlap:
      raise ValueError("enable_retract_state_preservation currently requires enable_overlap=False")
