"""SM70 forward attention using the pinned zhinianqin/flash-attention-v100 fork."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base_backend import AttentionBackend

if TYPE_CHECKING:
  from miniinfer.kvcache.kv_cache_manager import KVCacheManager


@dataclass
class FlashAttentionV100Metadata:
  cu_seqlens_q: torch.Tensor
  cache_seqlens: torch.Tensor
  block_table: torch.Tensor
  max_seqlen_q: int
  max_seqlen_k: int


class FlashAttentionV100Backend(AttentionBackend):
  """FP16 MHA/GQA self-attention; both extend and decode read paged NHD KV.

  MiniInfer writes the current token(s) before attention, so seqused_k includes
  them. Q contains only the new suffix. The kernel's bottom-right causal mask
  aligns that suffix with the full cached sequence, including prefix hits and
  one-token decode requests in a mixed batch. No full-prefix KV gather is needed.
  """

  def __init__(self, kv_cache_mgr: "KVCacheManager"):
    if kv_cache_mgr.dtype != torch.float16:
      raise ValueError("flash_attn_v100 requires FP16 KV cache and model weights")
    if kv_cache_mgr.page_size <= 0 or kv_cache_mgr.page_size % 16:
      raise ValueError("flash_attn_v100 page_size must be a positive multiple of 16")
    if kv_cache_mgr.head_dim not in (32, 64, 96, 128, 192, 256):
      raise ValueError("flash_attn_v100 supports head_dim 32, 64, 96, 128, 192, 256")
    device = torch.device(kv_cache_mgr.device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 0):
      raise ValueError("flash_attn_v100 requires a CUDA SM70 (V100) device")
    from vllm_flash_attn import flash_attn_varlen_func

    self._attention = flash_attn_varlen_func
    self.kv_cache_mgr = kv_cache_mgr
    self.page_size = kv_cache_mgr.page_size
    self.forward_metadata = None
    self._graph_metadata = {}
    self._extend_graph_metadata = {}

  def type(self) -> str:
    return "flash_attn_v100"

  def init_forward_metadata(self, forward_batch):
    bs = forward_batch.batch_size
    max_len = forward_batch.max_seq_len
    if max_len is None:
      cpu_lens = forward_batch.seq_lens_cpu
      if cpu_lens is None:
        raise ValueError("flash_attn_v100 requires CPU-side max_seq_len or seq_lens_cpu")
      max_len = int(max(cpu_lens))
    if forward_batch.forward_mode.is_decode():
      q_lens = [1] * bs
    else:
      q_lens = forward_batch.extend_seq_lens_cpu
      if q_lens is None:
        raise ValueError("flash_attn_v100 extend requires extend_seq_lens_cpu")
    if len(q_lens) != bs or any(length <= 0 for length in q_lens):
      raise ValueError("flash_attn_v100 requires one positive query length per request")
    # RequestPool stores token slots. One slot at each page boundary identifies
    # the physical page; page 0 remains the allocator's reserved padding page.
    slots = self.kv_cache_mgr.get_page_table(forward_batch, max_len)
    # Singleton columns can be "contiguous" while retaining stride(-1)>1.
    # The extension requires a literal unit stride, including one-page tables.
    table = (slots[:, :: self.page_size] // self.page_size).to(torch.int32).clone(
      memory_format=torch.contiguous_format
    )
    lengths = forward_batch.seq_lens_int32
    if lengths is None:
      lengths = forward_batch.seq_lens.to(torch.int32)
    cu_q = torch.zeros(bs + 1, dtype=torch.int32, device=lengths.device)
    cu_q[1:] = torch.tensor(q_lens, dtype=torch.int32, device=lengths.device).cumsum(0)
    self.forward_metadata = FlashAttentionV100Metadata(
      cu_q, lengths, table, max(q_lens), int(max_len)
    )

  def _forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
    metadata = self.forward_metadata
    if metadata is None:
      raise RuntimeError("init_forward_metadata must be called before attention")
    if (
      layer.head_dim != self.kv_cache_mgr.head_dim
      or layer.tp_k_head_num != self.kv_cache_mgr.num_heads
    ):
      raise ValueError("Attention layer dimensions do not match the V100 KV cache layout")
    if (
      layer.tp_k_head_num != layer.tp_v_head_num
      or layer.head_dim != layer.v_head_dim
      or layer.tp_q_head_num % layer.tp_k_head_num
    ):
      raise ValueError("flash_attn_v100 requires matching K/V dimensions and divisible Q/KV heads")
    if q.dtype != torch.float16 or (k is not None and k.dtype != q.dtype):
      raise ValueError("flash_attn_v100 requires FP16 Q/K/V")
    if (k is None) != (v is None):
      raise ValueError("K and V must be provided together")
    if v is not None and v.dtype != q.dtype:
      raise ValueError("flash_attn_v100 requires FP16 Q/K/V")
    if k is not None and save_kv_cache:
      self.kv_cache_mgr.set_kv_buffer(
        layer.layer_id,
        forward_batch.out_cache_loc,
        k.reshape(-1, layer.tp_k_head_num, layer.head_dim),
        v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim),
      )
    key_cache, value_cache = self.kv_cache_mgr.get_kv_buffer(layer.layer_id)
    out = self._attention(
      q=q.reshape(-1, layer.tp_q_head_num, layer.head_dim).contiguous(),
      k=key_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim),
      v=value_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim),
      cu_seqlens_q=metadata.cu_seqlens_q,
      seqused_k=metadata.cache_seqlens,
      block_table=metadata.block_table,
      max_seqlen_q=metadata.max_seqlen_q,
      max_seqlen_k=metadata.max_seqlen_k,
      softmax_scale=layer.scaling,
      causal=True,
      **kwargs,
    )
    return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

  forward_extend = _forward
  forward_decode = _forward

  def init_cuda_graph_metadata(self, batch_size, block_table, cache_seqlens, max_seq_len_k):
    # Graphs freeze launch-time scalar bounds. Use full table capacity, not the
    # warmup length (usually 2), so replay can grow across a physical page edge.
    metadata = FlashAttentionV100Metadata(
      torch.arange(batch_size + 1, dtype=torch.int32, device=cache_seqlens.device),
      cache_seqlens,
      block_table,
      1,
      block_table.shape[1] * self.page_size,
    )
    self._graph_metadata[batch_size] = metadata
    self.forward_metadata = metadata

  def update_cuda_graph_metadata(self, cache_seqlens, max_seq_len_k):
    metadata = self._graph_metadata[cache_seqlens.shape[0]]
    if max_seq_len_k > metadata.max_seqlen_k:
      raise ValueError("Decode sequence exceeds captured V100 page-table capacity")
    if cache_seqlens is not metadata.cache_seqlens:
      metadata.cache_seqlens.copy_(cache_seqlens)
    self.forward_metadata = metadata

  def init_extend_cuda_graph_metadata(
    self, bucket, cu_seqlens_q, cache_seqlens, block_table, max_seqlen_q
  ):
    metadata = FlashAttentionV100Metadata(
      cu_seqlens_q,
      cache_seqlens,
      block_table,
      max_seqlen_q,
      block_table.shape[1] * self.page_size,
    )
    self._extend_graph_metadata[bucket] = metadata
    self.forward_metadata = metadata

  def update_extend_cuda_graph_metadata(self, bucket):
    self.forward_metadata = self._extend_graph_metadata[bucket]
