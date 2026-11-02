"""Write per-token K/V tensors into their allocated physical KV slots."""

import torch
import triton
import triton.language as tl


@triton.jit
def store_kv_cache_kernel(
  k_ptr,
  v_ptr,
  k_cache_ptr,
  v_cache_ptr,
  indices_ptr,
  stride_k_token,
  stride_k_head,
  stride_k_dim,
  stride_v_token,
  stride_v_head,
  stride_v_dim,
  stride_k_cache_slot,
  stride_k_cache_head,
  stride_k_cache_dim,
  stride_v_cache_slot,
  stride_v_cache_head,
  stride_v_cache_dim,
  num_tokens,
  num_heads,
  head_dim,
  BLOCK_D: tl.constexpr,
):
  token_idx = tl.program_id(0)
  head_idx = tl.program_id(1)
  dim_start = tl.program_id(2) * BLOCK_D
  offsets_d = dim_start + tl.arange(0, BLOCK_D)
  mask_d = offsets_d < head_dim

  target_slot = tl.load(indices_ptr + token_idx)

  k_offset = (
    token_idx * stride_k_token + head_idx * stride_k_head + offsets_d * stride_k_dim
  )
  v_offset = (
    token_idx * stride_v_token + head_idx * stride_v_head + offsets_d * stride_v_dim
  )
  k_cache_offset = (
    target_slot * stride_k_cache_slot
    + head_idx * stride_k_cache_head
    + offsets_d * stride_k_cache_dim
  )
  v_cache_offset = (
    target_slot * stride_v_cache_slot
    + head_idx * stride_v_cache_head
    + offsets_d * stride_v_cache_dim
  )

  k_values = tl.load(k_ptr + k_offset, mask=mask_d, other=0)
  v_values = tl.load(v_ptr + v_offset, mask=mask_d, other=0)
  tl.store(k_cache_ptr + k_cache_offset, k_values, mask=mask_d)
  tl.store(v_cache_ptr + v_cache_offset, v_values, mask=mask_d)


def store_kv_cache(k_cache, v_cache, indices, k, v):
  """Store K/V with shapes [tokens, heads, head_dim] at physical slot IDs.

  ``indices[token]`` chooses the cache row for each token. The selected slots
  must be unique within this call, as they are for allocator-produced locations.
  Tensor strides are passed to Triton, so inputs and cache views need not be
  contiguous.
  """
  if k.ndim != 3 or v.ndim != 3:
    raise ValueError(f"K and V must have shape [tokens, heads, head_dim], got {k.shape}, {v.shape}")
  if k.shape != v.shape:
    raise ValueError(f"K and V shapes must match, got {k.shape} and {v.shape}")
  if k_cache.ndim != 3 or v_cache.ndim != 3:
    raise ValueError("K and V cache tensors must have shape [slots, heads, head_dim]")
  if k_cache.shape[1:] != k.shape[1:] or v_cache.shape[1:] != v.shape[1:]:
    raise ValueError("Input and cache head dimensions must match")
  if k_cache.shape[1:] != v_cache.shape[1:]:
    raise ValueError("K and V cache head dimensions must match")
  if indices.ndim != 1 or indices.numel() != k.shape[0]:
    raise ValueError("indices must be one-dimensional with one slot ID per input token")
  if k.shape[0] == 0:
    return

  tensors = (v, k_cache, v_cache, indices)
  if any(t.device != k.device for t in tensors):
    raise ValueError("K, V, caches, and indices must be on the same device")
  if k.dtype != v.dtype or k.dtype != k_cache.dtype or v.dtype != v_cache.dtype:
    raise ValueError("K, V, and their cache buffers must have matching dtypes")

  if not k.is_cuda:
    k_cache[indices.to(torch.long)] = k
    v_cache[indices.to(torch.long)] = v
    return

  num_tokens, num_heads, head_dim = k.shape
  block_d = min(triton.next_power_of_2(head_dim), 128)
  grid = (num_tokens, num_heads, triton.cdiv(head_dim, block_d))
  store_kv_cache_kernel[grid](
    k,
    v,
    k_cache,
    v_cache,
    indices,
    *k.stride(),
    *v.stride(),
    *k_cache.stride(),
    *v_cache.stride(),
    num_tokens,
    num_heads,
    head_dim,
    BLOCK_D=block_d,
  )
