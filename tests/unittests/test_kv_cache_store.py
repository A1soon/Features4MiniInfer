import pytest
import torch

from kernels.triton.store import store_kv_cache
from miniinfer.kvcache.memory_pool import MHAKVCacheStorage


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_store_kv_cache_writes_selected_slots(dtype):
  torch.manual_seed(0)
  num_tokens, num_slots, num_heads, head_dim = 5, 11, 3, 70
  indices = torch.tensor([8, 2, 10, 4, 6], device=DEVICE, dtype=torch.int64)
  k = torch.randn((num_tokens, num_heads, head_dim), device=DEVICE, dtype=dtype)
  v = torch.randn_like(k)
  k_cache = torch.zeros((num_slots, num_heads, head_dim), device=DEVICE, dtype=dtype)
  v_cache = torch.zeros_like(k_cache)

  store_kv_cache(k_cache, v_cache, indices, k, v)

  expected_k = torch.zeros_like(k_cache)
  expected_v = torch.zeros_like(v_cache)
  expected_k[indices] = k
  expected_v[indices] = v
  torch.testing.assert_close(k_cache, expected_k, rtol=0, atol=0)
  torch.testing.assert_close(v_cache, expected_v, rtol=0, atol=0)


def test_store_kv_cache_supports_noncontiguous_inputs_and_cache_views():
  num_tokens, num_slots, num_heads, head_dim = 3, 9, 2, 17
  indices = torch.tensor([7, 1, 5], device=DEVICE, dtype=torch.int32)
  k = torch.randn((num_tokens, num_heads, head_dim * 2), device=DEVICE)[..., ::2]
  v = torch.randn((num_tokens, num_heads, head_dim * 3), device=DEVICE)[..., ::3]
  k_cache_base = torch.zeros((num_slots, num_heads, head_dim * 2), device=DEVICE)
  v_cache_base = torch.zeros((num_slots, num_heads, head_dim * 3), device=DEVICE)
  k_cache = k_cache_base[..., ::2]
  v_cache = v_cache_base[..., ::3]

  store_kv_cache(k_cache, v_cache, indices, k, v)

  expected_k = torch.zeros_like(k_cache)
  expected_v = torch.zeros_like(v_cache)
  expected_k[indices.long()] = k
  expected_v[indices.long()] = v
  torch.testing.assert_close(k_cache, expected_k, rtol=0, atol=0)
  torch.testing.assert_close(v_cache, expected_v, rtol=0, atol=0)


@pytest.mark.parametrize("device", [DEVICE])
def test_mha_storage_uses_kv_store_for_physical_locations(device):
  storage = MHAKVCacheStorage(
    size=32,
    page_size=4,
    num_layers=2,
    num_heads=2,
    head_dim=8,
    dtype=torch.float16,
    device=device,
  )
  indices = torch.tensor([2, 9, 17], device=device, dtype=torch.int64)
  k = torch.randn((3, 2, 8), device=device, dtype=torch.float16)
  v = torch.randn_like(k)

  storage.set_kv_buffer(layer_id=1, loc=indices, cache_k=k, cache_v=v)

  k_cache, v_cache = storage.get_kv_buffer(layer_id=1)
  torch.testing.assert_close(k_cache[indices], k, rtol=0, atol=0)
  torch.testing.assert_close(v_cache[indices], v, rtol=0, atol=0)
  assert torch.count_nonzero(k_cache[0]).item() == 0
  assert torch.count_nonzero(v_cache[0]).item() == 0
  untouched_k, untouched_v = storage.get_kv_buffer(layer_id=0)
  assert torch.count_nonzero(untouched_k).item() == 0
  assert torch.count_nonzero(untouched_v).item() == 0


def test_store_kv_cache_empty_batch_is_noop():
  k_cache = torch.zeros((4, 2, 8), device=DEVICE)
  v_cache = torch.zeros_like(k_cache)
  k = torch.empty((0, 2, 8), device=DEVICE)
  v = torch.empty_like(k)
  indices = torch.empty((0,), dtype=torch.int64, device=DEVICE)

  store_kv_cache(k_cache, v_cache, indices, k, v)

  assert torch.count_nonzero(k_cache).item() == 0
  assert torch.count_nonzero(v_cache).item() == 0
