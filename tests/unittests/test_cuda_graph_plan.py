"""CPU-only invariants for CUDA Graph bucket and padding planning."""

import pytest
import torch
import gc
import weakref
from types import SimpleNamespace

from miniinfer.engine.cuda_graph_runner import CudaGraphRunner, retain_moe_workspaces
from miniinfer.layers.fused_moe import FusedMoE
from miniinfer.engine.piecewise_cuda_graph import PiecewiseCudaGraphRunner
from miniinfer.layers.attention_backend.flashattention_v100_backend import (
  FlashAttentionV100Backend,
)
from miniinfer.scheduler.scheduler_batch import ForwardMode


@pytest.mark.parametrize(
  ("maximum", "expected"),
  [(1, [1]), (3, [1, 2, 3]), (8, [1, 2, 4, 8]), (10, [1, 2, 4, 8, 10])],
)
def test_capture_buckets_are_deterministic(maximum, expected):
  assert CudaGraphRunner.capture_batch_sizes(maximum) == expected


@pytest.mark.parametrize(
  ("maximum", "batch", "expected"),
  [(8, 1, 1), (8, 3, 4), (8, 8, 8), (10, 9, 10)],
)
def test_padded_batch_uses_existing_final_bucket(maximum, batch, expected):
  assert CudaGraphRunner.padded_batch_size(maximum, batch) == expected


def test_graph_bucket_validation_does_not_silently_drop_requests():
  with pytest.raises(ValueError, match="positive"):
    CudaGraphRunner.capture_batch_sizes(0)
  with pytest.raises(ValueError, match="exceeds"):
    CudaGraphRunner.padded_batch_size(4, 5)


def test_prefill_graph_buckets_start_at_page_size_and_respect_limit():
  assert PiecewiseCudaGraphRunner.capture_buckets(256, 1024) == [256, 512, 1024]
  assert PiecewiseCudaGraphRunner.capture_buckets(256, 400) == [256, 400]


def test_prefill_graph_layout_pads_queries_and_preserves_last_token_rows():
  assert PiecewiseCudaGraphRunner.padded_query_lengths([7, 3], 16, 2) == [7, 3, 6]
  assert PiecewiseCudaGraphRunner.padded_query_lengths([15], 16, 2) is None
  assert PiecewiseCudaGraphRunner.padded_query_lengths([4, 3], 8, 2) == [4, 3, 1]


def test_v100_extend_graph_metadata_keeps_static_tensor_references():
  backend = FlashAttentionV100Backend.__new__(FlashAttentionV100Backend)
  backend.page_size = 16
  backend._extend_graph_metadata = {}
  cu_q = torch.tensor([0, 8], dtype=torch.int32)
  cache_lens = torch.tensor([8], dtype=torch.int32)
  block_table = torch.zeros((1, 4), dtype=torch.int32)
  backend.init_extend_cuda_graph_metadata(8, cu_q, cache_lens, block_table, 8)
  metadata = backend.forward_metadata
  assert metadata.cu_seqlens_q is cu_q
  assert metadata.cache_seqlens is cache_lens
  assert metadata.block_table is block_table
  assert metadata.max_seqlen_k == 64
  backend.forward_metadata = None
  backend.update_extend_cuda_graph_metadata(8)
  assert backend.forward_metadata is metadata


def test_graph_keeps_external_moe_workspace_alive_after_larger_bucket():
  model = FusedMoE(8, 2, 16, 24, dtype=torch.float32, device="cpu")
  workspace = model.experts._ensure_workspace(8, torch.device("cpu"), torch.float32)
  old_storage = weakref.ref(workspace)
  graph_buffers = retain_moe_workspaces(model)
  assert graph_buffers == (workspace,)
  bigger = model.experts._ensure_workspace(64, torch.device("cpu"), torch.float32)
  assert bigger is not workspace
  del workspace
  gc.collect()
  assert old_storage() is graph_buffers[0]
  del graph_buffers
  gc.collect()
  assert old_storage() is None


@pytest.mark.parametrize("batch_size", [1, 3])
def test_v100_single_page_table_has_literal_unit_stride(batch_size):
  backend = FlashAttentionV100Backend.__new__(FlashAttentionV100Backend)
  backend.page_size = 256
  slots = torch.arange(batch_size * 16).reshape(batch_size, 16) + 256
  backend.kv_cache_mgr = SimpleNamespace(get_page_table=lambda batch, maximum: slots)
  batch = SimpleNamespace(batch_size=batch_size, max_seq_len=16,
                          forward_mode=ForwardMode.EXTEND, extend_seq_lens_cpu=[16] * batch_size,
                          seq_lens_int32=torch.full((batch_size,), 16, dtype=torch.int32))
  backend.init_forward_metadata(batch)
  assert backend.forward_metadata.block_table.stride(-1) == 1
  torch.testing.assert_close(backend.forward_metadata.block_table,
                             (slots[:, ::256] // 256).int())
