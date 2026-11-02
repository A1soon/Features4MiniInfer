import pytest
import torch
from unittest.mock import MagicMock
from miniinfer.layers.attention_backend.flashinfer_backend import FlashInferBackend
from miniinfer.scheduler.scheduler_batch import ForwardBatch, ForwardMode


@pytest.fixture
def mock_kv_cache_mgr():
  mgr = MagicMock()
  mgr.page_size = 16
  mgr.device = "cuda" if torch.cuda.is_available() else "cpu"
  # get_page_table returns token slots: row 0 uses pages 0/1, row 1 uses 2/3.
  # Only the first 20 positions are needed for this batch's maximum length.
  mgr.get_page_table.return_value = torch.arange(
    64, device=mgr.device, dtype=torch.int64
  ).reshape(2, 32)[:, :20]
  return mgr


@pytest.fixture
def mock_forward_batch():
  batch = MagicMock(spec=ForwardBatch)
  batch.batch_size = 2
  batch.seq_lens = torch.tensor([10, 20], device="cuda")
  # In Extend, max_seq_len is usually max(seq_lens)
  batch.max_seq_len = 20
  batch.seq_lens_cpu = torch.tensor([10, 20], dtype=torch.int32)
  batch.req_pool_indices = torch.tensor([0, 1], device="cuda")
  batch.forward_mode = ForwardMode.EXTEND
  batch.extend_prefix_lens_cpu = [0, 0]
  batch.extend_seq_lens_cpu = [10, 20]
  batch.extend_seq_lens = torch.tensor([10, 20], device="cuda")
  return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer requires CUDA")
def test_flashinfer_init_metadata_extend(mock_kv_cache_mgr, mock_forward_batch):
  backend = FlashInferBackend(mock_kv_cache_mgr)

  backend.init_forward_metadata(mock_forward_batch)

  metadata = backend.forward_metadata
  assert metadata is not None

  # Check paged_kv_indptr
  # 0, 1, 3 (seq 10 -> 1 page, seq 20 -> 2 pages)
  expected_indptr = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
  torch.testing.assert_close(metadata.paged_kv_indptr, expected_indptr)

  # Check last_page_len
  # 10 % 16 = 10. (10-1)%16 + 1 = 10
  # 20 % 16 = 4.
  expected_last_len = torch.tensor([10, 4], device="cuda", dtype=torch.int32)
  torch.testing.assert_close(metadata.paged_kv_last_page_len, expected_last_len)

  # Slots at offsets 0/16 identify each page. Length 10 excludes row 0's page 1.
  expected_indices = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
  torch.testing.assert_close(metadata.paged_kv_indices, expected_indices)

  # Check qo_indptr (for extend)
  # [0, 10, 30]
  expected_qo = torch.tensor([0, 10, 30], device="cuda", dtype=torch.int32)
  torch.testing.assert_close(metadata.qo_indptr, expected_qo)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer requires CUDA")
def test_flashinfer_init_metadata_decode(mock_kv_cache_mgr, mock_forward_batch):
  mock_forward_batch.forward_mode = ForwardMode.DECODE

  backend = FlashInferBackend(mock_kv_cache_mgr)
  backend.init_forward_metadata(mock_forward_batch)

  metadata = backend.forward_metadata

  # Decode mode doesn't set qo_indptr usually, or sets it differently?
  # Extend meta logic was conditional on !decode.
  assert metadata.qo_indptr is None

  # paged_kv.. should be same
  expected_indptr = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
  torch.testing.assert_close(metadata.paged_kv_indptr, expected_indptr)
  expected_indices = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
  torch.testing.assert_close(metadata.paged_kv_indices, expected_indices)
  expected_last_len = torch.tensor([10, 4], device="cuda", dtype=torch.int32)
  torch.testing.assert_close(metadata.paged_kv_last_page_len, expected_last_len)
