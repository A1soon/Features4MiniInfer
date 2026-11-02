"""CPU-only tests for request history and committed-KV bookkeeping."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path


def _load_scheduler_batch():
  """Load Req without importing MiniInfer's CUDA/Triton package entrypoint."""
  module_path = Path(__file__).parents[2] / "miniinfer" / "scheduler" / "scheduler_batch.py"
  module_name = "_miniinfer_scheduler_batch_retract_test"

  fake_package = types.ModuleType("miniinfer")
  fake_package.__path__ = []
  fake_layers = types.ModuleType("miniinfer.layers")
  fake_layers.__path__ = []
  fake_attention = types.ModuleType("miniinfer.layers.attention_backend")
  fake_attention.__path__ = []
  fake_backend = types.ModuleType("miniinfer.layers.attention_backend.base_backend")
  fake_backend.AttentionBackend = object
  fake_utils = types.ModuleType("miniinfer.utils")
  fake_utils.get_logger = lambda name: logging.getLogger(name)
  fake_sampling = types.ModuleType("miniinfer.utils.sampling_params")

  class SamplingParams:
    def __init__(
      self,
      temperature: float = 1.0,
      max_tokens: int = 1,
      ignore_eos: bool = False,
      top_k: int = 0,
      top_p: float = 1.0,
    ):
      self.temperature = temperature
      self.max_tokens = max_tokens
      self.ignore_eos = ignore_eos
      self.top_k = top_k
      self.top_p = top_p

  fake_sampling.SamplingParams = SamplingParams

  names = (
    "miniinfer",
    "miniinfer.layers",
    "miniinfer.layers.attention_backend",
    "miniinfer.layers.attention_backend.base_backend",
    "miniinfer.utils",
    "miniinfer.utils.sampling_params",
  )
  saved = {name: sys.modules.get(name) for name in names}
  sys.modules.update(
    {
      "miniinfer": fake_package,
      "miniinfer.layers": fake_layers,
      "miniinfer.layers.attention_backend": fake_attention,
      "miniinfer.layers.attention_backend.base_backend": fake_backend,
      "miniinfer.utils": fake_utils,
      "miniinfer.utils.sampling_params": fake_sampling,
    }
  )
  try:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
  finally:
    sys.modules.pop(module_name, None)
    for name, previous in saved.items():
      if previous is None:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = previous


scheduler_batch = _load_scheduler_batch()
Req = scheduler_batch.Req


def test_sampled_token_is_not_committed_until_next_forward():
  req = Req([1, 2, 3])

  assert req.kv_committed_len == 0
  assert req.committed_token_ids == []

  # The prompt forward writes prompt KV, then sampling produces token 9.
  req.mark_kv_committed(req.current_seq_len)
  req.append_output_token(9)
  assert req.current_seq_len == 4
  assert req.kv_committed_len == 3
  assert req.committed_token_ids == [1, 2, 3]

  # The next forward consumes token 9 and makes its KV reusable.
  req.mark_kv_committed(req.current_seq_len)
  req.append_output_token(10)
  assert req.kv_committed_len == 4
  assert req.committed_token_ids == [1, 2, 3, 9]


def test_retract_reset_preserves_logical_history_but_drops_cache_metadata():
  req = Req([1, 2, 3])
  req.mark_kv_committed(3)
  req.append_output_token(9)
  req.mark_kv_committed(4)
  req.append_output_token(10)
  req.prefix_indices = object()
  req.cache_protected_len = 4
  req.last_node = object()
  req.req_pool_idx = 7
  req.extend_input_len = 2

  req.reset_cache_state_after_retract()

  assert req.output_ids == [9, 10]
  assert req.current_seq_len == 5
  assert req.get_last_token() == 10
  assert req.committed_token_ids == []
  assert req.req_pool_idx == -1
  assert req.prefix_indices is None
  assert req.cache_protected_len == 0
  assert req.last_node is None
  assert req.extend_input_len == 0
  assert req.kv_committed_len == 0
