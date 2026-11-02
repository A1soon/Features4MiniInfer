"""CPU-only tests for the L2 prefill budget policies.

These tests load ``PrefillAdder`` in isolation so they do not require Triton,
CUDA, or a Hugging Face model.  The scheduler integration is exercised on the
GPU machine; this file focuses on the arithmetic contract of one iteration.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path


def _load_prefill_adder():
  """Load the scheduler module without importing MiniInfer's CUDA stack."""
  module_path = Path(__file__).parents[2] / "miniinfer" / "scheduler" / "prefill_adder.py"
  module_name = "_miniinfer_prefill_adder_budget_test"

  fake_package = types.ModuleType("miniinfer")
  fake_package.__path__ = []
  fake_utils = types.ModuleType("miniinfer.utils")
  fake_utils.get_logger = lambda name: logging.getLogger(name)

  saved = {
    name: sys.modules.get(name)
    for name in ("miniinfer", "miniinfer.utils")
  }
  sys.modules["miniinfer"] = fake_package
  sys.modules["miniinfer.utils"] = fake_utils
  try:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module
  finally:
    for name, previous in saved.items():
      if previous is None:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = previous


prefill_adder = _load_prefill_adder()
AddReqResult = prefill_adder.AddReqResult
PrefillAdder = prefill_adder.PrefillAdder


class DummyReq:
  def __init__(self, extend_len: int, *, max_tokens: int = 0, prefix_len: int = 0):
    self.extend_input_len = extend_len
    self.max_tokens = max_tokens
    self.output_ids = []
    self.prefix_indices = list(range(prefix_len))
    self.fill_ids = list(range(prefix_len + extend_len))
    self.is_chunked = False


def _adder(*, compute_budget=None, available_kv=8192, chunk_size=4096, max_extend=4096):
  return PrefillAdder(
    available_kv_tokens=available_kv,
    running_reqs=[],
    new_token_ratio=0.0,
    max_extend_len=max_extend,
    max_batch_size=16,
    chunk_size=chunk_size,
    page_size=256,
    compute_budget=compute_budget,
  )


def test_fixed_policy_keeps_compute_budget_unbounded():
  adder = _adder(compute_budget=None, chunk_size=None)
  req = DummyReq(extend_len=600)

  assert adder.try_add_prefill(req) is AddReqResult.CONTINUE
  assert adder.rem_compute_tokens is None
  # KV/input accounting is page based: ceil(600 / 256) * 256 == 768.
  assert adder.rem_input_tokens == 4096 - 768
  assert adder.log_input_tokens == 768


def test_aggregate_exact_fit_uses_actual_compute_tokens():
  adder = _adder(compute_budget=992)
  req = DummyReq(extend_len=992)

  assert adder.try_add_prefill(req) is AddReqResult.CONTINUE
  assert adder.rem_compute_tokens == 0
  # The model computes 992 tokens, even though KV allocation rounds to 1024.
  assert adder.log_input_tokens == 1024
  assert adder.rem_input_tokens == 4096 - 1024


def test_decode_reserve_is_subtracted_from_aggregate_budget():
  # B=1024 and D=32 leaves 992 prefill compute tokens.
  adder = _adder(compute_budget=992)
  req = DummyReq(extend_len=992)

  assert adder.try_add_prefill(req) is AddReqResult.CONTINUE
  assert adder.rem_compute_tokens == 0
  assert req.extend_input_len == 992


def test_running_request_reserve_is_kv_accounting_not_compute_budget():
  running = DummyReq(extend_len=0, max_tokens=100)
  adder = PrefillAdder(
    available_kv_tokens=8192,
    running_reqs=[running],
    new_token_ratio=0.5,
    max_extend_len=4096,
    max_batch_size=16,
    chunk_size=4096,
    page_size=256,
    compute_budget=1024,
  )

  # 50 tokens are reserved in KV accounting for future decode.  The
  # aggregate compute budget remains an independent 1024-token limit.
  assert adder.rem_total_tokens == 8192 - 50
  assert adder.rem_compute_tokens == 1024


def test_two_requests_never_exceed_aggregate_compute_budget():
  adder = _adder(compute_budget=992)
  first = DummyReq(extend_len=600)
  second = DummyReq(extend_len=600)

  assert adder.try_add_prefill(first) is AddReqResult.CONTINUE
  assert adder.try_add_prefill(second) is AddReqResult.CONTINUE

  # First request uses 600 actual tokens.  Only 392 remain; the second
  # request is chunked to one page (256), so P_t = 856 <= 992.
  assert first.extend_input_len == 600
  assert second.extend_input_len == 256
  assert second.is_chunked is True
  assert adder.rem_compute_tokens == 136
  assert sum(req.extend_input_len for req in adder.can_run_list) <= 992


def test_zero_compute_budget_rejects_prefill_even_when_kv_is_available():
  adder = _adder(compute_budget=0)
  req = DummyReq(extend_len=256)

  assert adder.try_add_prefill(req) is AddReqResult.NO_TOKEN
  assert adder.can_run_list == []


def test_kv_budget_and_compute_budget_are_independent_limits():
  # KV has less than one page: compute capacity alone must not permit a run.
  adder = _adder(compute_budget=4096, available_kv=128)
  req = DummyReq(extend_len=128)

  assert adder.try_add_prefill(req) is AddReqResult.NO_TOKEN
  assert adder.can_run_list == []


def test_continuation_chunk_respects_remaining_compute_budget():
  adder = _adder(compute_budget=392)
  req = DummyReq(extend_len=600)

  remaining = adder.add_chunked_req(req)

  assert remaining is req
  assert req.extend_input_len == 256
  assert req.is_chunked is True
  assert adder.rem_compute_tokens == 136
  assert len(adder.can_run_list) == 1


def test_continuation_with_less_than_one_page_budget_waits():
  adder = _adder(compute_budget=255)
  req = DummyReq(extend_len=600)

  assert adder.add_chunked_req(req) is req
  assert req.extend_input_len == 0
  assert adder.can_run_list == []
  assert adder.rem_compute_tokens == 255


def test_page_alignment_does_not_inflate_compute_accounting():
  adder = _adder(compute_budget=257)
  req = DummyReq(extend_len=257)

  assert adder.try_add_prefill(req) is AddReqResult.CONTINUE
  assert adder.rem_compute_tokens == 0
  # KV uses two pages, but the aggregate invariant uses actual model work.
  assert adder.log_input_tokens == 512
  assert req.extend_input_len == 257


def test_input_budget_is_soft_after_a_request_has_been_added():
  adder = _adder(compute_budget=None, chunk_size=None, max_extend=1024)
  first = DummyReq(extend_len=600)
  second = DummyReq(extend_len=600)

  assert adder.try_add_prefill(first) is AddReqResult.CONTINUE
  # 600 rounds to 768, leaving 256 input-budget tokens.  The second request
  # remains for a later iteration instead of being partially scheduled.
  assert adder.try_add_prefill(second) is AddReqResult.OTHER
  assert len(adder.can_run_list) == 1
