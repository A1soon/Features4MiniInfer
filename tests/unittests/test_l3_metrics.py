"""CPU-only tests for the L3 metrics recorder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_metrics():
  module_path = Path(__file__).parents[2] / "miniinfer" / "engine" / "metrics.py"
  spec = importlib.util.spec_from_file_location("_miniinfer_l3_metrics_test", module_path)
  module = importlib.util.module_from_spec(spec)
  assert spec is not None and spec.loader is not None
  sys.modules[spec.name] = module
  try:
    spec.loader.exec_module(module)
  finally:
    sys.modules.pop(spec.name, None)
  return module


metrics = _load_metrics()
MetricsRecorder = metrics.MetricsRecorder
extract_batch_counts = metrics.extract_batch_counts


class Mode:
  def __init__(self, name):
    self.name = name


def _batch(mode, extend_lens, decoding_reqs=()):
  reqs = [SimpleNamespace(req_id=index + 1, extend_input_len=length)
          for index, length in enumerate(extend_lens)]
  return SimpleNamespace(
    reqs=reqs,
    forward_mode=Mode(mode),
    decoding_reqs=list(decoding_reqs),
    extend_lens=list(extend_lens),
  )


def _scheduler(budget=None, policy="aggregate"):
  return SimpleNamespace(
    max_num_batched_tokens=budget,
    prefill_budget_policy=policy,
    waiting_queue=[],
    running_batch=SimpleNamespace(reqs=[]),
    chunking_reqs=[],
    num_retracted_reqs=0,
  )


def test_extract_pure_extend_counts_all_extend_tokens():
  counts = extract_batch_counts(_batch("EXTEND", [3, 5]))
  assert counts["prefill_tokens"] == 8
  assert counts["decode_tokens"] == 0
  assert counts["batch_size"] == 2


def test_extract_decode_counts_requests_not_extend_lengths():
  counts = extract_batch_counts(_batch("DECODE", [1, 1, 1]))
  assert counts["prefill_tokens"] == 0
  assert counts["decode_tokens"] == 3


def test_extract_mixed_excludes_decode_extend_len_from_prefill():
  batch = _batch("MIXED", [3, 1])
  batch.decoding_reqs = [batch.reqs[1]]
  counts = extract_batch_counts(batch)
  assert counts["prefill_tokens"] == 3
  assert counts["decode_tokens"] == 1
  assert counts["decode_request_ids"] == [2]


def test_iteration_budget_invariant():
  recorder = MetricsRecorder(enabled=True)
  batch = _batch("MIXED", [3, 1])
  batch.decoding_reqs = [batch.reqs[1]]
  record = recorder.begin_iteration(batch, _scheduler(4), 0.0, 0.1)
  assert record is not None
  assert record.total_compute_tokens == 4
  assert record.invariant_ok is True

  violating = recorder.begin_iteration(batch, _scheduler(3), 0.0, 0.1)
  assert violating is not None
  assert violating.invariant_ok is False


def test_request_timing_and_jsonl_export(tmp_path):
  recorder = MetricsRecorder(enabled=True)
  recorder.record_request_arrival(7, input_tokens=12, timestamp=0.0)
  recorder.observe_outputs(
    [SimpleNamespace(request_id=7, finished=False)], timestamp=0.2
  )
  recorder.observe_outputs(
    [SimpleNamespace(request_id=7, finished=False)], timestamp=0.7
  )
  recorder.observe_outputs(
    [SimpleNamespace(request_id=7, finished=True)], timestamp=1.2
  )

  request = recorder.requests[7]
  assert request.output_tokens == 3
  assert request.ttft_ms == pytest.approx(200.0)
  assert request.tpot_ms == pytest.approx(500.0)
  assert request.itl_ms == pytest.approx([500.0, 500.0])

  batch = _batch("EXTEND", [12])
  iteration = recorder.begin_iteration(batch, _scheduler(), 0.0, 0.1)
  recorder.finish_iteration(iteration, [], _scheduler(), completed_at=1.3)

  path = tmp_path / "metrics.jsonl"
  recorder.export_jsonl(path)
  records = [json.loads(line) for line in path.read_text().splitlines()]
  assert [record["record_type"] for record in records] == [
    "iteration", "request", "summary"
  ]
  assert records[-1]["completed_requests"] == 1


def test_disabled_recorder_is_noop():
  recorder = MetricsRecorder(enabled=False)
  recorder.record_request_arrival(1, 4, timestamp=0.0)
  assert recorder.requests == {}
  assert recorder.summary()["iterations"] == 0
