"""Lightweight observability for local L3 scheduling experiments.

The recorder is deliberately independent from torch and the model runtime so
its accounting can be tested on a CPU-only machine.  It records serial
``LLMEngine.step()`` runs first; overlap-specific timestamps can be added once
the serial baseline is understood.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Optional
import time


def _now() -> float:
  return time.perf_counter()


def _ms(start: Optional[float], end: Optional[float]) -> Optional[float]:
  if start is None or end is None:
    return None
  return max((end - start) * 1000.0, 0.0)


def _p95(values: Iterable[float]) -> Optional[float]:
  ordered = sorted(float(value) for value in values)
  if not ordered:
    return None
  # Nearest-rank p95 keeps the result deterministic for short experiments.
  index = min(len(ordered) - 1, max(0, int((0.95 * len(ordered) + 0.999999) - 1)))
  return ordered[index]


def _mode_name(mode: Any) -> str:
  return str(getattr(mode, "name", mode)).upper()


def extract_batch_counts(batch: Any) -> dict[str, Any]:
  """Extract iteration-level token counts from a ``ScheduledBatch``.

  ``batch.extend_lens`` contains ``1`` for decode requests in a mixed batch,
  so decode requests must be excluded when computing ``prefill_tokens``.
  """
  reqs = list(getattr(batch, "reqs", []) or [])
  mode = _mode_name(getattr(batch, "forward_mode", "IDLE"))
  decoding_reqs = list(getattr(batch, "decoding_reqs", []) or [])
  decoding_ids = {id(req) for req in decoding_reqs}
  req_ids = [int(getattr(req, "req_id", -1)) for req in reqs]
  decode_req_ids = [int(getattr(req, "req_id", -1)) for req in decoding_reqs]

  raw_extend_lens = list(getattr(batch, "extend_lens", []) or [])
  if len(raw_extend_lens) != len(reqs):
    raw_extend_lens = [int(getattr(req, "extend_input_len", 0)) for req in reqs]
  extend_lens = [max(int(length), 0) for length in raw_extend_lens]

  if mode == "DECODE":
    prefill_tokens = 0
    decode_tokens = len(reqs)
  else:
    prefill_tokens = sum(
      length for req, length in zip(reqs, extend_lens) if id(req) not in decoding_ids
    )
    decode_tokens = len(decoding_reqs)

  return {
    "mode": mode,
    "batch_size": len(reqs),
    "prefill_tokens": int(prefill_tokens),
    "decode_tokens": int(decode_tokens),
    "extend_lens": extend_lens,
    "request_ids": req_ids,
    "decode_request_ids": decode_req_ids,
  }


@dataclass
class RequestMetrics:
  request_id: int
  input_tokens: int
  arrival_time: float
  first_token_time: Optional[float] = None
  finish_time: Optional[float] = None
  token_times: list[float] = field(default_factory=list)

  @property
  def output_tokens(self) -> int:
    return len(self.token_times)

  @property
  def ttft_ms(self) -> Optional[float]:
    return _ms(self.arrival_time, self.first_token_time)

  @property
  def tpot_ms(self) -> Optional[float]:
    if self.first_token_time is None or self.finish_time is None or self.output_tokens <= 1:
      return None
    return _ms(self.first_token_time, self.finish_time) / (self.output_tokens - 1)

  @property
  def itl_ms(self) -> list[float]:
    if len(self.token_times) < 2:
      return []
    return [
      (later - earlier) * 1000.0
      for earlier, later in zip(self.token_times, self.token_times[1:])
    ]

  def to_dict(self) -> dict[str, Any]:
    data = asdict(self)
    data.update(
      output_tokens=self.output_tokens,
      ttft_ms=self.ttft_ms,
      tpot_ms=self.tpot_ms,
      itl_ms=self.itl_ms,
    )
    return data


@dataclass
class IterationMetrics:
  iteration_id: int
  mode: str
  batch_size: int
  prefill_tokens: int
  decode_tokens: int
  budget: Optional[int]
  extend_lens: list[int]
  request_ids: list[int]
  decode_request_ids: list[int]
  waiting_queue_len: int
  running_batch_size: int
  chunking_reqs: int
  available_kv_tokens: Optional[int]
  retract_total_before: int
  started_at: float
  scheduled_at: float
  forward_started_at: Optional[float] = None
  forward_finished_at: Optional[float] = None
  completed_at: Optional[float] = None
  retract_total_after: Optional[int] = None

  @property
  def total_compute_tokens(self) -> int:
    return self.prefill_tokens + self.decode_tokens

  @property
  def invariant_ok(self) -> Optional[bool]:
    if self.budget is None:
      return None
    return self.total_compute_tokens <= self.budget

  def to_dict(self) -> dict[str, Any]:
    data = asdict(self)
    data.update(
      total_compute_tokens=self.total_compute_tokens,
      invariant_ok=self.invariant_ok,
      schedule_ms=_ms(self.started_at, self.scheduled_at),
      forward_ms=_ms(self.forward_started_at, self.forward_finished_at),
      iteration_ms=_ms(self.started_at, self.completed_at),
      retract_count=(
        None
        if self.retract_total_after is None
        else max(self.retract_total_after - self.retract_total_before, 0)
      ),
    )
    return data


class MetricsRecorder:
  """Collect request and iteration records for a single experiment run."""

  def __init__(self, enabled: bool = False):
    self.enabled = bool(enabled)
    self.reset()

  def reset(self) -> None:
    self.iterations: list[IterationMetrics] = []
    self.requests: dict[int, RequestMetrics] = {}
    self._next_iteration_id = 0

  def record_request_arrival(
    self, request_id: int, input_tokens: int, timestamp: Optional[float] = None
  ) -> None:
    if not self.enabled:
      return
    self.requests[int(request_id)] = RequestMetrics(
      request_id=int(request_id),
      input_tokens=int(input_tokens),
      arrival_time=_now() if timestamp is None else float(timestamp),
    )

  def begin_iteration(
    self,
    batch: Any,
    scheduler: Any,
    started_at: float,
    scheduled_at: Optional[float] = None,
  ) -> Optional[IterationMetrics]:
    if not self.enabled or batch is None:
      return None
    counts = extract_batch_counts(batch)
    budget = getattr(scheduler, "max_num_batched_tokens", None)
    policy = getattr(scheduler, "prefill_budget_policy", None)
    if policy is not None and policy != "aggregate":
      budget = None
    available_kv = None
    try:
      available_kv = int(scheduler.kv_cache_mgr.available_tokens())
    except (AttributeError, TypeError, RuntimeError):
      pass
    record = IterationMetrics(
      iteration_id=self._next_iteration_id,
      budget=None if budget is None else int(budget),
      waiting_queue_len=len(getattr(scheduler, "waiting_queue", []) or []),
      running_batch_size=len(
        getattr(getattr(scheduler, "running_batch", None), "reqs", []) or []
      ),
      chunking_reqs=len(getattr(scheduler, "chunking_reqs", []) or []),
      available_kv_tokens=available_kv,
      retract_total_before=int(getattr(scheduler, "num_retracted_reqs", 0)),
      started_at=float(started_at),
      scheduled_at=_now() if scheduled_at is None else float(scheduled_at),
      **counts,
    )
    self._next_iteration_id += 1
    return record

  def mark_forward_start(self, record: Optional[IterationMetrics]) -> None:
    if self.enabled and record is not None:
      record.forward_started_at = _now()

  def mark_forward_end(self, record: Optional[IterationMetrics]) -> None:
    if self.enabled and record is not None:
      record.forward_finished_at = _now()

  def finish_iteration(
    self,
    record: Optional[IterationMetrics],
    outputs: Iterable[Any],
    scheduler: Any,
    completed_at: Optional[float] = None,
  ) -> None:
    if not self.enabled or record is None:
      return
    timestamp = _now() if completed_at is None else float(completed_at)
    self.observe_outputs(outputs, timestamp=timestamp)
    record.completed_at = timestamp
    record.retract_total_after = int(getattr(scheduler, "num_retracted_reqs", 0))
    self.iterations.append(record)

  def observe_outputs(self, outputs: Iterable[Any], timestamp: Optional[float] = None) -> None:
    if not self.enabled:
      return
    observed_at = _now() if timestamp is None else float(timestamp)
    for output in outputs or []:
      request_id = int(getattr(output, "request_id"))
      request = self.requests.get(request_id)
      if request is None:
        request = RequestMetrics(request_id, 0, observed_at)
        self.requests[request_id] = request
      request.token_times.append(observed_at)
      if request.first_token_time is None:
        request.first_token_time = observed_at
      if bool(getattr(output, "finished", False)):
        request.finish_time = observed_at

  def summary(self) -> dict[str, Any]:
    completed_requests = [
      request for request in self.requests.values() if request.finish_time is not None
    ]
    ttfts = [request.ttft_ms for request in completed_requests if request.ttft_ms is not None]
    tpots = [request.tpot_ms for request in completed_requests if request.tpot_ms is not None]
    itls = [itl for request in completed_requests for itl in request.itl_ms]
    total_output_tokens = sum(request.output_tokens for request in self.requests.values())
    if self.iterations:
      elapsed_s = max(
        self.iterations[-1].completed_at - self.iterations[0].started_at,
        1e-12,
      )
    else:
      elapsed_s = 0.0
    violations = sum(record.invariant_ok is False for record in self.iterations)
    return {
      "iterations": len(self.iterations),
      "requests": len(self.requests),
      "completed_requests": len(completed_requests),
      "total_prefill_tokens": sum(record.prefill_tokens for record in self.iterations),
      "total_decode_tokens": sum(record.decode_tokens for record in self.iterations),
      "total_output_tokens": total_output_tokens,
      "mode_counts": dict(Counter(record.mode for record in self.iterations)),
      "invariant_violations": violations,
      "retract_total": sum(
        max(record.retract_total_after - record.retract_total_before, 0)
        for record in self.iterations
        if record.retract_total_after is not None
      ),
      "elapsed_ms": elapsed_s * 1000.0,
      "output_throughput_tok_s": total_output_tokens / elapsed_s if elapsed_s else 0.0,
      "ttft_ms_median": median(ttfts) if ttfts else None,
      "ttft_ms_p95": _p95(ttfts),
      "tpot_ms_median": median(tpots) if tpots else None,
      "tpot_ms_p95": _p95(tpots),
      "itl_ms_median": median(itls) if itls else None,
      "itl_ms_p95": _p95(itls),
    }

  def export_jsonl(self, path: str | Path) -> None:
    """Write iteration records, request records, and one summary as JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
      for record in self.iterations:
        handle.write(json.dumps({"record_type": "iteration", **record.to_dict()}) + "\n")
      for request in self.requests.values():
        handle.write(json.dumps({"record_type": "request", **request.to_dict()}) + "\n")
      handle.write(json.dumps({"record_type": "summary", **self.summary()}) + "\n")
