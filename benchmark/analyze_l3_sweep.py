"""Validate and summarize a completed MiniInfer L3 scheduling sweep."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median
from typing import Iterable


def _p95(values: Iterable[float]) -> float | None:
  ordered = sorted(float(value) for value in values)
  if not ordered:
    return None
  return ordered[min(len(ordered) - 1, max(0, (95 * len(ordered) + 99) // 100 - 1))]


def _records(path: Path) -> list[dict]:
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _setting(benchmark: dict) -> str:
  if benchmark["policy"] == "aggregate":
    return f"B={benchmark['budget']}"
  return f"chunk={benchmark['chunk_size']}"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("sweep_dir", type=Path)
  args = parser.parse_args()

  manifest = json.loads((args.sweep_dir / "manifest.json").read_text())
  if manifest["status"] != "ok":
    raise RuntimeError("sweep did not complete successfully")
  grouped_runs: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
  grouped_requests: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
  mixed_intervals: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
    lambda: {"before": [], "during": []}
  )

  for path in sorted((args.sweep_dir / "results").glob("*.jsonl")):
    records = _records(path)
    benchmark, = [record for record in records if record["record_type"] == "benchmark"]
    iterations = [record for record in records if record["record_type"] == "iteration"]
    requests = [record for record in records if record["record_type"] == "request"]
    expected_prefill = (
      benchmark["num_reqs"] * benchmark["prompt_len"]
      + benchmark["second_wave_num_reqs"] * benchmark["second_wave_prompt_len"]
    )
    expected_output = benchmark["total_num_reqs"] * benchmark["max_tokens"]
    expected_decode = benchmark["total_num_reqs"] * (benchmark["max_tokens"] - 1)
    assert benchmark["completed_requests"] == benchmark["total_num_reqs"], path
    assert benchmark["total_prefill_tokens"] == expected_prefill, path
    assert benchmark["total_output_tokens"] == expected_output, path
    assert benchmark["total_decode_tokens"] == expected_decode, path
    assert benchmark["invariant_violations"] == 0, path
    assert benchmark["all_pages_reclaimed"], path
    assert benchmark["all_request_rows_reclaimed"], path
    assert all(len(row["request_ids"]) == len(set(row["request_ids"])) for row in iterations), path
    if benchmark["policy"] == "aggregate":
      assert all(row["invariant_ok"] for row in iterations), path
    if benchmark["workload"] == "mixed-arrival":
      assert benchmark["mode_counts"].get("MIXED", 0) > 0, path

    key = (benchmark["workload"], benchmark["policy"], _setting(benchmark))
    grouped_runs[key].append(benchmark)
    for request in requests:
      grouped_requests[key + (request["input_tokens"],)].append(request)
    if benchmark["workload"] == "mixed-arrival":
      short_input_tokens = min(request["input_tokens"] for request in requests)
      long_input_tokens = max(request["input_tokens"] for request in requests)
      short_requests = [
        request for request in requests if request["input_tokens"] == short_input_tokens
      ]
      long_requests = [
        request for request in requests if request["input_tokens"] == long_input_tokens
      ]
      interference_started_at = min(request["arrival_time"] for request in long_requests)
      interference_finished_at = max(request["first_token_time"] for request in long_requests)
      for request in short_requests:
        for earlier, later in zip(request["token_times"], request["token_times"][1:]):
          interval_ms = (later - earlier) * 1000.0
          if later <= interference_started_at:
            mixed_intervals[key]["before"].append(interval_ms)
          elif earlier < interference_finished_at and later > interference_started_at:
            mixed_intervals[key]["during"].append(interval_ms)

  expected_runs = manifest["planned_runs"]
  actual_runs = sum(len(rows) for rows in grouped_runs.values())
  assert actual_runs == expected_runs, (actual_runs, expected_runs)

  summary_rows = []
  for (workload, policy, setting), rows in sorted(grouped_runs.items()):
    throughputs = [row["output_throughput_tok_s"] for row in rows]
    walls = [row["wall_ms"] for row in rows]
    ttfts = [row["ttft_ms_median"] for row in rows]
    tpots = [row["tpot_ms_median"] for row in rows]
    itls = [row["itl_ms_median"] for row in rows]
    summary_rows.append({
      "workload": workload,
      "policy": policy,
      "setting": setting,
      "repetitions": len(rows),
      "throughput_tok_s_median": median(throughputs),
      "throughput_tok_s_p95": _p95(throughputs),
      "wall_ms_median": median(walls),
      "wall_ms_p95": _p95(walls),
      "ttft_ms_median_of_runs": median(ttfts),
      "ttft_ms_p95_of_runs": _p95(ttfts),
      "tpot_ms_median_of_runs": median(tpots),
      "tpot_ms_p95_of_runs": _p95(tpots),
      "itl_ms_median_of_runs": median(itls),
      "itl_ms_p95_of_runs": _p95(itls),
      "mixed_iterations_median": median(row["mode_counts"].get("MIXED", 0) for row in rows),
      "retract_total": sum(row["retract_total"] for row in rows),
      "invariant_violations": sum(row["invariant_violations"] for row in rows),
    })

  request_rows = []
  for (workload, policy, setting, input_tokens), rows in sorted(grouped_requests.items()):
    ttfts = [row["ttft_ms"] for row in rows if row["ttft_ms"] is not None]
    tpots = [row["tpot_ms"] for row in rows if row["tpot_ms"] is not None]
    itls = [value for row in rows for value in row["itl_ms"]]
    request_rows.append({
      "workload": workload,
      "policy": policy,
      "setting": setting,
      "input_tokens": input_tokens,
      "requests": len(rows),
      "ttft_ms_median": median(ttfts),
      "ttft_ms_p95": _p95(ttfts),
      "tpot_ms_median": median(tpots),
      "tpot_ms_p95": _p95(tpots),
      "itl_ms_median": median(itls),
      "itl_ms_p95": _p95(itls),
    })

  interference_rows = []
  for (workload, policy, setting), intervals in sorted(mixed_intervals.items()):
    before_p95 = _p95(intervals["before"])
    during_p95 = _p95(intervals["during"])
    interference_rows.append({
      "workload": workload,
      "policy": policy,
      "setting": setting,
      "before_intervals": len(intervals["before"]),
      "during_intervals": len(intervals["during"]),
      "before_itl_ms_median": median(intervals["before"]),
      "before_itl_ms_p95": before_p95,
      "during_itl_ms_median": median(intervals["during"]),
      "during_itl_ms_p95": during_p95,
      "p95_increase_ms": during_p95 - before_p95,
      "p95_increase_ratio": during_p95 / before_p95,
    })

  for name, rows in (
    ("summary.csv", summary_rows),
    ("request-classes.csv", request_rows),
    ("mixed-interference.csv", interference_rows),
  ):
    with (args.sweep_dir / name).open("w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
  result = {
    "status": "ok",
    "validated_runs": actual_runs,
    "groups": len(summary_rows),
    "summary": summary_rows,
    "request_classes": request_rows,
    "mixed_interference": interference_rows,
  }
  (args.sweep_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"status": "ok", "validated_runs": actual_runs, "groups": len(summary_rows)}))


if __name__ == "__main__":
  main()
