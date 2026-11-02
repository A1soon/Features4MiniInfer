"""Run the reproducible MiniInfer fixed/aggregate scheduling matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter


@dataclass(frozen=True)
class Workload:
  name: str
  num_reqs: int
  prompt_len: int
  max_tokens: int
  second_wave_num_reqs: int = 0
  second_wave_prompt_len: int = 0
  second_wave_after_steps: int = 0


@dataclass(frozen=True)
class Policy:
  name: str
  value: int

  @property
  def label(self) -> str:
    prefix = "c" if self.name == "fixed" else "b"
    return f"{self.name}-{prefix}{self.value}"


WORKLOADS = (
  Workload("decode-heavy", num_reqs=16, prompt_len=128, max_tokens=128),
  Workload("prefill-heavy", num_reqs=8, prompt_len=2048, max_tokens=16),
  Workload(
    "mixed-arrival",
    num_reqs=8,
    prompt_len=128,
    max_tokens=128,
    second_wave_num_reqs=8,
    second_wave_prompt_len=2048,
    second_wave_after_steps=8,
  ),
)
def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--repetitions", type=int, default=3)
  parser.add_argument("--seed", type=int, default=20261020)
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--fixed-chunks", default="512,1024,2048,4096")
  parser.add_argument("--aggregate-budgets", default="512,1024,2048,4096")
  return parser.parse_args()


def _int_list(value: str) -> tuple[int, ...]:
  if not value.strip():
    return ()
  values = tuple(int(item.strip()) for item in value.split(","))
  if any(item <= 0 for item in values):
    raise ValueError("policy values must be positive")
  if len(set(values)) != len(values):
    raise ValueError("policy values must be unique")
  return values


def _nvidia_smi() -> str:
  return subprocess.run(
    ["nvidia-smi", "--query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate", "--format=csv,noheader"],
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()


def main() -> None:
  args = _parse_args()
  if args.repetitions < 3:
    raise ValueError("formal sweep requires at least three repetitions")
  if args.output_dir.exists():
    raise FileExistsError(args.output_dir)
  policies = tuple(
    [Policy("fixed", value) for value in _int_list(args.fixed_chunks)]
    + [Policy("aggregate", value) for value in _int_list(args.aggregate_budgets)]
  )
  if not policies:
    raise ValueError("at least one fixed chunk or aggregate budget is required")

  repo_root = Path(__file__).resolve().parents[1]
  results_dir = args.output_dir / "results"
  logs_dir = args.output_dir / "logs"
  results_dir.mkdir(parents=True)
  logs_dir.mkdir()

  runs = [
    (repeat, workload, policy)
    for repeat in range(1, args.repetitions + 1)
    for workload in WORKLOADS
    for policy in policies
  ]
  random.Random(args.seed).shuffle(runs)
  manifest = {
    "schema_version": 1,
    "status": "running",
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "model": str(Path(args.model).resolve()),
    "python": sys.executable,
    "seed": args.seed,
    "repetitions": args.repetitions,
    "warmup": args.warmup,
    "workloads": [asdict(workload) for workload in WORKLOADS],
    "policies": [asdict(policy) | {"label": policy.label} for policy in policies],
    "nvidia_smi_start": _nvidia_smi(),
    "planned_runs": len(runs),
    "completed_runs": [],
  }
  manifest_path = args.output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

  env = os.environ.copy()
  env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
  sweep_started = perf_counter()
  try:
    for order, (repeat, workload, policy) in enumerate(runs, start=1):
      run_id = f"{workload.name}__{policy.label}__r{repeat:02d}"
      output_path = results_dir / f"{run_id}.jsonl"
      log_path = logs_dir / f"{run_id}.log"
      command = [
        sys.executable,
        str(repo_root / "benchmark/bench_l3_budget.py"),
        "--model", args.model,
        "--policy", policy.name,
        "--budget", str(policy.value),
        "--chunk-size", str(policy.value if policy.name == "fixed" else 4096),
        "--max-extend-len", "4096",
        "--num-reqs", str(workload.num_reqs),
        "--prompt-len", str(workload.prompt_len),
        "--max-tokens", str(workload.max_tokens),
        "--second-wave-num-reqs", str(workload.second_wave_num_reqs),
        "--second-wave-prompt-len", str(workload.second_wave_prompt_len or workload.prompt_len),
        "--second-wave-after-steps", str(workload.second_wave_after_steps),
        "--max-num-seqs", str(workload.num_reqs + workload.second_wave_num_reqs),
        "--warmup", str(args.warmup),
        "--seed", str(args.seed + repeat),
        "--workload", workload.name,
        "--run-id", run_id,
        "--output", str(output_path),
      ]
      run_started = perf_counter()
      print(f"[{order}/{len(runs)}] {run_id}", flush=True)
      with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
          command,
          cwd=repo_root,
          env=env,
          stdout=log,
          stderr=subprocess.STDOUT,
          text=True,
        )
      run_record = {
        "order": order,
        "run_id": run_id,
        "repeat": repeat,
        "workload": workload.name,
        "policy": policy.name,
        "value": policy.value,
        "elapsed_seconds": perf_counter() - run_started,
        "returncode": completed.returncode,
        "output": str(output_path),
        "log": str(log_path),
      }
      manifest["completed_runs"].append(run_record)
      manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
      if completed.returncode != 0:
        raise RuntimeError(f"benchmark failed: {run_id}; see {log_path}")
  except BaseException:
    manifest["status"] = "failed"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_seconds"] = perf_counter() - sweep_started
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    raise

  manifest.update(
    status="ok",
    finished_at_utc=datetime.now(timezone.utc).isoformat(),
    elapsed_seconds=perf_counter() - sweep_started,
    nvidia_smi_end=_nvidia_smi(),
  )
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  print(json.dumps({"status": "ok", "runs": len(runs), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
  main()
