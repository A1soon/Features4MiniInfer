"""L3 serial benchmark for fixed versus aggregate prefill budgets.

The runner deliberately uses ``LLMEngine.step()`` rather than ``generate()``:
the latter uses the overlap path by default, while this first L3 experiment
needs a simple, request-visible timeline for TTFT/TPOT/ITL.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", required=True, help="Hugging Face model path or id")
  parser.add_argument("--policy", choices=("fixed", "aggregate"), required=True)
  parser.add_argument("--budget", type=int, default=2048,
                      help="aggregate B; ignored by fixed policy")
  parser.add_argument("--chunk-size", type=int, default=4096,
                      help="per-iteration chunk cap used by both policies")
  parser.add_argument("--max-extend-len", type=int, default=4096)
  parser.add_argument("--num-reqs", type=int, default=8)
  parser.add_argument("--prompt-len", type=int, default=512)
  parser.add_argument("--max-tokens", type=int, default=32)
  parser.add_argument("--second-wave-num-reqs", type=int, default=0)
  parser.add_argument("--second-wave-prompt-len", type=int, default=None)
  parser.add_argument("--second-wave-after-steps", type=int, default=8)
  parser.add_argument("--warmup", type=int, default=1,
                      help="number of short warmup requests")
  parser.add_argument("--max-num-seqs", type=int, default=None)
  parser.add_argument("--max-context-len", type=int, default=4096)
  parser.add_argument("--page-size", type=int, default=256)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--workload", default="custom")
  parser.add_argument("--run-id", default=None)
  parser.add_argument("--output", type=Path, required=True,
                      help="JSONL output path")
  return parser.parse_args()


def _run_steps(engine) -> int:
  steps = 0
  while not engine.is_finished():
    engine.step()
    steps += 1
  return steps


def main() -> None:
  args = _parse_args()

  # Keep --help usable on a CPU-only checkout: MiniInfer imports Triton and
  # CUDA backends at module import time, so defer the runtime import.
  import torch

  repo_root = Path(__file__).resolve().parents[1]
  if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

  from miniinfer.engine.llm_engine import LLMEngine
  from miniinfer.utils.sampling_params import SamplingParams

  if args.output.exists():
    raise FileExistsError(args.output)
  if args.num_reqs <= 0 or args.prompt_len <= 0 or args.max_tokens <= 0:
    raise ValueError("num-reqs, prompt-len, and max-tokens must be positive")
  if args.second_wave_num_reqs < 0 or args.second_wave_after_steps < 0:
    raise ValueError("second-wave-num-reqs and second-wave-after-steps must be non-negative")
  second_wave_prompt_len = args.second_wave_prompt_len or args.prompt_len
  if args.second_wave_num_reqs and second_wave_prompt_len <= 0:
    raise ValueError("second-wave-prompt-len must be positive")
  if args.chunk_size <= 0 or args.max_extend_len <= 0:
    raise ValueError("chunk-size and max-extend-len must be positive")
  if args.chunk_size % args.page_size != 0:
    raise ValueError("chunk-size must be page-aligned")
  if args.policy == "aggregate" and args.budget <= 0:
    raise ValueError("budget must be positive for aggregate policy")

  total_num_reqs = args.num_reqs + args.second_wave_num_reqs
  max_num_seqs = args.max_num_seqs or max(total_num_reqs, 8)
  if max_num_seqs < total_num_reqs:
    raise ValueError("max-num-seqs must fit both request waves")
  if args.policy == "aggregate" and args.budget < max_num_seqs:
    raise ValueError(
      "aggregate budget must be at least max-num-seqs so a decode-only batch "
      "fits the P_t + D_t <= B invariant"
    )

  config = {
    "max_num_seqs": max_num_seqs,
    "max_context_len": args.max_context_len,
    "max_extend_len": args.max_extend_len,
    "chunked_prefill_size": args.chunk_size,
    "enable_chunked_prefill": True,
    "prefill_budget_policy": args.policy,
    "enable_prefix_cache": False,
    "enforce_eager": True,
    "enable_overlap": False,
    "attention_backend": "flashinfer",
    "page_size": args.page_size,
    "enable_metrics": True,
  }
  if args.policy == "aggregate":
    config["max_num_batched_tokens"] = args.budget

  torch.manual_seed(args.seed)
  engine = LLMEngine(model=args.model, **config)
  try:
    # Repeated token ids make prompts deterministic and avoid tokenizer timing
    # being part of the measured serving loop.  Token id 1 is valid for the
    # supported causal LMs used in this experiment.
    warmup_prompt = [[1] * min(args.prompt_len, 32)]
    warmup_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=2)
    for _ in range(max(args.warmup, 0)):
      engine.add_requests(warmup_prompt, [warmup_params])
      _run_steps(engine)
    engine.scheduler.drain_pending_releases()
    if torch.cuda.is_available():
      torch.cuda.synchronize()

    engine.reset_metrics()
    prompts = [[1] * args.prompt_len for _ in range(args.num_reqs)]
    params = [
      SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.max_tokens)
      for _ in range(args.num_reqs)
    ]
    engine.add_requests(prompts, params)

    if torch.cuda.is_available():
      torch.cuda.synchronize()
    measurement_started_at_utc = datetime.now(timezone.utc).isoformat()
    started_at = perf_counter()
    steps_before_second_wave = 0
    if args.second_wave_num_reqs:
      while steps_before_second_wave < args.second_wave_after_steps and not engine.is_finished():
        engine.step()
        steps_before_second_wave += 1
      second_wave_prompts = [
        [2] * second_wave_prompt_len for _ in range(args.second_wave_num_reqs)
      ]
      second_wave_params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.max_tokens)
        for _ in second_wave_prompts
      ]
      engine.add_requests(second_wave_prompts, second_wave_params)
    remaining_steps = _run_steps(engine)
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    wall_ms = (perf_counter() - started_at) * 1000.0
    measurement_finished_at_utc = datetime.now(timezone.utc).isoformat()

    summary = engine.get_metrics_summary()
    summary.update({
      "workload": args.workload,
      "run_id": args.run_id,
      "seed": args.seed,
      "policy": args.policy,
      "budget": args.budget if args.policy == "aggregate" else None,
      "chunk_size": args.chunk_size,
      "max_extend_len": args.max_extend_len,
      "num_reqs": args.num_reqs,
      "second_wave_num_reqs": args.second_wave_num_reqs,
      "second_wave_prompt_len": second_wave_prompt_len,
      "second_wave_after_steps": args.second_wave_after_steps,
      "steps_before_second_wave": steps_before_second_wave,
      "remaining_steps": remaining_steps,
      "total_num_reqs": total_num_reqs,
      "prompt_len": args.prompt_len,
      "max_tokens": args.max_tokens,
      "wall_ms": wall_ms,
      "measurement_started_at_utc": measurement_started_at_utc,
      "measurement_finished_at_utc": measurement_finished_at_utc,
    })

    engine.scheduler.drain_pending_releases()
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    all_pages_reclaimed = (
      engine.kv_cache_mgr.token_allocator.available_size() == engine.kv_cache_mgr.size
    )
    all_request_rows_reclaimed = (
      len(engine.kv_cache_mgr.request_pool.free_slots) == max_num_seqs
    )
    summary.update(
      all_pages_reclaimed=all_pages_reclaimed,
      all_request_rows_reclaimed=all_request_rows_reclaimed,
    )
    expected_prefill_tokens = (
      args.num_reqs * args.prompt_len
      + args.second_wave_num_reqs * second_wave_prompt_len
    )
    expected_output_tokens = total_num_reqs * args.max_tokens
    expected_decode_tokens = total_num_reqs * (args.max_tokens - 1)
    if summary["completed_requests"] != total_num_reqs:
      raise RuntimeError(f"completed request mismatch: {summary}")
    if summary["total_prefill_tokens"] != expected_prefill_tokens:
      raise RuntimeError(f"prefill token mismatch: {summary}")
    if summary["total_output_tokens"] != expected_output_tokens:
      raise RuntimeError(f"output token mismatch: {summary}")
    if summary["total_decode_tokens"] != expected_decode_tokens:
      raise RuntimeError(f"decode token mismatch: {summary}")
    if summary["invariant_violations"] != 0:
      raise RuntimeError(f"budget invariant violation: {summary}")
    if not all_pages_reclaimed or not all_request_rows_reclaimed:
      raise RuntimeError(f"resource leak after benchmark: {summary}")
    engine.export_metrics(args.output)

    # Append a configuration-rich summary as the final JSONL record.  Keeping
    # the recorder's native summary intact makes the file useful to scripts
    # that expect its stable schema while this record captures CLI settings.
    with args.output.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps({"record_type": "benchmark", **summary}) + "\n")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
  finally:
    engine.stop()


if __name__ == "__main__":
  main()
