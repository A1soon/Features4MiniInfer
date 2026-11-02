"""End-to-end Qwen3-30B EP smoke test for an already assigned four-GPU group."""

import json
import os
import time

import torch

from miniinfer.engine.llm_engine import LLMEngine
from miniinfer.utils.sampling_params import SamplingParams


def drain(engine, *, max_steps=32):
  records = []
  for _ in range(max_steps):
    print(f"SMOKE step_begin index={len(records)}", flush=True)
    step = engine.step_overlap()
    records.append(step)
    print(
      f"SMOKE step_end index={len(records) - 1} prefill={step.num_prefill_tokens} "
      f"decode={step.num_decode_tokens} outputs={len(step.outputs)}",
      flush=True,
    )
    if not engine.scheduler.has_unfinished():
      return records
  raise TimeoutError("EP request loop exceeded the smoke-test step bound")


def main():
  model_path = os.environ.get("MINIINFER_MOE_MODEL_PATH", "/model")
  torch.manual_seed(20261028)
  config = dict(
    attention_backend="flash_attn_v100",
    dtype=torch.float16,
    expert_parallel_size=4,
    enforce_eager=True,
    enable_overlap=False,
    enable_prefix_cache=True,
    enable_chunked_prefill=True,
    chunked_prefill_size=256,
    max_extend_len=512,
    max_num_seqs=2,
    max_context_len=512,
    max_total_tokens=4096,
    page_size=256,
    ep_timeout_seconds=90,
    ep_startup_timeout_seconds=1800,
    gpu_memory_utilization=0.9,
    debug=False,
  )
  report = {"status": "RUNNING", "world_size": 4, "scenarios": []}
  start = time.perf_counter()
  print("SMOKE engine_init_begin", flush=True)
  with LLMEngine(model_path, **config) as engine:
    print("SMOKE engine_init_done", flush=True)
    report["model_load_seconds"] = time.perf_counter() - start
    assert engine.ep_executor.rank_capacities and len(engine.ep_executor.rank_capacities) == 4
    report["rank_kv_capacities"] = engine.ep_executor.rank_capacities
    report["common_kv_capacity"] = engine.max_total_tokens

    # One normal request exercises actual model prefill, decode and rank-0 sampling.
    params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
    request_id = engine.add_request(list(range(1000, 1016)), params)
    steps = drain(engine)
    assert sum(len(s.outputs) for s in steps) == 2
    assert any(s.num_prefill_tokens > 0 for s in steps)
    assert any(s.num_decode_tokens > 0 for s in steps)
    report["scenarios"].append(
      {
        "name": "prefill_decode_sampling",
        "request_id": request_id,
        "steps": len(steps),
        "sampled_tokens": sum(len(s.outputs) for s in steps),
      }
    )
    assert all(not states for states in engine.ep_executor.inspect_workers())

    # Cache one full page, then submit a prompt with that exact shared prefix.
    prefix = list(range(2000, 2257))
    engine.add_request(prefix, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
    drain(engine)
    second = prefix[:256] + [9000]
    engine.add_request(second, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
    prefix_steps = drain(engine)
    prefix_hit = max(
      (max(step.prefill_prefix_lens.values(), default=0) for step in prefix_steps),
      default=0,
    )
    assert prefix_hit >= 256, f"expected at least one cached page, got {prefix_hit}"
    report["scenarios"].append({"name": "shared_prefix_page", "prefix_hit_tokens": prefix_hit})
    assert all(not states for states in engine.ep_executor.inspect_workers())

    # Keep one decode request active while admitting a long, chunked prompt.
    short_id = engine.add_request(
      list(range(3000, 3016)),
      SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
    )
    initial = engine.step_overlap()
    assert any(output.request_id == short_id for output in initial.outputs)
    long_id = engine.add_request(
      list(range(4000, 4300)),
      SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
    )
    mixed_steps = drain(engine)
    saw_mixed = any(s.num_prefill_tokens > 0 and s.num_decode_tokens > 0 for s in mixed_steps)
    assert saw_mixed, "scheduler did not produce an actual mixed batch"
    assert any(output.request_id == long_id for step in mixed_steps for output in step.outputs)
    report["scenarios"].append(
      {
        "name": "chunked_mixed_prefill_decode",
        "steps": len(mixed_steps),
        "mixed_steps": sum(
          s.num_prefill_tokens > 0 and s.num_decode_tokens > 0 for s in mixed_steps
        ),
        "prefill_tokens": sum(s.num_prefill_tokens for s in mixed_steps),
        "decode_tokens": sum(s.num_decode_tokens for s in mixed_steps),
      }
    )
    assert all(not states for states in engine.ep_executor.inspect_workers())
    report["status"] = "PASS"

  report["total_seconds"] = time.perf_counter() - start
  report["cuda_graph"] = "disabled; EP currently requires eager"
  print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
