"""Validate bounded V100 EP prefill/mixed CUDA Graph buckets on Qwen3-30B."""

import json
import os

import torch

from miniinfer.engine.llm_engine import LLMEngine
from miniinfer.utils.sampling_params import SamplingParams


def drain(engine, *, max_steps=24):
  records = []
  for _ in range(max_steps):
    step = engine.step_overlap()
    records.append(step)
    print(
      f"PREFILL_GRAPH step={len(records) - 1} prefill={step.num_prefill_tokens} "
      f"decode={step.num_decode_tokens} outputs={len(step.outputs)}",
      flush=True,
    )
    if not engine.scheduler.has_unfinished():
      return records
  raise TimeoutError("piecewise EP graph request loop exceeded its step bound")


def main():
  os.environ["MINIINFER_PREFILL_CG"] = "1"
  os.environ["MINIINFER_PREFILL_CG_VERIFY"] = "1"
  model_path = os.environ.get("MINIINFER_MOE_MODEL_PATH", "/model")
  config = dict(
    attention_backend="flash_attn_v100",
    dtype=torch.float16,
    expert_parallel_size=4,
    enforce_eager=True,
    enable_ep_cuda_graph=True,
    enable_overlap=False,
    enable_prefix_cache=True,
    enable_chunked_prefill=True,
    chunked_prefill_size=256,
    max_extend_len=512,
    max_num_seqs=2,
    max_context_len=512,
    max_total_tokens=4096,
    page_size=256,
    cuda_graph_max_bs=1,
    gpu_memory_utilization=0.9,
    ep_timeout_seconds=60,
    ep_startup_timeout_seconds=1800,
  )
  report = {"status": "RUNNING", "world_size": 4, "scenarios": []}
  print("PREFILL_GRAPH init_begin", flush=True)
  with LLMEngine(model_path, **config) as engine:
    print("PREFILL_GRAPH init_end", flush=True)
    executor = engine.ep_executor
    assert executor is not None
    prefill_graphs = executor.adapter.runner.piecewise_graph_runner
    assert prefill_graphs is not None and prefill_graphs.is_available()
    report["prefill_graph_buckets"] = sorted(prefill_graphs.graphs)
    del prefill_graphs

    short_id = engine.add_request(
      list(range(1000, 1016)),
      SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
    )
    print("PREFILL_GRAPH initial_step_begin", flush=True)
    initial = engine.step_overlap()
    print("PREFILL_GRAPH initial_step_end", flush=True)
    assert initial.num_prefill_tokens > 0
    assert len(initial.outputs) == 1
    assert not executor._ep_graph_verified
    long_id = engine.add_request(
      list(range(4000, 4300)),
      SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
    )
    mixed_steps = drain(engine)
    mixed_count = sum(
      step.num_prefill_tokens > 0 and step.num_decode_tokens > 0 for step in mixed_steps
    )
    assert mixed_count > 0, "scheduler did not produce a real mixed batch"
    assert executor._ep_graph_verified, "decode graph was not compared with eager"
    assert any(output.request_id == long_id for step in mixed_steps for output in step.outputs)
    assert all(not states for states in executor.inspect_workers())
    report["scenarios"].append(
      {
        "name": "single_prefill_decode_and_chunked_mixed",
        "request_ids": [short_id, long_id],
        "mixed_steps": mixed_count,
        "prefill_tokens": sum(step.num_prefill_tokens for step in mixed_steps),
        "decode_tokens": sum(step.num_decode_tokens for step in mixed_steps),
      }
    )

    prefix = list(range(6000, 6257))
    engine.add_request(prefix, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))
    drain(engine)
    second_id = engine.add_request(
      prefix[:256] + [9000],
      SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
    )
    prefix_steps = drain(engine)
    prefix_hit = max(
      (max(step.prefill_prefix_lens.values(), default=0) for step in prefix_steps),
      default=0,
    )
    assert prefix_hit >= 256
    assert any(output.request_id == second_id for step in prefix_steps for output in step.outputs)
    assert all(not states for states in executor.inspect_workers())
    report["scenarios"].append(
      {"name": "shared_prefix_prefill_graph", "prefix_hit_tokens": prefix_hit}
    )
    report["status"] = "PASS"
  report["cuda_graph"] = "prefill/mixed buckets 256,512; unsupported layouts fallback eager"
  print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
