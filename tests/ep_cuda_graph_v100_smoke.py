"""Validate one real EP decode CUDA Graph against eager on the same batch."""

import json
import os

import torch

from miniinfer.engine.llm_engine import LLMEngine
from miniinfer.utils.sampling_params import SamplingParams


def main():
  model_path = os.environ.get("MINIINFER_MOE_MODEL_PATH", "/model")
  config = dict(
    attention_backend="flash_attn_v100",
    dtype=torch.float16,
    expert_parallel_size=4,
    enforce_eager=True,
    enable_ep_cuda_graph=True,
    enable_overlap=False,
    enable_prefix_cache=True,
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
  print("EP_GRAPH smoke_init_begin", flush=True)
  with LLMEngine(model_path, **config) as engine:
    assert engine.ep_executor is not None
    assert engine.max_total_tokens == 4096
    request_id = engine.add_request(
      list(range(1000, 1016)),
      SamplingParams(temperature=0, max_tokens=3, ignore_eos=True),
    )
    steps = []
    for index in range(10):
      print(
        f"EP_GRAPH step_begin={index} unfinished={engine.scheduler.has_unfinished()}",
        flush=True,
      )
      step = engine.step_overlap()
      steps.append(step)
      unfinished = engine.scheduler.has_unfinished()
      print(
        f"EP_GRAPH step={index} prefill={step.num_prefill_tokens} "
        f"decode={step.num_decode_tokens} outputs={len(step.outputs)} "
        f"unfinished={unfinished}",
        flush=True,
      )
      if not unfinished:
        break
    assert sum(len(step.outputs) for step in steps) == 3
    assert any(step.num_prefill_tokens > 0 for step in steps)
    assert engine.ep_executor._ep_graph_verified
    assert all(not states for states in engine.ep_executor.inspect_workers())
    report["scenarios"].append(
      {
        "name": "decode_graph_vs_eager_same_batch",
        "request_id": request_id,
        "steps": len(steps),
        "sampled_tokens": sum(len(step.outputs) for step in steps),
      }
    )
    report["status"] = "PASS"
  report["cuda_graph"] = "EP decode bucket 1; prefill remains eager"
  print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
