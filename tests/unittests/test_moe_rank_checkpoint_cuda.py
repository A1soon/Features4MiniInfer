"""Load one EP rank from the real Qwen3-MoE checkpoint on an assigned GPU."""

import gc
import os

import pytest
import torch

from miniinfer.config.engine.config import EngineConfig
from miniinfer.engine.model_runner import ModelRunner
from miniinfer.layers.fused_moe import MoEParallelConfig


pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available(), reason="requires one assigned CUDA GPU"
)


def test_real_checkpoint_streams_one_fp16_expert_rank():
  model_path = os.environ.get("MINIINFER_MOE_MODEL_PATH", "/model")
  config = EngineConfig(
    model_path,
    dtype=torch.float16,
    expert_parallel_size=4,
    enforce_eager=True,
    enable_overlap=False,
  )
  torch.cuda.reset_peak_memory_stats()
  runner = None
  try:
    runner = ModelRunner(
      config,
      device="cuda:0",
      moe_parallel=MoEParallelConfig(ep_size=4, ep_rank=0),
    )
    torch.cuda.synchronize()
    params = list(runner.model.parameters())
    assert params
    assert all(p.device.type == "cuda" and p.dtype == torch.float16 for p in params)
    experts = runner.model.qwen3_moe.layers[0].mlp.experts.experts
    assert experts.num_local_experts == 32
    assert experts.local_expert_start == 0
    assert runner.weight_load_report.loaded_tensors > 0
    assert runner.weight_load_report.skipped_tensors > 0
    peak = torch.cuda.max_memory_allocated()
    print(
      f"rank=0 experts=[0,32) loaded={runner.weight_load_report.loaded_tensors} "
      f"skipped={runner.weight_load_report.skipped_tensors} "
      f"destination_bytes={runner.weight_load_report.destination_bytes} "
      f"peak_allocated_bytes={peak}"
    )
  finally:
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
