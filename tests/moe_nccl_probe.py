"""Four-rank NCCL MoE output parity with replicated tokens and sharded experts."""

import datetime
import json
import os

import torch
import torch.distributed as dist

from miniinfer.layers.fused_moe import FusedMoE, MoEParallelConfig
from tests.moe_reference import moe_reference_with_routes


def _make_parameters():
  generator = torch.Generator().manual_seed(20261028)
  router = (torch.randn(128, 64, generator=generator) * 0.1).half()
  gate = (torch.randn(128, 96, 64, generator=generator) * 0.05).half()
  up = (torch.randn(128, 96, 64, generator=generator) * 0.05).half()
  down = (torch.randn(128, 64, 96, generator=generator) * 0.05).half()
  hidden = torch.randn(17, 64, generator=generator).half()
  return router, gate, up, down, hidden


def _run_case(layer, hidden_cpu, router_cpu, gate, up, down, *, concentrated):
  device = torch.device("cuda", torch.cuda.current_device())
  if concentrated:
    hidden_cpu = torch.zeros_like(hidden_cpu)
    hidden_cpu[:, 0] = 1
    router_cpu = torch.zeros_like(router_cpu)
    router_cpu[:8, 0] = torch.tensor([2.0 - 0.1 * i for i in range(8)], dtype=torch.float16)
    with torch.no_grad():
      layer.gate.weight.copy_(router_cpu.to(device))

  hidden = (
    hidden_cpu.to(device) if dist.get_rank() == 0 else torch.empty_like(hidden_cpu, device=device)
  )
  dist.broadcast(hidden, src=0)
  ids, weights, _ = layer._route(hidden)
  all_ids = [torch.empty_like(ids) for _ in range(dist.get_world_size())]
  dist.all_gather(all_ids, ids)
  for replica in all_ids:
    torch.testing.assert_close(replica, ids, rtol=0, atol=0)

  output = layer(hidden)
  expected = moe_reference_with_routes(hidden_cpu, gate, up, down, ids.cpu(), weights.cpu())
  torch.testing.assert_close(output.cpu().float(), expected.float(), rtol=0.03, atol=0.002)

  replicas = [torch.empty_like(output) for _ in range(dist.get_world_size())]
  dist.all_gather(replicas, output)
  for replica in replicas:
    torch.testing.assert_close(replica, output, rtol=0, atol=0)

  error = (output.cpu().float() - expected.float()).abs()
  return {
    "case": "concentrated_rank0" if concentrated else "balanced",
    "routed_experts": int(ids.unique().numel()),
    "max_abs_error": float(error.max()),
    "max_rel_error": float((error / expected.float().abs().clamp_min(1e-5)).max()),
  }


def _run_large_all_reduce_case(rank):
  """Exercise the 1 MiB hidden-state collective size seen in the engine hang."""
  generator = torch.Generator().manual_seed(20261029)
  router = (torch.randn(128, 2048, generator=generator) * 0.02).half()
  gate = (torch.randn(128, 16, 2048, generator=generator) * 0.04).half()
  up = (torch.randn(128, 16, 2048, generator=generator) * 0.04).half()
  down = (torch.randn(128, 2048, 16, generator=generator) * 0.04).half()
  hidden_cpu = torch.randn(256, 2048, generator=generator).half()
  layer = FusedMoE(
    num_experts=128,
    top_k=8,
    hidden_size=2048,
    intermediate_size=16,
    parallel=MoEParallelConfig(ep_size=4, ep_rank=rank, ep_group=dist.group.WORLD),
    dtype=torch.float16,
    device=f"cuda:{rank}",
  )
  with torch.no_grad():
    layer.gate.weight.copy_(router.to(f"cuda:{rank}"))
    for expert in range(layer.experts.local_expert_start, layer.experts.local_expert_end):
      layer.load_expert_weight(expert, "gate_proj", gate[expert])
      layer.load_expert_weight(expert, "up_proj", up[expert])
      layer.load_expert_weight(expert, "down_proj", down[expert])

  device = torch.device("cuda", rank)
  hidden = hidden_cpu.to(device) if rank == 0 else torch.empty_like(hidden_cpu, device=device)
  dist.broadcast(hidden, src=0)
  ids, weights, _ = layer._route(hidden)
  id_replicas = [torch.empty_like(ids) for _ in range(dist.get_world_size())]
  dist.all_gather(id_replicas, ids)
  for replica in id_replicas:
    torch.testing.assert_close(replica, ids, rtol=0, atol=0)
  output = layer(hidden)
  # Independently evaluate a few rows; full-rank output replicas cover the entire 1 MiB result.
  expected = moe_reference_with_routes(
    hidden_cpu[:2], gate, up, down, ids[:2].cpu(), weights[:2].cpu()
  )
  torch.testing.assert_close(output[:2].cpu().float(), expected.float(), rtol=0.03, atol=0.002)
  replicas = [torch.empty_like(output) for _ in range(dist.get_world_size())]
  dist.all_gather(replicas, output)
  for replica in replicas:
    torch.testing.assert_close(replica, output, rtol=0, atol=0)
  error = (output[:2].cpu().float() - expected.float()).abs()
  return {
    "case": "balanced_large_1mib_output",
    "output_numel": output.numel(),
    "output_bytes": output.numel() * output.element_size(),
    "routed_experts": int(ids.unique().numel()),
    "reference_rows": 2,
    "max_abs_error_reference_rows": float(error.max()),
  }


def main():
  rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(rank)
  torch.set_num_threads(1)
  dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
  try:
    router, gate, up, down, hidden = _make_parameters()
    parallel = MoEParallelConfig(ep_size=4, ep_rank=rank, ep_group=dist.group.WORLD)
    layer = FusedMoE(
      num_experts=128,
      top_k=8,
      hidden_size=64,
      intermediate_size=96,
      parallel=parallel,
      dtype=torch.float16,
      device=f"cuda:{rank}",
    )
    with torch.no_grad():
      layer.gate.weight.copy_(router.to(f"cuda:{rank}"))
      for expert in range(layer.experts.local_expert_start, layer.experts.local_expert_end):
        layer.load_expert_weight(expert, "gate_proj", gate[expert])
        layer.load_expert_weight(expert, "up_proj", up[expert])
        layer.load_expert_weight(expert, "down_proj", down[expert])

    reports = []
    reports.append(_run_case(layer, hidden, router, gate, up, down, concentrated=False))
    reports.append(_run_case(layer, hidden, router, gate, up, down, concentrated=True))
    reports.append(_run_large_all_reduce_case(rank))
    torch.cuda.synchronize()
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, reports)
    if rank == 0:
      print(
        json.dumps(
          {
            "status": "PASS",
            "world_size": dist.get_world_size(),
            "backend": dist.get_backend(),
            "experts": 128,
            "experts_per_rank": layer.experts.num_local_experts,
            "checks": [
              "balanced_expert_parallel",
              "rank0_only_routes",
              "zero_remote_contribution",
              "all_reduce_reference_parity",
              "1mib_fp16_all_reduce",
              "replicated_output_consistency",
            ],
            "rank_reports": gathered,
          }
        ),
        flush=True,
      )
  finally:
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
