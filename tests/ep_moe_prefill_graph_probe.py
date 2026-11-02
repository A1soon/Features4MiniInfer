"""Isolate 256x2048 Triton, NCCL, and combined EP graph paths."""

import datetime
import json
import os

import torch
import torch.distributed as dist

from miniinfer.layers.fused_moe import FusedMoE, MoEParallelConfig
from miniinfer.engine.cuda_graph_runner import retain_moe_workspaces
from tests.moe_reference import moe_reference_with_routes


def mark(rank, phase):
  print(f"MOE_GRAPH_PROBE rank={rank} phase={phase}", flush=True)


def main():
  rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(rank)
  torch.set_num_threads(1)
  dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
  try:
    generator = torch.Generator().manual_seed(20261029)
    router = torch.randn(128, 2048, generator=generator).mul_(0.02).half()
    gate = torch.randn(128, 16, 2048, generator=generator).mul_(0.04).half()
    up = torch.randn(128, 16, 2048, generator=generator).mul_(0.04).half()
    down = torch.randn(128, 2048, 16, generator=generator).mul_(0.04).half()
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

    hidden = hidden_cpu.to(f"cuda:{rank}") if rank == 0 else torch.empty_like(
      hidden_cpu, device=f"cuda:{rank}"
    )
    dist.broadcast(hidden, src=0)
    ids, weights, _ = layer._route(hidden)
    torch.cuda.synchronize()
    mark(rank, "initialized")

    with torch.inference_mode():
      eager_local = layer.experts(hidden, ids, weights).clone()
      torch.cuda.synchronize()
      mark(rank, "eager_local")

      static_hidden = hidden.clone()
      static_ids = ids.clone()
      static_weights = weights.clone()
      local_graph_output = torch.empty_like(eager_local)
      layer.experts(static_hidden, static_ids, static_weights)
      torch.cuda.synchronize()
      local_graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(local_graph):
        local_graph_output.copy_(layer.experts(static_hidden, static_ids, static_weights))
      torch.cuda.synchronize()
      mark(rank, "local_capture")
      dist.barrier()
      local_graph.replay()
      torch.cuda.synchronize()
      torch.testing.assert_close(local_graph_output, eager_local, rtol=1e-3, atol=1e-3)
      mark(rank, "local_replay_pass")
      dist.barrier()

      eager_full = eager_local.clone()
      dist.all_reduce(eager_full, op=dist.ReduceOp.SUM)
      torch.cuda.synchronize()
      mark(rank, "eager_all_reduce")

      collective_input = eager_local.clone()
      for _ in range(2):
        warmup = eager_local.clone()
        dist.all_reduce(warmup, op=dist.ReduceOp.SUM)
      torch.cuda.synchronize()
      collective_graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(collective_graph):
        dist.all_reduce(collective_input, op=dist.ReduceOp.SUM)
      torch.cuda.synchronize()
      mark(rank, "collective_capture")
      collective_input.copy_(eager_local)
      dist.barrier()
      collective_graph.replay()
      torch.cuda.synchronize()
      torch.testing.assert_close(collective_input, eager_full, rtol=0, atol=0)
      mark(rank, "collective_replay_pass")
      dist.barrier()

      for _ in range(2):
        layer(hidden)
      torch.cuda.synchronize()
      joint_output = torch.empty_like(eager_full)
      joint_graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(joint_graph):
        joint_output.copy_(layer(static_hidden))
      torch.cuda.synchronize()
      mark(rank, "joint_capture")
      dist.barrier()
      joint_graph.replay()
      torch.cuda.synchronize()
      torch.testing.assert_close(joint_output, eager_full, rtol=1e-3, atol=1e-3)
      mark(rank, "joint_replay_pass")
      dist.barrier()

      bucket_graphs = []
      for tokens in (1, 256, 512):
        x = hidden.repeat(2, 1)[:tokens].clone()
        layer(x)
        torch.cuda.synchronize()
        workspaces = retain_moe_workspaces(layer)
        output = torch.empty_like(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
          output.copy_(layer(x))
        bucket_graphs.append((tokens, graph, x, output, workspaces))
      # A larger eager fallback replaces the module's current workspace too.
      layer(hidden.repeat(4, 1))
      torch.cuda.synchronize()
      dist.barrier()
      for tokens, graph, x, output, workspaces in bucket_graphs:
        graph.replay()
        torch.cuda.synchronize()
        eager = layer(x).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, eager, rtol=1e-3, atol=1e-3)
        mark(rank, f"retained_workspace_bucket_{tokens}_pass")
        dist.barrier()

    expected = moe_reference_with_routes(
      hidden_cpu[:2], gate, up, down, ids[:2].cpu(), weights[:2].cpu()
    )
    torch.testing.assert_close(
      joint_output[:2].cpu().float(), expected.float(), rtol=0.03, atol=0.002
    )
    replicas = [torch.empty_like(joint_output) for _ in range(dist.get_world_size())]
    dist.all_gather(replicas, joint_output)
    for replica in replicas:
      torch.testing.assert_close(replica, joint_output, rtol=0, atol=0)
    if rank == 0:
      print(
        json.dumps(
          {
            "status": "PASS",
            "world_size": dist.get_world_size(),
            "tokens": hidden.shape[0],
            "hidden_size": hidden.shape[1],
            "collective_bytes": eager_full.numel() * eager_full.element_size(),
            "max_abs_error_reference_rows": float(
              (joint_output[:2].cpu().float() - expected.float()).abs().max()
            ),
            "checks": ["local_expert_graph", "1mib_all_reduce_graph", "joint_moe_graph",
                       "multi_bucket_workspace_lifetime"],
          }
        ),
        flush=True,
      )
    local_graph = None
    collective_graph = None
    joint_graph = None
    graph = None
    bucket_graphs.clear()
    del output, x, workspaces
    del local_graph_output, joint_output, collective_input
    del static_hidden, static_ids, static_weights
    torch.cuda.synchronize()
    dist.barrier()
    mark(rank, "cleanup_ready")
  finally:
    if dist.is_initialized():
      dist.destroy_process_group()


if __name__ == "__main__":
  main()
