"""Validate graph replay parity for a four-rank local-expert MoE forward."""

import datetime
import json
import os

import torch
import torch.distributed as dist

from miniinfer.layers.fused_moe import FusedMoE, MoEParallelConfig
from tests.moe_nccl_probe import _make_parameters
from tests.moe_reference import moe_reference_with_routes


def main():
  rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(rank)
  torch.set_num_threads(1)
  dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
  graph = None
  try:
    router, gate, up, down, hidden_cpu = _make_parameters()
    hidden_cpu = hidden_cpu[:1].contiguous()
    layer = FusedMoE(
      num_experts=128,
      top_k=8,
      hidden_size=64,
      intermediate_size=96,
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

    hidden = hidden_cpu.to(f"cuda:{rank}")
    dist.broadcast(hidden, src=0)
    ids, weights, _ = layer._route(hidden)
    with torch.inference_mode():
      eager = layer(hidden).clone()
      for _ in range(2):
        layer(hidden)
    torch.cuda.synchronize()
    dist.barrier()

    static_hidden = hidden.clone()
    graph_output = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
      graph_output.copy_(layer(static_hidden))
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()

    expected = moe_reference_with_routes(
      hidden_cpu, gate, up, down, ids.cpu(), weights.cpu()
    )
    torch.testing.assert_close(graph_output, eager, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(graph_output.cpu().float(), expected.float(), rtol=0.03, atol=0.002)
    replicas = [torch.empty_like(graph_output) for _ in range(dist.get_world_size())]
    dist.all_gather(replicas, graph_output)
    for replica in replicas:
      torch.testing.assert_close(replica, graph_output, rtol=0, atol=0)
    if rank == 0:
      print(
        json.dumps(
          {
            "status": "PASS",
            "world_size": dist.get_world_size(),
            "batch_size": hidden.shape[0],
            "hidden_size": hidden.shape[1],
            "routed_experts": int(ids.unique().numel()),
            "max_abs_error_graph_eager": float(
              (graph_output.float() - eager.float()).abs().max()
            ),
            "max_abs_error_graph_reference": float(
              (graph_output.cpu().float() - expected.float()).abs().max()
            ),
          }
        ),
        flush=True,
      )
    graph = None
    del graph_output, static_hidden
    torch.cuda.synchronize()
    dist.barrier()
  finally:
    graph = None
    if dist.is_initialized():
      dist.destroy_process_group()


if __name__ == "__main__":
  main()
