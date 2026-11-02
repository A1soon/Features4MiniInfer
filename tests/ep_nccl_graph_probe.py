"""Isolate four-rank NCCL CUDA Graph capture and replay at a 1 MiB payload."""

import datetime
import json
import os

import torch
import torch.distributed as dist


def main():
  rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(rank)
  torch.set_num_threads(1)
  dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
  try:
    size = 524288
    expected = float(dist.get_world_size() * (dist.get_world_size() + 1) // 2)
    for _ in range(2):
      warmup = torch.full((size,), rank + 1, dtype=torch.float16, device=f"cuda:{rank}")
      dist.all_reduce(warmup)
      torch.testing.assert_close(warmup, torch.full_like(warmup, expected), rtol=0, atol=0)
    torch.cuda.synchronize()
    dist.barrier()

    captured = torch.full((size,), rank + 1, dtype=torch.float16, device=f"cuda:{rank}")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
      dist.all_reduce(captured)
    dist.barrier()

    captured.fill_(rank + 1)
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    torch.testing.assert_close(captured, torch.full_like(captured, expected), rtol=0, atol=0)

    replicas = [torch.empty_like(captured) for _ in range(dist.get_world_size())]
    dist.all_gather(replicas, captured)
    for replica in replicas:
      torch.testing.assert_close(replica, captured, rtol=0, atol=0)
    if rank == 0:
      print(
        json.dumps(
          {
            "status": "PASS",
            "world_size": dist.get_world_size(),
            "backend": dist.get_backend(),
            "nccl_version": torch.cuda.nccl.version(),
            "payload_numel": size,
            "payload_bytes": captured.numel() * captured.element_size(),
            "checks": ["eager_warmup", "nccl_graph_capture", "graph_replay", "replica_parity"],
          }
        ),
        flush=True,
      )
    del graph
    del captured
    torch.cuda.synchronize()
    dist.barrier()
  finally:
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
