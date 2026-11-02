"""Compare prefix-equivalent small/large NCCL reductions without model weights."""

import datetime
import json
import os

import torch
import torch.distributed as dist


def main():
  rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(rank)
  dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
  try:
    generator = torch.Generator().manual_seed(20261029)
    partials = torch.randn(4, 512, 2048, generator=generator).half()
    expected = partials[:, :16].double().sum(0).half().float()
    local = partials[rank].cuda(rank)
    reports = []
    for dtype in (torch.float16, torch.float32):
      small = local[:16].to(dtype).clone()
      dist.all_reduce(small)
      small_cpu = small.half().float().cpu()
      for tokens in (256, 512):
        padded = local[:tokens].to(dtype).clone()
        dist.all_reduce(padded)
        large_cpu = padded[:16].half().float().cpu()
        reports.append({
          "dtype": str(dtype), "tokens": tokens,
          "padding_max_abs_error": (small_cpu - large_cpu).abs().max().item(),
          "small_max_abs_reference_error": (small_cpu - expected).abs().max().item(),
          "large_max_abs_reference_error": (large_cpu - expected).abs().max().item(),
          "mismatched_elements": (small_cpu != large_cpu).sum().item(),
        })
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
      print(json.dumps({"status": "DIAGNOSTIC_COMPLETE", "checks": reports}), flush=True)
  finally:
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
