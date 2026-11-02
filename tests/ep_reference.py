"""CPU-only rank adapter: production model/loader/sampler, SDPA and tiny KV storage."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from miniinfer.engine.expert_parallel import CudaEPRank


class CpuKV:
  def __init__(self, config, size):
    c = config.hf_config
    self.table = torch.zeros(config.max_num_seqs, config.max_context_len, dtype=torch.int32)
    self.request_pool = SimpleNamespace(
      req_to_token_pool=lambda: self.table, free=lambda rows: None
    )
    shape = (size + config.page_size, c.num_key_value_heads, c.head_dim)
    self.buffers = [(torch.zeros(shape), torch.zeros(shape)) for _ in range(c.num_hidden_layers)]

  def update_unfinished_req_radix_cache(self, req):
    pass


class SDPABackend:
  def __init__(self, kv):
    self.kv = kv

  def init_forward_metadata(self, batch):
    pass

  def forward(self, q, k, v, layer, batch, save_kv_cache=True):
    keys, values = self.kv.buffers[layer.layer_id]
    keys[batch.out_cache_loc] = k
    values[batch.out_cache_loc] = v
    q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
    output = []
    offset = 0
    qs = batch.extend_seq_lens_cpu or [1] * batch.batch_size
    for row, n, count in zip(batch.req_pool_indices.tolist(), batch.seq_lens_cpu.tolist(), qs):
      slots = self.kv.table[row, :n].long()
      repeat = layer.tp_q_head_num // layer.tp_k_head_num
      query = q[offset : offset + count].transpose(0, 1).unsqueeze(0)
      key = keys[slots].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
      value = values[slots].repeat_interleave(repeat, dim=1).transpose(0, 1).unsqueeze(0)
      mask = torch.arange(n).unsqueeze(0) <= torch.arange(n - count, n).unsqueeze(1)
      result = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=layer.scaling
      )
      output.append(result.squeeze(0).transpose(0, 1).reshape(count, -1))
      offset += count
    return torch.cat(output)


class CpuEPRank(CudaEPRank):
  def __init__(self, config, rank):
    from miniinfer.engine.model_runner import ModelRunner
    from miniinfer.layers.fused_moe import MoEParallelConfig

    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    torch.set_num_threads(1)
    if rank == getattr(config, "fail_init_rank", -1):
      raise RuntimeError("injected initialization failure")
    self.config = config
    self.rank = rank
    self.device = "cpu"
    self.capacity = (64, 60, 56, 52)[rank]
    self.runner = ModelRunner(
      config,
      device="cpu",
      rank=rank,
      moe_parallel=MoEParallelConfig(
        ep_size=config.expert_parallel_size, ep_rank=rank, ep_group=dist.group.WORLD
      ),
    )
    self.sample_count = 0

  def init_kv(self, capacity):
    self.kv = CpuKV(self.config, capacity)
    self.runner.kv_cache_mgr = self.kv
    self.runner.attn_backend = SDPABackend(self.kv)

  def init_cuda_graph(self, max_batch_size, max_context_len):
    dist.barrier()
    self.cuda_graph_request = (max_batch_size, max_context_len)
    dist.barrier()

  def release_cuda_graphs(self):
    self.cuda_graph_released = True

  def prepare(self, packet):
    if self.rank == getattr(self.config, "fail_prepare_rank", -1):
      raise RuntimeError("injected preparation failure")
    if self.rank == getattr(self.config, "exit_prepare_rank", -1):
      os._exit(19)
    if self.rank == getattr(self.config, "stall_prepare_rank", -1):
      import time

      time.sleep(30)
    return super().prepare(packet)

  def sample(self, logits, batch, **kwargs):
    if self.rank != 0:
      raise AssertionError("follower attempted to sample")
    self.sample_count += 1
    params = [r.sampling_params for r in batch.all_seqs] if batch.all_seqs else None
    temperatures = [p.temperature for p in params] if params else [0.7] * batch.batch_size
    batch.sampling_temperatures = torch.tensor(temperatures)
    batch.sampling_top_ps = (
      torch.tensor([p.top_p for p in params]) if params else torch.ones(batch.batch_size)
    )
    batch.sampling_top_ks = (
      torch.tensor([p.top_k for p in params])
      if params
      else torch.zeros(batch.batch_size, dtype=torch.int64)
    )
    batch.sampling_enable_top_k_top_p = bool(params) and any(
      p.top_k > 0 or p.top_p < 1 for p in params
    )
    batch.sampling_all_non_greedy = all(t > 0 for t in temperatures)
    batch.sampling_all_greedy = all(t <= 0 for t in temperatures)
    return self.runner.sample(logits, batch, **kwargs)

  def synchronize(self):
    pass
