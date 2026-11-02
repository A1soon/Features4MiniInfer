"""Leader-scheduled, eager expert parallel execution with owned spawned workers.

Control messages contain CPU data only. Model collectives use the owned WORLD;
followers never run a scheduler, allocate token slots, or sample independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import tempfile
import time
import traceback

import torch
import torch.distributed as dist


def _trace_ep_graph(rank, phase):
  if os.getenv("MINIINFER_PREFILL_CG_TRACE") == "1":
    print(f"EP_GRAPH rank={rank} phase={phase}", flush=True)


def common_kv_capacity(capacities, page_size, requested=None):
  if not capacities or page_size <= 0 or any(c < 0 for c in capacities):
    raise ValueError("invalid KV capacities/page size")
  capacity = min(capacities)
  if requested is not None:
    capacity = min(capacity, requested)
  capacity = capacity // page_size * page_size
  if capacity < 2 * page_size:
    raise ValueError(f"insufficient KV capacity across ranks: {capacities}")
  return capacity


def kv_capacity_from_bytes(available, bytes_per_token, page_size, reserved_ratio=0.20):
  if bytes_per_token <= 0 or page_size <= 0 or not 0 <= reserved_ratio < 1:
    raise ValueError("invalid KV memory budget")
  # Storage includes one reserved page in addition to the allocatable capacity.
  pages = int(max(0, available) * (1 - reserved_ratio)) // (bytes_per_token * page_size)
  return max(0, pages - 1) * page_size


@dataclass(frozen=True)
class RequestState:
  req_id: int
  row: int
  history: tuple[int, ...]
  chunked: bool = False
  finished: bool = False
  retracted: bool = False

  @classmethod
  def from_req(cls, req):
    return cls(
      int(req.req_id),
      int(req.req_pool_idx),
      tuple(req.output_ids),
      bool(req.is_chunked),
      bool(req.finished),
      bool(req.is_retracted),
    )


@dataclass(frozen=True)
class BatchPacket:
  mode: int
  requests: tuple[RequestState, ...]
  input_ids: tuple[int, ...]
  positions: tuple[int, ...]
  seq_lens: tuple[int, ...]
  q_lens: tuple[int, ...]
  out_slots: tuple[int, ...]
  slot_rows: tuple[tuple[int, ...], ...]

  @classmethod
  def from_forward(cls, batch, kv):
    def ints(tensor):
      return tuple(int(x) for x in tensor.detach().cpu().tolist())

    lengths = ints(batch.seq_lens_cpu if batch.seq_lens_cpu is not None else batch.seq_lens)
    table = kv.request_pool.req_to_token_pool()
    states = tuple(RequestState.from_req(req) for req in batch.all_seqs)
    if ints(batch.req_pool_indices) != tuple(s.row for s in states):
      raise ValueError("request pool rows disagree with leader batch")
    return cls(
      int(batch.forward_mode),
      states,
      ints(batch.input_ids),
      ints(batch.positions),
      lengths,
      tuple(batch.extend_seq_lens_cpu)
      if batch.forward_mode.is_extend()
      else (1,) * batch.batch_size,
      ints(batch.out_cache_loc),
      tuple(ints(table[state.row, :length]) for state, length in zip(states, lengths)),
    )

  def validate(self, capacity, page_size, max_requests, max_context_len, vocab_size):
    from miniinfer.scheduler.scheduler_batch import ForwardMode

    mode = ForwardMode(self.mode)
    if not (mode.is_extend() or mode.is_decode()):
      raise ValueError("unsupported EP forward mode")
    bs = len(self.requests)
    if not 0 < bs <= max_requests or not all(
      len(values) == bs for values in (self.seq_lens, self.q_lens, self.slot_rows)
    ):
      raise ValueError("inconsistent EP batch dimensions")
    ids = [r.req_id for r in self.requests]
    rows = [r.row for r in self.requests]
    if len(set(ids)) != bs or len(set(rows)) != bs:
      raise ValueError("duplicate EP request IDs or pool rows")
    if any(r.finished or r.retracted or not 0 <= r.row < max_requests for r in self.requests):
      raise ValueError("inactive request or invalid pool row")
    if any(not 0 <= t < vocab_size for t in self.input_ids):
      raise ValueError("invalid input token (EP does not accept placeholders)")
    if len(self.input_ids) != sum(self.q_lens) or len(self.positions) != len(self.input_ids):
      raise ValueError("query/input dimensions disagree")
    if len(self.out_slots) != len(self.input_ids):
      raise ValueError("query/output slot dimensions disagree")
    offset = 0
    written = set()
    prefixes = set()
    for length, q, slots in zip(self.seq_lens, self.q_lens, self.slot_rows):
      if not 0 < q <= length <= max_context_len or len(slots) != length:
        raise ValueError("invalid sequence/query lengths")
      if mode.is_decode() and q != 1:
        raise ValueError("decode requires one query per request")
      if any(not page_size <= s < capacity + page_size for s in slots):
        raise ValueError("KV slot outside negotiated capacity")
      for j, slot in enumerate(slots):
        if (
          slot % page_size != j % page_size
          or slot // page_size != slots[j // page_size * page_size] // page_size
        ):
          raise ValueError("invalid paged KV mapping")
      prefix = length - q
      prefixes.update(slots[:prefix])
      if self.positions[offset : offset + q] != tuple(range(prefix, length)):
        raise ValueError("positions disagree with prefix/query lengths")
      suffix = slots[prefix:]
      if self.out_slots[offset : offset + q] != suffix:
        raise ValueError("output KV slots disagree with request mapping")
      if written.intersection(suffix) or len(set(suffix)) != q:
        raise ValueError("duplicate KV writes across requests")
      written.update(suffix)
      offset += q
    if prefixes.intersection(written):
      raise ValueError("KV write aliases a shared prefix")

  def to_forward(self, kv, device, attn_backend):
    from miniinfer.scheduler.scheduler_batch import ForwardBatch, ForwardMode

    def tensor(values, dtype=torch.int64):
      return torch.tensor(values, dtype=dtype, device=device)

    rows = [r.row for r in self.requests]
    table = kv.request_pool.req_to_token_pool()
    for row, slots in zip(rows, self.slot_rows):
      table[row].zero_()
      table[row, : len(slots)] = tensor(slots, table.dtype)
    prefix = [n - q for n, q in zip(self.seq_lens, self.q_lens)]
    mode = ForwardMode(self.mode)
    batch = ForwardBatch(
      forward_mode=mode,
      batch_size=len(rows),
      input_ids=tensor(self.input_ids),
      positions=tensor(self.positions),
      req_pool_indices=tensor(rows),
      out_cache_loc=tensor(self.out_slots),
      attn_backend=attn_backend,
      seq_lens=tensor(self.seq_lens),
      seq_lens_int32=tensor(self.seq_lens, torch.int32),
      seq_lens_cpu=torch.tensor(self.seq_lens),
      max_seq_len=max(self.seq_lens),
    )
    if mode.is_extend():
      batch.extend_seq_lens = tensor(self.q_lens)
      batch.extend_prefix_lens = tensor(prefix)
      batch.extend_seq_lens_cpu = list(self.q_lens)
      batch.extend_prefix_lens_cpu = prefix
      batch.last_token_indices = tensor(self.q_lens).cumsum(0) - 1
    return batch


class RequestMirror:
  """Validate the leader's history transitions, including chunk-only forwards."""

  def __init__(self):
    self.states = {}
    self.packet = None
    self.tokens = None

  def prepare(self, packet):
    if self.packet is not None:
      raise RuntimeError("previous EP batch has not been committed")
    for state in packet.requests:
      previous = self.states.get(state.req_id)
      if previous and (previous.finished or previous.history != state.history):
        raise ValueError(f"request history diverged: {state.req_id}")
    self.packet = packet

  def sample(self, tokens):
    if self.packet is None or self.tokens is not None or len(tokens) != len(self.packet.requests):
      raise ValueError("sample does not match pending batch")
    self.tokens = tuple(int(t) for t in tokens)

  def commit(self, states):
    if self.packet is None or self.tokens is None:
      raise RuntimeError("commit requires forward and synchronized samples")
    if len(states) != len(self.packet.requests):
      raise ValueError("commit request count mismatch")
    for before, after, token in zip(self.packet.requests, states, self.tokens):
      expected = before.history if before.chunked else before.history + (token,)
      if after.req_id != before.req_id or after.history != expected:
        raise ValueError("commit history disagrees with rank 0 samples")
      if before.chunked and (after.row != -1 or after.finished):
        raise ValueError("chunk-only forward must release its temporary request row")
      if not before.chunked and after.row != before.row:
        raise ValueError("request row changed before release")
    self.states.update((s.req_id, s) for s in states)
    self.packet = self.tokens = None

  def release(self, ids):
    if self.packet is not None:
      raise RuntimeError("cannot release during an uncommitted batch")
    for req_id in ids:
      self.states.pop(req_id, None)


class CudaEPRank:
  """Production rank adapter. No CUDA imports are required for CPU protocol tests."""

  def __init__(self, config, rank):
    from miniinfer.engine.model_runner import ModelRunner
    from miniinfer.layers.fused_moe import MoEParallelConfig
    from miniinfer.utils.memory_utils import get_available_gpu_memory, get_kv_cache_bytes_per_token

    self.config = config
    self.device = f"cuda:{rank}"
    self.runner = ModelRunner(
      config,
      rank=rank,
      device=self.device,
      attn_backend=config.attention_backend,
      moe_parallel=MoEParallelConfig(
        ep_size=config.expert_parallel_size, ep_rank=rank, ep_group=dist.group.WORLD
      ),
    )
    if os.getenv("MINIINFER_EP_TRACE") == "1":
      for layer_id, decoder_layer in enumerate(self.runner.model.qwen3_moe.layers):
        moe = getattr(decoder_layer, "mlp", None)
        experts = getattr(moe, "experts", None)
        if experts is None:
          continue

        def trace_enter(module, args, rank=rank, layer_id=layer_id):
          shape = tuple(args[0].shape) if args and hasattr(args[0], "shape") else None
          print(f"EP_TRACE rank={rank} layer={layer_id} enter shape={shape}", flush=True)

        def trace_exit(module, args, output, rank=rank, layer_id=layer_id):
          print(f"EP_TRACE rank={rank} layer={layer_id} exit", flush=True)

        experts.register_forward_pre_hook(trace_enter)
        experts.register_forward_hook(trace_exit)
    torch.cuda.empty_cache()
    available = get_available_gpu_memory(
      gpu_memory_utilization=config.gpu_memory_utilization, device=rank
    )
    # Include request-table and pinned metadata device copies in the budget.
    available -= config.max_num_seqs * config.max_context_len * 8
    self.capacity = kv_capacity_from_bytes(
      available, get_kv_cache_bytes_per_token(config), config.page_size
    )

  def init_kv(self, capacity):
    from miniinfer.kvcache.kv_cache_manager import KVCacheManager

    c = self.config
    self.kv = KVCacheManager(
      size=capacity,
      max_requests=c.max_num_seqs,
      max_context_len=c.max_context_len,
      num_layers=c.hf_config.num_hidden_layers,
      num_heads=c.hf_config.num_key_value_heads,
      head_dim=getattr(
        c.hf_config, "head_dim", c.hf_config.hidden_size // c.hf_config.num_attention_heads
      ),
      dtype=c.dtype,
      device=self.device,
      enable_prefix_cache=c.enable_prefix_cache,
      page_size=c.page_size,
      max_extend_tokens=c.max_extend_len,
    )
    self.runner.set_kv_cache_mgr(self.kv)

  def init_cuda_graph(self, max_batch_size, max_context_len):
    if not self.config.enable_ep_cuda_graph:
      raise RuntimeError("EP CUDA Graph requires enable_ep_cuda_graph=True")
    dist.barrier()
    self.runner.init_cuda_graph(max_batch_size=max_batch_size, max_context_len=max_context_len)
    decode_buckets = tuple(sorted(self.runner.cuda_graph_runner.graphs))
    prefill_buckets = tuple(sorted(self.runner.piecewise_graph_runner.graphs)) if (
      self.runner.piecewise_graph_runner is not None
    ) else ()
    signatures = [None] * dist.get_world_size()
    dist.all_gather_object(signatures, (decode_buckets, prefill_buckets))
    if len({signature[0] for signature in signatures}) != 1:
      raise RuntimeError(f"EP ranks captured different decode graph buckets: {signatures}")
    if len({signature[1] for signature in signatures}) != 1:
      raise RuntimeError(f"EP ranks captured different prefill graph buckets: {signatures}")
    self.synchronize()
    dist.barrier()

  def prepare(self, packet):
    return packet.to_forward(self.kv, self.device, self.runner.attn_backend)

  def forward(self, batch):
    return self.runner.forward(batch)

  def forward_eager(self, batch):
    use_cuda_graph = self.runner.use_cuda_graph
    self.runner.use_cuda_graph = False
    try:
      return self.runner.forward(batch)
    finally:
      self.runner.use_cuda_graph = use_cuda_graph

  def release_cuda_graphs(self):
    runner = self.runner
    self.synchronize()
    # Exception frames may retain captured buffers while close() is running.
    # Reset handles explicitly before NCCL teardown instead of relying on GC.
    decode = runner.cuda_graph_runner
    if decode is not None:
      for captured in decode.graphs.values():
        captured.graph.reset()
      decode.graphs.clear()
    prefill = runner.piecewise_graph_runner
    if prefill is not None:
      for buffers in prefill.graphs.values():
        buffers["graph"].reset()
      prefill.graphs.clear()
    runner.cuda_graph_runner = None
    runner.piecewise_graph_runner = None
    runner.use_cuda_graph = False
    backend = runner.attn_backend
    if backend is not None and hasattr(backend, "_graph_metadata"):
      backend._graph_metadata.clear()
      if hasattr(backend, "_extend_graph_metadata"):
        backend._extend_graph_metadata.clear()
      backend.forward_metadata = None
    self.synchronize()

  def sample(self, logits, batch, **kwargs):
    return self.runner.sample(logits, batch, **kwargs)

  def synchronize(self):
    torch.cuda.synchronize(self.device)


def _init_group(config, rank, rendezvous, backend):
  if backend == "nccl":
    torch.cuda.set_device(rank)
  dist.init_process_group(
    backend,
    init_method=rendezvous,
    rank=rank,
    world_size=config.expert_parallel_size,
    timeout=timedelta(seconds=config.ep_timeout_seconds),
  )


def _validate(packet, config, capacity):
  packet.validate(
    capacity,
    config.page_size,
    config.max_num_seqs,
    config.max_context_len,
    config.hf_config.vocab_size,
  )


def _follower(conn, config, rank, rendezvous, backend, factory):
  adapter = None
  try:
    torch.set_num_threads(1)
    _init_group(config, rank, rendezvous, backend)
    adapter = factory(config, rank)
    mirror = RequestMirror()
    conn.send(("READY", adapter.capacity))
    capacity = None
    batch = None
    while True:
      command, value = conn.recv()
      if command == "INIT_KV":
        capacity = value
        adapter.init_kv(value)
        adapter.synchronize()
      elif command == "INIT_CUDA_GRAPH":
        adapter.init_cuda_graph(**value)
        adapter.synchronize()
      elif command == "RELEASE_CUDA_GRAPH":
        adapter.release_cuda_graphs()
      elif command == "PREPARE":
        _validate(value, config, capacity)
        mirror.prepare(value)
        batch = adapter.prepare(value)
      elif command in ("RUN", "RUN_EAGER"):
        if batch is None:
          raise RuntimeError("RUN without PREPARE")
        _trace_ep_graph(rank, f"{command.lower()}_begin")
        with torch.inference_mode():
          if command == "RUN_EAGER":
            adapter.forward_eager(batch)
          else:
            adapter.forward(batch)
        _trace_ep_graph(rank, f"{command.lower()}_forward_return")
        adapter.synchronize()
        _trace_ep_graph(rank, f"{command.lower()}_synchronize_return")
      elif command == "SAMPLE":
        mirror.sample(value)
      elif command == "COMMIT":
        mirror.commit(value)
        batch = None
      elif command == "RELEASE":
        mirror.release(value)
      elif command == "INSPECT":
        conn.send((command, mirror.states))
        continue
      elif command != "STOP":
        raise ValueError(f"unknown EP command: {command}")
      conn.send((command, None))
      if command == "STOP":
        break
  except BaseException:
    try:
      conn.send(("ERROR", traceback.format_exc()))
    except (OSError, EOFError):
      pass
  finally:
    if adapter is not None and getattr(config, "enable_ep_cuda_graph", False):
      try:
        adapter.release_cuda_graphs()
      except BaseException:
        pass
    conn.close()
    if dist.is_initialized():
      dist.destroy_process_group()


class ExpertParallelExecutor:
  """Rank 0 facade consumed by LLMEngine; injectable adapter exercises real IPC on CPU."""

  def __init__(self, config, *, backend="nccl", rank_factory=CudaEPRank):
    if dist.is_initialized():
      raise RuntimeError("EP engine requires ownership of an uninitialized distributed WORLD")
    if config.expert_parallel_size < 2:
      raise ValueError("EP executor requires at least two ranks")
    if backend == "nccl" and torch.cuda.device_count() < config.expert_parallel_size:
      raise ValueError("not enough visible CUDA devices for expert_parallel_size")
    self.config = config
    self.connections = []
    self.processes = []
    self.closed = False
    self.mirror = RequestMirror()
    self._verify_ep_graph = bool(
      getattr(config, "enable_ep_cuda_graph", False)
      and os.getenv("MINIINFER_EP_GRAPH_VERIFY") == "1"
    )
    self._ep_graph_verified = False
    self._owns_group = False
    self._temp = tempfile.TemporaryDirectory(prefix="miniinfer-ep-")
    rendezvous = (Path(self._temp.name) / "world").as_uri()
    ctx = mp.get_context("spawn")
    try:
      for rank in range(1, config.expert_parallel_size):
        parent, child = ctx.Pipe()
        process = ctx.Process(
          target=_follower,
          args=(child, config, rank, rendezvous, backend, rank_factory),
          name=f"miniinfer-ep-{rank}",
        )
        process.start()
        child.close()
        self.connections.append(parent)
        self.processes.append(process)
      _init_group(config, 0, rendezvous, backend)
      self._owns_group = True
      self.adapter = rank_factory(config, 0)
      self.rank_capacities = [self.adapter.capacity] + self._receive(
        "READY", config.ep_startup_timeout_seconds
      )
      self.max_total_tokens = common_kv_capacity(
        self.rank_capacities, config.page_size, config.max_total_tokens
      )
      self._send("INIT_KV", self.max_total_tokens)
      self.adapter.init_kv(self.max_total_tokens)
      self.adapter.synchronize()
      self._receive("INIT_KV", config.ep_startup_timeout_seconds)
    except BaseException:
      self.close(abort=True)
      raise

  @property
  def kv_cache_mgr(self):
    return self.adapter.kv

  def __getattr__(self, name):
    adapter = self.__dict__.get("adapter")
    if adapter is None:
      raise AttributeError(name)
    return getattr(adapter.runner, name)

  def _send(self, command, payload=None):
    if self.closed:
      raise RuntimeError("EP executor is closed")
    for conn in self.connections:
      conn.send((command, payload))

  def _receive(self, expected, timeout=None):
    deadline = time.monotonic() + (timeout or self.config.ep_timeout_seconds)
    pending = set(self.connections)
    results = {}
    while pending:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        raise TimeoutError(f"EP workers timed out during {expected}")
      for conn in wait(pending, timeout=min(remaining, 0.1)):
        rank = self.connections.index(conn) + 1
        try:
          command, value = conn.recv()
        except (EOFError, OSError) as exc:
          raise RuntimeError(f"EP rank {rank} exited during {expected}") from exc
        if command == "ERROR":
          raise RuntimeError(f"EP rank {rank} failed during {expected}:\n{value}")
        if command != expected:
          raise RuntimeError(f"EP rank {rank}: expected {expected}, received {command}")
        _trace_ep_graph(0, f"{expected.lower()}_rank{rank}_ack")
        results[conn] = value
        pending.remove(conn)
      for conn in pending:
        rank = self.connections.index(conn)
        if not self.processes[rank].is_alive() and not conn.poll():
          raise RuntimeError(f"EP rank {rank + 1} exited during {expected}")
    return [results[conn] for conn in self.connections]

  def execute(self, packet, leader_batch=None):
    try:
      _validate(packet, self.config, self.max_total_tokens)
      self.mirror.prepare(packet)
      self._send("PREPARE", packet)
      if leader_batch is None:
        leader_batch = self.adapter.prepare(packet)
      self._receive("PREPARE")

      from miniinfer.scheduler.scheduler_batch import ForwardMode

      verify_graph = (
        self._verify_ep_graph
        and not self._ep_graph_verified
        and ForwardMode(packet.mode).is_decode()
      )
      eager_output = None
      eager_reference = None
      if verify_graph:
        self._send("RUN_EAGER")
        with torch.inference_mode():
          eager_output = self.adapter.forward_eager(leader_batch)
        self.adapter.synchronize()
        self._receive("RUN_EAGER")
        eager_reference = eager_output.logits.detach().float().cpu().clone()
        eager_output = None

      self._send("RUN")
      with torch.inference_mode():
        output = self.adapter.forward(leader_batch)
      self.adapter.synchronize()
      self._receive("RUN")
      if verify_graph:
        graph_logits_cpu = output.logits.detach().float().cpu()
        torch.testing.assert_close(graph_logits_cpu, eager_reference, rtol=1e-3, atol=1e-3)
        self._ep_graph_verified = True
      return output
    except BaseException:
      self.close(abort=True)
      raise

  def forward(self, batch):
    return self.execute(BatchPacket.from_forward(batch, self.kv_cache_mgr), batch)

  def init_cuda_graph(self, max_batch_size, max_context_len):
    if not getattr(self.config, "enable_ep_cuda_graph", False):
      raise RuntimeError("EP CUDA Graph requires enable_ep_cuda_graph=True")
    payload = {"max_batch_size": max_batch_size, "max_context_len": max_context_len}
    try:
      self._send("INIT_CUDA_GRAPH", payload)
      self.adapter.init_cuda_graph(**payload)
      self.adapter.synchronize()
      self._receive("INIT_CUDA_GRAPH", self.config.ep_startup_timeout_seconds)
    except BaseException:
      self.close(abort=True)
      raise

  def sample(self, logits, batch, **kwargs):
    try:
      tokens = self.adapter.sample(logits, batch, **kwargs)
      values = tokens.detach().cpu().tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
      if any(not 0 <= t < self.config.hf_config.vocab_size for t in values):
        raise ValueError("invalid sampled token")
      self.mirror.sample(values)
      self._send("SAMPLE", values)
      self._receive("SAMPLE")
      return tokens
    except BaseException:
      self.close(abort=True)
      raise

  def commit(self, requests):
    self.commit_states(tuple(RequestState.from_req(req) for req in requests))

  def commit_states(self, states):
    try:
      self.mirror.commit(states)
      self._send("COMMIT", states)
      self._receive("COMMIT")
    except BaseException:
      self.close(abort=True)
      raise

  def release(self, ids):
    try:
      self.mirror.release(ids)
      self._send("RELEASE", ids)
      self._receive("RELEASE")
    except BaseException:
      self.close(abort=True)
      raise

  def inspect_workers(self):
    self._send("INSPECT")
    return self._receive("INSPECT")

  def close(self, abort=False):
    if self.closed:
      return
    try:
      if not abort and getattr(self.config, "enable_ep_cuda_graph", False):
        self._send("RELEASE_CUDA_GRAPH")
        self.adapter.release_cuda_graphs()
        self._receive("RELEASE_CUDA_GRAPH", min(5, self.config.ep_timeout_seconds))
      if not abort:
        self._send("STOP")
        self._receive("STOP", min(5, self.config.ep_timeout_seconds))
    except (OSError, RuntimeError, TimeoutError):
      abort = True
    finally:
      self.closed = True
      if abort:
        for process in self.processes:
          if process.is_alive():
            process.terminate()
      if getattr(self.config, "enable_ep_cuda_graph", False):
        try:
          self.adapter.release_cuda_graphs()
        except BaseException:
          pass
      if self._owns_group and dist.is_initialized():
        dist.destroy_process_group()
      for process in self.processes:
        process.join(timeout=3)
        if process.is_alive():
          process.kill()
          process.join(timeout=3)
      for conn in self.connections:
        conn.close()
      self._temp.cleanup()
