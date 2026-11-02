"""Exercise production EP IPC and full tiny Qwen3-MoE forwards in four Gloo ranks."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import json
import multiprocessing as mp
import time

import pytest
import torch
import torch.distributed as dist

from miniinfer.config.engine.config import EngineConfig
from miniinfer.engine.expert_parallel import (
  BatchPacket,
  CudaEPRank,
  ExpertParallelExecutor,
  RequestMirror,
  RequestState,
  common_kv_capacity,
  kv_capacity_from_bytes,
)
from miniinfer.scheduler.scheduler_batch import ForwardMode
from tests.ep_reference import CpuEPRank
from tests.moe_reference import make_checkpoint


def test_graph_release_resets_handles_even_with_external_references():
  decode_graph = Mock()
  prefill_graph = Mock()
  decode = SimpleNamespace(graphs={1: SimpleNamespace(graph=decode_graph)})
  prefill = SimpleNamespace(graphs={256: {"graph": prefill_graph}})
  backend = SimpleNamespace(_graph_metadata={1: None}, _extend_graph_metadata={256: None},
                            forward_metadata=object())
  rank = CudaEPRank.__new__(CudaEPRank)
  rank.runner = SimpleNamespace(cuda_graph_runner=decode, piecewise_graph_runner=prefill,
                                use_cuda_graph=True, attn_backend=backend)
  rank.synchronize = Mock()
  rank.release_cuda_graphs()
  decode_graph.reset.assert_called_once()
  prefill_graph.reset.assert_called_once()
  assert decode.graphs == prefill.graphs == {}
  assert rank.runner.cuda_graph_runner is None
  assert rank.runner.piecewise_graph_runner is None
  assert not rank.runner.use_cuda_graph
  assert backend.forward_metadata is None


@pytest.fixture
def config(tmp_path, monkeypatch):
  monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  c, _ = make_checkpoint(tmp_path)
  c["model_type"] = "qwen3_moe"
  c.update(norm_topk_prob=True, rope_theta=1000000.0)
  (tmp_path / "config.json").write_text(json.dumps(c))
  return EngineConfig(
    str(tmp_path),
    dtype=torch.float32,
    expert_parallel_size=4,
    enforce_eager=True,
    enable_overlap=False,
    page_size=4,
    max_num_seqs=4,
    max_context_len=32,
    max_total_tokens=53,
    ep_timeout_seconds=12,
    ep_startup_timeout_seconds=60,
  )


def packet(*, history=(), chunked=False):
  return BatchPacket(
    int(ForwardMode.EXTEND),
    (RequestState(10, 0, tuple(history), chunked),),
    (2, 3, 4),
    (0, 1, 2),
    (3,),
    (3,),
    (4, 5, 6),
    ((4, 5, 6),),
  )


def assert_no_children(executor):
  assert executor.closed
  assert all(not p.is_alive() for p in executor.processes)
  assert not dist.is_initialized()


def finish(executor, batch, output, *, done=False):
  forward = batch.to_forward(executor.adapter.kv, "cpu", executor.attn_backend)
  tokens = executor.sample(output.logits, forward)
  states = tuple(
    replace(
      s,
      history=s.history if s.chunked else s.history + (int(token),),
      row=-1 if s.chunked else s.row,
      finished=done and not s.chunked,
    )
    for s, token in zip(batch.requests, tokens)
  )
  executor.commit_states(states)
  for replica in executor.inspect_workers():
    assert all(replica[s.req_id] == s for s in states)
  return states


def hf_logits(config, sequences):
  from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
  from safetensors.torch import load_file
  from pathlib import Path

  c = json.loads((Path(config.model_path) / "config.json").read_text())
  model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**c)).eval()
  weights = {}
  for shard in Path(config.model_path).glob("*.safetensors"):
    weights.update(load_file(shard))
  model.load_state_dict(weights, strict=True)
  with torch.inference_mode():
    return torch.cat([model(torch.tensor([s])).logits[:, -1] for s in sequences])


def test_four_rank_model_protocol_sampling_and_kv(config):
  executor = ExpertParallelExecutor(config, backend="gloo", rank_factory=CpuEPRank)
  try:
    assert executor.rank_capacities == [64, 60, 56, 52]
    assert executor.max_total_tokens == 52
    first = replace(
      packet(chunked=True),
      input_ids=(2, 3, 4, 5),
      positions=(0, 1, 2, 3),
      seq_lens=(4,),
      q_lens=(4,),
      out_slots=(4, 5, 6, 7),
      slot_rows=((4, 5, 6, 7),),
    )
    result = executor.execute(first)
    torch.testing.assert_close(
      result.logits, hf_logits(config, [[2, 3, 4, 5]]), atol=2e-6, rtol=2e-5
    )
    states = finish(executor, first, result)
    assert states[0].history == ()

    # A temporary row was released; another request shares one full prefix page.
    mixed = BatchPacket(
      int(ForwardMode.MIXED),
      (replace(states[0], row=2, chunked=False), RequestState(20, 0, ())),
      (6, 7, 8),
      (4, 4, 5),
      (5, 6),
      (1, 2),
      (12, 16, 17),
      ((4, 5, 6, 7, 12), (4, 5, 6, 7, 16, 17)),
    )
    result = executor.execute(mixed)
    torch.testing.assert_close(
      result.logits, hf_logits(config, [[2, 3, 4, 5, 6], [2, 3, 4, 5, 7, 8]]), atol=2e-6, rtol=2e-5
    )
    states = finish(executor, mixed, result)
    tokens = [s.history[-1] for s in states]

    decode = BatchPacket(
      int(ForwardMode.DECODE),
      states,
      tuple(tokens),
      (5, 6),
      (6, 7),
      (1, 1),
      (13, 18),
      (mixed.slot_rows[0] + (13,), mixed.slot_rows[1] + (18,)),
    )
    result = executor.execute(decode)
    expected = hf_logits(config, [[2, 3, 4, 5, 6, tokens[0]], [2, 3, 4, 5, 7, 8, tokens[1]]])
    torch.testing.assert_close(result.logits, expected, atol=2e-6, rtol=2e-5)
    states = finish(executor, decode, result, done=True)
    assert executor.adapter.sample_count == 3
    executor.release([10, 20])
    assert all(s == {} for s in executor.inspect_workers())
    # Reuse both a request row and physical pages after finish/release.
    reused = replace(packet(), requests=(RequestState(30, 0, ()),))
    finish(executor, reused, executor.execute(reused), done=True)
  finally:
    executor.close()
  executor.close()
  assert_no_children(executor)


def test_ep_cuda_graph_capture_command_is_coordinated(config):
  config.enable_ep_cuda_graph = True
  executor = ExpertParallelExecutor(config, backend="gloo", rank_factory=CpuEPRank)
  try:
    executor.init_cuda_graph(max_batch_size=1, max_context_len=32)
    assert executor.adapter.cuda_graph_request == (1, 32)
  finally:
    executor.close()
  assert_no_children(executor)
  assert executor.adapter.cuda_graph_released


@pytest.mark.parametrize("field", ["fail_prepare_rank", "exit_prepare_rank", "stall_prepare_rank"])
def test_worker_failure_aborts_owned_processes(config, field):
  setattr(config, field, 2)
  executor = ExpertParallelExecutor(config, backend="gloo", rank_factory=CpuEPRank)
  executor.config.ep_timeout_seconds = 2
  started = time.monotonic()
  with pytest.raises((RuntimeError, TimeoutError), match="EP rank|timed out"):
    executor.execute(packet())
  assert time.monotonic() - started < 15
  assert_no_children(executor)


@pytest.mark.parametrize("rank", [0, 2])
def test_partial_initialization_failure(config, rank):
  config.fail_init_rank = rank
  before = {p.pid for p in mp.active_children()}
  with pytest.raises(RuntimeError, match="initialization failure"):
    ExpertParallelExecutor(config, backend="gloo", rank_factory=CpuEPRank)
  assert {p.pid for p in mp.active_children()} == before
  assert not dist.is_initialized()


def test_invalid_packet_aborts_before_forward(config):
  executor = ExpertParallelExecutor(config, backend="gloo", rank_factory=CpuEPRank)
  with pytest.raises(ValueError, match="outside negotiated"):
    executor.execute(replace(packet(), out_slots=(1000, 5, 6), slot_rows=((1000, 5, 6),)))
  assert_no_children(executor)


@pytest.mark.parametrize(
  "changes,match",
  [
    ({"positions": (0, 1, 5)}, "positions"),
    ({"input_ids": (-1, 3, 4)}, "input token"),
    ({"out_slots": (4, 5)}, "dimensions"),
    ({"q_lens": (4,)}, "dimensions"),
    ({"slot_rows": ((4, 5, 8),), "out_slots": (4, 5, 8)}, "paged KV"),
    ({"requests": (RequestState(10, 9, ()),)}, "pool row"),
  ],
)
def test_packet_validation(changes, match):
  with pytest.raises(ValueError, match=match):
    replace(packet(), **changes).validate(52, 4, 4, 32, 32)


def test_capacity_does_not_hide_oom():
  assert common_kv_capacity([101, 70, 99, 120], 16) == 64
  assert common_kv_capacity([101, 70], 16, 40) == 32
  assert kv_capacity_from_bytes(100, 10, 4, 0) == 4
  assert kv_capacity_from_bytes(-100, 10, 4) == 0
  with pytest.raises(ValueError, match="insufficient"):
    common_kv_capacity([16, 100], 16)


def test_mirror_rejects_divergent_history():
  mirror = RequestMirror()
  mirror.prepare(packet())
  mirror.sample([9])
  with pytest.raises(ValueError, match="history"):
    mirror.commit([RequestState(10, 0, (8,))])
  mirror.commit([RequestState(10, 0, (9,))])
  with pytest.raises(ValueError, match="history diverged"):
    mirror.prepare(packet())


@pytest.mark.parametrize(
  "overrides,match",
  [
    ({"expert_parallel_size": 0}, "positive"),
    ({"tensor_parallel_size": 2}, "Attention TP"),
    ({"enforce_eager": False}, "requires"),
    ({"enable_overlap": True}, "requires"),
    ({"expert_parallel_size": 1, "enable_ep_cuda_graph": True}, "requires expert_parallel_size"),
    ({"expert_parallel_size": 3}, "divisible"),
    ({"max_total_tokens": 1}, "two pages"),
  ],
)
def test_config_validation(config, overrides, match):
  from dataclasses import asdict

  values = asdict(config)
  values.update(overrides)
  values.pop("hf_config")
  with pytest.raises(ValueError, match=match):
    EngineConfig(**values)


def test_llm_engine_entry_generate_stream_submit_and_cleanup(config, monkeypatch):
  import queue
  import miniinfer.engine.expert_parallel as ep_module
  from miniinfer.engine.llm_engine import LLMEngine
  from miniinfer.engine.detokenizer import IncrementalDecoder
  from miniinfer.scheduler.scheduler_batch import ForwardBatch, ScheduledBatch
  from miniinfer.utils.sampling_params import SamplingParams
  from miniinfer.scheduler.scheduler import Scheduler

  executor_class = ep_module.ExpertParallelExecutor
  monkeypatch.setattr(
    ep_module,
    "ExpertParallelExecutor",
    lambda c: executor_class(c, backend="gloo", rank_factory=CpuEPRank),
  )

  class Tokenizer:
    eos_token_id = -1

    def decode(self, ids, **kwargs):
      return " ".join(map(str, ids))

    def batch_decode(self, sequences, **kwargs):
      return [self.decode(ids) for ids in sequences]

  class ControlledScheduler:
    # Only scheduling/allocation is replaced. Public engine control, model,
    # loader, sampler, result handling and follower protocol stay active.
    eos_token_id = -1
    _handle_finished_requests = Scheduler._handle_finished_requests

    def __init__(self, kv):
      self.kv = kv
      self.reqs = []
      self.running_batch = ScheduledBatch(reqs=[])
      self.pending_release_reqs = []
      self.finished_reqs = []

    def add(self, req):
      self.reqs.append(req)

    def _filter_batch(self, batch):
      batch.reqs = [r for r in batch.reqs if not r.finished]

    def has_unfinished(self):
      return bool(self.pending_release_reqs) or any(not r.finished for r in self.reqs)

    def drain_pending_releases(self):
      for req in self.pending_release_reqs:
        req.req_pool_idx = -1
      self.pending_release_reqs.clear()

    def schedule(self, device, **kwargs):
      pending = [r for r in self.reqs if not r.finished]
      if not pending:
        return None
      req = pending[0]
      req.req_pool_idx = 0
      req.fill_ids = req.origin_input_ids + req.output_ids
      n = len(req.fill_ids)
      q = 1 if req.output_ids else n
      slots = tuple(range(4, 4 + n))
      self.kv.table[0, :n] = torch.tensor(slots)
      batch = ScheduledBatch(
        reqs=[req],
        device="cpu",
        forward_mode=ForwardMode.DECODE if q == 1 else ForwardMode.EXTEND,
        input_ids=torch.tensor(req.fill_ids[-q:]),
        req_pool_indices=torch.tensor([0]),
        out_cache_loc=torch.tensor(slots[-q:]),
        seq_lens=torch.tensor([n]),
        seq_lens_cpu=torch.tensor([n]),
        prefix_lens=[n - q],
        extend_lens=[q],
      )
      self.running_batch = batch
      return batch

  def init_cpu_state(engine, c):
    engine.tokenizer = Tokenizer()
    engine.detokenizer = IncrementalDecoder(engine.tokenizer)
    engine.scheduler = ControlledScheduler(engine.kv_cache_mgr)
    engine.overlap_executor = None
    engine._pending_requests = queue.Queue()
    engine._started = True

  def cpu_forward(batch, backend):
    n = int(batch.seq_lens[0])
    q = len(batch.input_ids)
    p = BatchPacket(
      int(batch.forward_mode),
      tuple(RequestState.from_req(r) for r in batch.reqs),
      tuple(batch.input_ids.tolist()),
      tuple(range(n - q, n)),
      (n,),
      (q,),
      tuple(batch.out_cache_loc.tolist()),
      (tuple(range(4, 4 + n)),),
    )
    result = p.to_forward(backend.kv, "cpu", backend)
    result.all_seqs = batch.reqs
    return result

  monkeypatch.setattr(LLMEngine, "_init_scheduler_and_tokenizer", init_cpu_state)
  monkeypatch.setattr(ForwardBatch, "init_new", cpu_forward)
  with LLMEngine(
    config.model_path,
    dtype=torch.float32,
    expert_parallel_size=4,
    enforce_eager=True,
    enable_overlap=False,
    page_size=4,
    max_num_seqs=4,
    max_context_len=32,
    max_total_tokens=52,
    ep_timeout_seconds=12,
  ) as engine:
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
    result = engine.generate([[2, 3, 4]], params, use_tqdm=False)
    assert len(result[0]["token_ids"]) == 3
    assert all(s == {} for s in engine.ep_executor.inspect_workers())
    # Non-overlap serial step consumes submit_request, and stream_generate
    # drains pending releases without depending on an OverlapExecutor.
    req_id = engine.submit_request([2, 3, 4], params)
    seen = []
    while not engine.is_finished() or not engine._pending_requests.empty():
      seen.extend(o.token_id for o in engine.step_overlap().outputs if o.request_id == req_id)
    assert seen == result[0]["token_ids"]
    streamed = list(engine.stream_generate([[2, 3, 4]], params))
    assert [o.token_id for o in streamed] == seen
    assert engine.ep_executor.adapter.sample_count == 9
    executor = engine.ep_executor
  assert_no_children(executor)
