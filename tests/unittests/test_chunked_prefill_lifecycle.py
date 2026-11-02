"""GPU regressions for chunk ownership, scheduling, and decode page boundaries."""

from types import SimpleNamespace

import pytest
import torch

from miniinfer.engine.llm_engine import LLMEngine
from miniinfer.kvcache.kv_cache_manager import KVCacheManager
from miniinfer.scheduler.scheduler import Scheduler
from miniinfer.scheduler.scheduler_batch import BatchResult, Req, ScheduledBatch
from miniinfer.utils.sampling_params import SamplingParams


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
PAGE_SIZE = 256


def _manager(enable_prefix_cache=False):
  return KVCacheManager(
    size=PAGE_SIZE * 16,
    max_requests=8,
    max_context_len=2048,
    num_layers=1,
    num_heads=1,
    head_dim=64,
    dtype=torch.float16,
    device="cuda",
    enable_prefix_cache=enable_prefix_cache,
    page_size=PAGE_SIZE,
  )


@pytest.mark.parametrize("enable_prefix_cache", [False, True])
def test_schedule_warmup_restores_free_pages_and_scheduler_state(enable_prefix_cache):
  manager = _manager(enable_prefix_cache)
  config = SimpleNamespace(
    max_num_seqs=8,
    max_extend_len=4096,
    page_size=PAGE_SIZE,
    enable_chunked_prefill=True,
    chunked_prefill_size=4096,
    prefill_budget_policy="aggregate",
    max_num_batched_tokens=512,
  )
  scheduler = Scheduler(config, SimpleNamespace(eos_token_id=0), manager)
  waiting = [Req([3] * 768)]
  scheduler.waiting_queue = waiting
  running = scheduler.running_batch
  initial_free_pages = manager.token_allocator.free_pages.clone()

  scheduler.warmup_schedule_path(torch.device("cuda"))

  assert scheduler.waiting_queue is waiting
  assert scheduler.running_batch is running
  assert not scheduler.chunking_reqs
  assert not scheduler.pending_release_reqs
  assert torch.equal(manager.token_allocator.free_pages, initial_free_pages)
  assert sorted(manager.request_pool.free_slots) == list(range(8))
  if manager.prefix_cache is not None:
    assert manager.prefix_cache.evictable_size() == 0


@pytest.mark.parametrize("enable_prefix_cache", [False, True])
def test_chunked_mixed_lifecycle_preserves_slots_and_reclaims_pages(enable_prefix_cache):
  manager = _manager(enable_prefix_cache)
  config = SimpleNamespace(
    max_num_seqs=8,
    max_extend_len=4096,
    page_size=PAGE_SIZE,
    enable_chunked_prefill=True,
    chunked_prefill_size=4096,
    prefill_budget_policy="aggregate",
    max_num_batched_tokens=512,
  )
  scheduler = Scheduler(config, SimpleNamespace(eos_token_id=0), manager)
  requests = [
    Req([token] * 768, SamplingParams(max_tokens=4, ignore_eos=True))
    for token in (1, 2)
  ]
  scheduler.waiting_queue.extend(requests)
  engine = SimpleNamespace(
    kv_cache_mgr=manager,
    scheduler=scheduler,
    detokenizer=SimpleNamespace(flush=lambda _: "", cleanup=lambda _: None),
    _streaming=False,
  )
  previous_slots = {}
  prefill_tokens = 0
  saw_mixed = False

  for _ in range(20):
    batch = scheduler.schedule(torch.device("cuda"))
    if batch is None:
      break
    assert len({id(req) for req in batch.reqs}) == len(batch.reqs)
    assert len({id(chunk.req) for chunk in scheduler.chunking_reqs}) == len(
      scheduler.chunking_reqs
    )
    decode_ids = {id(req) for req in (batch.decoding_reqs or [])}
    saw_mixed |= bool(decode_ids)
    if batch.forward_mode.is_decode():
      compute_tokens = len(batch.reqs)
    else:
      compute_tokens = sum(batch.extend_lens)
      prefill_tokens += sum(
        length for req, length in zip(batch.reqs, batch.extend_lens)
        if id(req) not in decode_ids
      )
    assert compute_tokens <= 512

    for req, length in zip(batch.reqs, batch.seq_lens_cpu.tolist()):
      slots = manager.request_pool.read(req.req_pool_idx, slice(0, length)).clone()
      old_slots = previous_slots.get(req.req_id)
      if old_slots is not None:
        assert torch.equal(slots[:len(old_slots)], old_slots)
      assert slots.unique().numel() == len(slots)
      previous_slots[req.req_id] = slots

    result = BatchResult(logits=None, next_token_ids=torch.full((len(batch.reqs),), 7))
    LLMEngine._process_step_result(engine, batch, result)
    scheduler.drain_pending_releases()
  else:
    pytest.fail("chunked requests did not finish within 20 scheduler iterations")

  assert not scheduler.has_unfinished()
  assert saw_mixed
  assert prefill_tokens == 2 * 768
  assert [len(req.output_ids) for req in requests] == [4, 4]
  assert all(req.finished and req.req_pool_idx == -1 for req in requests)
  assert sorted(manager.request_pool.free_slots) == list(range(8))
  if manager.prefix_cache is not None:
    manager.prefix_cache.evict(manager.size)
  assert manager.token_allocator.available_size() == manager.size


@pytest.mark.parametrize("mode", ["decode", "mixed"])
@pytest.mark.parametrize("prompt_len", [255, 256, 257])
def test_decode_allocator_uses_post_append_length(mode, prompt_len):
  manager = _manager()
  req = Req([1] * prompt_len, SamplingParams(max_tokens=4))
  manager.prefix_for_waiting_req(req)
  initial = ScheduledBatch.init_new([req], device=torch.device("cuda"))
  manager.prepare_for_extend(initial)
  old_slots = initial.out_cache_loc.clone()
  free_before = manager.token_allocator.available_size()
  first_free_page = int(manager.token_allocator.free_pages[0].item())
  req.append_output_token(7)

  if mode == "decode":
    batch = initial
    manager.prepare_for_decode(batch)
  else:
    new_req = Req([2] * 8, SamplingParams(max_tokens=4))
    manager.prefix_for_waiting_req(new_req)
    batch = ScheduledBatch.init_new([new_req, req], device=torch.device("cuda"))
    manager.prepare_for_mixed(
      batch, [new_req], [req],
      decode_seq_lens=initial.seq_lens,
      decode_seq_lens_cpu=initial.seq_lens_cpu,
    )

  expected_slot = (
    first_free_page * PAGE_SIZE if prompt_len % PAGE_SIZE == 0
    else int(old_slots[-1].item()) + 1
  )
  assert int(batch.out_cache_loc[-1].item()) == expected_slot
  actual_slots = manager.request_pool.read(req.req_pool_idx, slice(0, prompt_len + 1))
  assert torch.equal(actual_slots[:prompt_len], old_slots)
  assert int(actual_slots[-1].item()) == expected_slot
  pages_used = int(prompt_len % PAGE_SIZE == 0) + int(mode == "mixed")
  assert manager.token_allocator.available_size() == free_before - pages_used * PAGE_SIZE
