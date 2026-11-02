"""V100 tests: FP32 references use logical K/V, independent of the page table."""

from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
  reason="requires V100 (SM70) and the pinned vllm_flash_attn extension",
)


def reference(q, keys, values, q_lens, kv_lens=None):
  result = []
  start = 0
  for i, length in enumerate(q_lens):
    k_len = len(keys[i]) if kv_lens is None else kv_lens[i]
    qb = q[start : start + length].float()
    kb = keys[i][:k_len].float().repeat_interleave(q.shape[1] // keys[i].shape[1], dim=1)
    vb = values[i][:k_len].float().repeat_interleave(q.shape[1] // values[i].shape[1], dim=1)
    scores = torch.einsum("qhd,khd->hqk", qb, kb) * q.shape[-1] ** -0.5
    rows = torch.arange(length, device=q.device)[:, None] + k_len - length
    cols = torch.arange(k_len, device=q.device)[None, :]
    scores.masked_fill_(cols > rows, float("-inf"))
    result.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), vb))
    start += length
  return torch.cat(result)


def paged_case(q_lens, kv_lens, head_dim=64, page_size=256, heads=(8, 2), shared=False):
  torch.manual_seed(123)
  hq, hk = heads
  pages_per_req = [(length + page_size - 1) // page_size for length in kv_lens]
  num_pages = sum(pages_per_req) + 3
  # Interleaved K/V have nontrivial page strides, as in common serving layouts.
  cache = torch.full(
    (num_pages, 2, page_size, hk, head_dim), float("nan"), device="cuda", dtype=torch.float16
  )
  k_cache, v_cache = cache.unbind(1)
  table_cpu = torch.zeros(len(q_lens), max(pages_per_req), dtype=torch.int32)
  # Reversed random physical pages catch assumptions of consecutive page ids.
  permutation = (torch.randperm(num_pages - 1) + 1).tolist()
  keys, values, locations = [], [], []
  cursor = 0
  for i, length in enumerate(kv_lens):
    pages = permutation[cursor : cursor + pages_per_req[i]]
    cursor += pages_per_req[i]
    if shared and i > 0:
      pages[0] = int(table_cpu[0, 0])
    table_cpu[i, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
    kb = torch.randn(length, hk, head_dim, device="cuda", dtype=torch.float16)
    vb = torch.randn_like(kb)
    if shared and i > 0:
      kb[:page_size].copy_(keys[0][:page_size])
      vb[:page_size].copy_(values[0][:page_size])
    for j, page in enumerate(pages):
      block_len = min(page_size, length - j * page_size)
      k_cache[page, :block_len].copy_(kb[j * page_size : j * page_size + block_len])
      v_cache[page, :block_len].copy_(vb[j * page_size : j * page_size + block_len])
    keys.append(kb)
    values.append(vb)
    slots = torch.tensor(pages, device="cuda")[:, None] * page_size
    slots = (slots + torch.arange(page_size, device="cuda")).flatten()[:length]
    locations.append(slots)
  # Simulate a Q view split from a fused QKV projection.
  qkv = torch.randn(sum(q_lens), (hq + 2 * hk) * head_dim, device="cuda", dtype=torch.float16)
  q = qkv[:, : hq * head_dim].view(-1, hq, head_dim)
  return SimpleNamespace(
    q=q,
    keys=keys,
    values=values,
    locations=locations,
    k_cache=k_cache,
    v_cache=v_cache,
    table=table_cpu.to("cuda"),
    cu_q=torch.tensor(
      [0] + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32, device="cuda"
    ),
    lengths=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
    q_lens=q_lens,
    kv_lens=kv_lens,
    page_size=page_size,
  )


def operator(case):
  from vllm_flash_attn import flash_attn_varlen_func

  return flash_attn_varlen_func(
    q=case.q,
    k=case.k_cache,
    v=case.v_cache,
    cu_seqlens_q=case.cu_q,
    seqused_k=case.lengths,
    max_seqlen_q=max(case.q_lens),
    max_seqlen_k=max(case.kv_lens),
    block_table=case.table,
    causal=True,
  )


@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 192, 256])
@pytest.mark.parametrize("page_size", [16, 256])
@pytest.mark.parametrize("kind", ["prefill", "prefix", "decode", "mixed"])
def test_operator_paged(head_dim, page_size, kind):
  lengths = {
    "prefill": ([17, 257], [17, 257]),
    "prefix": ([3, 17], [259, 513]),
    "decode": ([1, 1], [257, 511]),
    "mixed": ([17, 1, 9], [273, 511, 9]),
  }
  q_lens, kv_lens = lengths[kind]
  case = paged_case(q_lens, kv_lens, head_dim, page_size)
  torch.testing.assert_close(
    operator(case).float(), reference(case.q, case.keys, case.values, q_lens), rtol=0.01, atol=0.01
  )


@pytest.mark.parametrize("heads", [(4, 4), (8, 1), (8, 2)])
def test_operator_shared_prefix(heads):
  case = paged_case([3, 7], [259, 263], heads=heads, shared=True)
  torch.testing.assert_close(
    operator(case).float(),
    reference(case.q, case.keys, case.values, case.q_lens),
    rtol=0.01,
    atol=0.01,
  )


@pytest.mark.parametrize("heads", [(4, 4), (8, 1), (8, 2)])
def test_operator_decode_head_layouts(heads):
  # MHA decode skips the kernel's GQA head-to-query transpose; MQA uses a
  # different group size. Both must follow the same logical reference.
  case = paged_case([1, 1], [257, 513], heads=heads)
  torch.testing.assert_close(
    operator(case).float(),
    reference(case.q, case.keys, case.values, case.q_lens),
    rtol=0.01,
    atol=0.01,
  )


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("page_size", [16, 256])
@pytest.mark.parametrize("q_len,kv_len", [(1057, 1057), (1057, 2097), (1, 4097)])
def test_operator_long_context(head_dim, page_size, q_len, kv_len):
  # max_seqlen_q >= 1024 selects the fork's separate prefill specialization;
  # long decode also exercises automatic split-KV selection.
  case = paged_case([q_len], [kv_len], head_dim, page_size)
  torch.testing.assert_close(
    operator(case).float(), reference(case.q, case.keys, case.values, [q_len]), rtol=0.01, atol=0.01
  )


def test_operator_graph_changes_lengths_and_values():
  case = paged_case([1, 1], [511, 511], head_dim=128)
  operator(case)
  torch.cuda.synchronize()
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    out = operator(case)
  case.q.mul_(0.5)
  case.lengths.copy_(torch.tensor([257, 129], device="cuda", dtype=torch.int32))
  graph.replay()
  torch.testing.assert_close(
    out.float(), reference(case.q, case.keys, case.values, [1, 1], [257, 129]), rtol=0.01, atol=0.01
  )


@pytest.mark.parametrize("bucket", [256, 512])
def test_operator_extend_graph_changes_query_layout_and_prefix(bucket):
  from vllm_flash_attn import flash_attn_varlen_func

  case = paged_case([1, 1, bucket - 2], [512] * 3, head_dim=128, heads=(32, 4))
  case.q = case.q.contiguous()

  def forward():
    return flash_attn_varlen_func(
      q=case.q,
      k=case.k_cache,
      v=case.v_cache,
      cu_seqlens_q=case.cu_q,
      seqused_k=case.lengths,
      max_seqlen_q=bucket,
      max_seqlen_k=512,
      block_table=case.table,
      causal=True,
    )

  forward()
  torch.cuda.synchronize()
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    out = forward()
  for q_lens in ([16, bucket - 17, 1], [bucket - 2, 1, 1], [1, 44, bucket - 45]):
    kv_lens = [length + 256 if length <= 256 else length for length in q_lens]
    assert max(kv_lens) <= 512
    case.q.mul_(0.75)
    case.cu_q.copy_(torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0).tolist()),
                                dtype=torch.int32, device="cuda"))
    case.lengths.copy_(torch.tensor(kv_lens, dtype=torch.int32, device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
      out.float(), reference(case.q, case.keys, case.values, q_lens, kv_lens),
      rtol=0.01, atol=0.01,
    )


def backend_case(q_lens, kv_lens, head_dim=128, page_size=256, shared=False):
  from miniinfer.kvcache.kv_cache_manager import KVCacheManager
  from miniinfer.layers.attention import AttentionImpl
  from miniinfer.layers.attention_backend import FlashAttentionV100Backend
  from miniinfer.scheduler.scheduler_batch import ForwardBatch, ForwardMode

  case = paged_case(q_lens, kv_lens, head_dim, page_size, shared=shared)
  bs = len(q_lens)
  manager = KVCacheManager(
    size=(len(case.k_cache) - 1) * page_size,
    max_requests=bs + 2,
    max_context_len=max(kv_lens) + 32,
    num_layers=1,
    num_heads=2,
    head_dim=head_dim,
    device="cuda",
    page_size=page_size,
    max_extend_tokens=sum(q_lens),
  )
  keys, values = manager.get_kv_buffer(0)
  keys.copy_(case.k_cache.reshape_as(keys))
  values.copy_(case.v_cache.reshape_as(values))
  req_indices = torch.tensor(list(reversed(range(bs))), device="cuda")
  for i, slots in enumerate(case.locations):
    manager.request_pool.req_to_token[int(req_indices[i]), : len(slots)] = slots
  slots = torch.cat([loc[-length:] for loc, length in zip(case.locations, q_lens)])
  new_k = torch.cat([kb[-length:] for kb, length in zip(case.keys, q_lens)])
  new_v = torch.cat([vb[-length:] for vb, length in zip(case.values, q_lens)])
  # Require the actual MiniInfer store path to fill the newly allocated suffix.
  keys[slots] = float("nan")
  values[slots] = float("nan")
  mode = ForwardMode.DECODE if max(q_lens) == 1 else ForwardMode.MIXED
  batch = ForwardBatch(
    forward_mode=mode,
    batch_size=bs,
    input_ids=torch.zeros(sum(q_lens), device="cuda", dtype=torch.int64),
    req_pool_indices=req_indices,
    out_cache_loc=slots,
    seq_lens=case.lengths.to(torch.int64),
    seq_lens_int32=case.lengths,
    seq_lens_cpu=torch.tensor(kv_lens),
    max_seq_len=max(kv_lens),
    extend_seq_lens_cpu=q_lens,
    extend_prefix_lens_cpu=[s - q for s, q in zip(kv_lens, q_lens)],
  )
  backend = FlashAttentionV100Backend(manager)
  batch.attn_backend = backend
  layer = AttentionImpl(8, head_dim, head_dim**-0.5, 2, 0)
  return case, manager, backend, layer, batch, new_k, new_v


@pytest.mark.parametrize("page_size", [16, 256])
@pytest.mark.parametrize(
  "lengths",
  [
    ([17, 257], [17, 257]),
    ([3, 17], [259, 513]),
    ([1, 1], [257, 511]),
    ([17, 1, 9], [273, 511, 9]),
  ],
)
def test_backend_matches_reference_and_writes_slots(page_size, lengths):
  q_lens, kv_lens = lengths
  case, manager, backend, layer, batch, k, v = backend_case(q_lens, kv_lens, page_size=page_size)
  backend.init_forward_metadata(batch)
  out = layer(case.q, k, v, batch)
  torch.testing.assert_close(
    out.view_as(case.q).float(),
    reference(case.q, case.keys, case.values, q_lens),
    rtol=0.01,
    atol=0.01,
  )
  torch.testing.assert_close(backend.forward_metadata.block_table, case.table)
  keys, values = manager.get_kv_buffer(0)
  torch.testing.assert_close(keys[batch.out_cache_loc], k, rtol=0, atol=0)
  torch.testing.assert_close(values[batch.out_cache_loc], v, rtol=0, atol=0)


def test_backend_save_false_shared_prefix():
  case, manager, backend, layer, batch, k, v = backend_case([3, 7], [259, 263], shared=True)
  manager.set_kv_buffer(0, batch.out_cache_loc, k, v)
  backend.init_forward_metadata(batch)
  out = backend.forward(case.q, None, None, layer, batch, save_kv_cache=False)
  torch.testing.assert_close(
    out.view_as(case.q).float(),
    reference(case.q, case.keys, case.values, case.q_lens),
    rtol=0.01,
    atol=0.01,
  )


def test_backend_graph_survives_prefill_metadata_and_page_growth():
  case, manager, backend, layer, batch, k, v = backend_case([1, 1], [511, 511])
  manager.set_kv_buffer(0, batch.out_cache_loc, k, v)
  table = case.table.clone()
  lengths = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
  backend.init_cuda_graph_metadata(2, table, lengths, max_seq_len_k=2)
  backend.forward(case.q, None, None, layer, batch, save_kv_cache=False)
  torch.cuda.synchronize()
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    out = backend.forward(case.q, None, None, layer, batch, save_kv_cache=False)
  # Another batch may replace eager metadata between two decode replays.
  backend.init_forward_metadata(batch)
  lengths.copy_(torch.tensor([257, 511], dtype=torch.int32, device="cuda"))
  backend.update_cuda_graph_metadata(lengths, max_seq_len_k=511)
  graph.replay()
  torch.testing.assert_close(
    out.view_as(case.q).float(),
    reference(case.q, case.keys, case.values, [1, 1], [257, 511]),
    rtol=0.01,
    atol=0.01,
  )


def test_backend_validation_and_selection():
  from miniinfer.engine.model_runner import ModelRunner
  from miniinfer.layers.attention_backend import FlashAttentionV100Backend

  _, manager, _, _, _, _, _ = backend_case([1], [17])
  runner = ModelRunner.__new__(ModelRunner)
  runner.kv_cache_mgr = manager
  runner.init_attn_backend("flash_attn_v100")
  assert isinstance(runner.attn_backend, FlashAttentionV100Backend)
  manager.dtype = torch.bfloat16
  with pytest.raises(ValueError, match="FP16"):
    FlashAttentionV100Backend(manager)


def test_sampling_without_flashinfer(monkeypatch):
  import miniinfer.layers.sample as module

  monkeypatch.setattr(module, "sampling", None)
  sampler = module.Sampler()
  logits = torch.tensor(
    [[0.0, 2.0, 1.0, -1.0], [2.0, 0.0, 1.0, -1.0], [0.0, 1.0, 3.0, 2.0]], device="cuda"
  )
  out = sampler._sample_impl(
    logits,
    temperature=torch.tensor([0.0, 1.0, 1.0], device="cuda"),
    top_p=torch.tensor([1.0, 1.0, 0.01], device="cuda"),
    top_k=torch.tensor([0, 1, 0], device="cuda"),
    enable_top_k_top_p=True,
  )
  assert out.tolist() == [1, 0, 2]
  out = sampler._sample_impl(
    logits[1:],
    temperature=torch.ones(2, device="cuda"),
    top_p=torch.tensor([1.0, 0.01], device="cuda"),
    top_k=torch.tensor([1, 0], device="cuda"),
    enable_top_k_top_p=True,
    all_non_greedy=True,
  )
  assert out.tolist() == [0, 2]


@pytest.mark.parametrize(
  "options,match",
  [
    ({"dtype": torch.float32}, "float16"),
    ({"page_size": 0}, "multiple of 16"),
    ({"page_size": 17}, "multiple of 16"),
    ({"enable_chunked_prefill": True, "chunked_prefill_size": 128}, "at least one page"),
  ],
)
def test_backend_config_rejects_unsupported_options(options, match):
  from miniinfer.config.engine.config import EngineConfig

  with pytest.raises(ValueError, match=match):
    EngineConfig("unused-local-path", attention_backend="flash_attn_v100", **options)


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory):
  from tokenizers import Tokenizer
  from tokenizers.models import WordLevel
  from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

  path = tmp_path_factory.mktemp("v100-qwen2")
  torch.manual_seed(42)
  config = Qwen2Config(
    vocab_size=64,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=768,
    tie_word_embeddings=False,
    eos_token_id=2,
    pad_token_id=0,
    attn_implementation="eager",
  )
  hf = Qwen2ForCausalLM(config).half().eval().to("cuda")
  hf.save_pretrained(path)
  tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t1")),
    unk_token="t1",
    eos_token="t2",
    pad_token="t0",
  )
  tokenizer.save_pretrained(path)
  return str(path), hf


def hf_greedy(model, prompt, count):
  ids = torch.tensor([prompt], device="cuda")
  output = []
  with torch.inference_mode():
    for _ in range(count):
      token = model(ids, use_cache=False).logits[:, -1].argmax(-1)
      output.append(int(token))
      ids = torch.cat((ids, token[:, None]), dim=1)
  return output


def tiny_engine(path, monkeypatch, *, graph=False, overlap=False, chunked=False):
  import miniinfer.engine.llm_engine as module

  # A fixed, small test cache avoids reserving most of a shared 32 GiB GPU for
  # a randomly initialized model. All allocator/scheduler logic stays active.
  monkeypatch.setattr(module, "calc_max_total_tokens", lambda *a, **kw: 4096)
  return module.LLMEngine(
    path,
    attention_backend="flash_attn_v100",
    dtype=torch.float16,
    page_size=256,
    max_num_seqs=4,
    max_context_len=768,
    max_extend_len=1024,
    enforce_eager=not graph,
    enable_overlap=overlap,
    enable_prefix_cache=True,
    enable_chunked_prefill=chunked,
    chunked_prefill_size=256,
    debug=False,
    use_chat_template=False,
  )


@pytest.mark.parametrize("graph,overlap", [(False, False), (True, False), (True, True)])
def test_tiny_engine_generation_and_prefix(tiny_qwen, monkeypatch, graph, overlap):
  from miniinfer.utils.sampling_params import SamplingParams

  path, hf = tiny_qwen
  first = [3 + i % 61 for i in range(260)]
  second = first[:256] + [7, 13, 19, 23, 29]
  params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
  with tiny_engine(path, monkeypatch, graph=graph, overlap=overlap) as engine:
    result = engine.generate([first], params, use_tqdm=False)
    assert result[0]["token_ids"] == hf_greedy(hf, first, 4)
    # Unequal requests exercise batch padding in the graph path as well.
    prompts = [second, [8, 9, 10], [11, 12, 13, 14, 15]]
    ids = engine.add_requests(prompts, [params] * len(prompts))
    observed = {request: [] for request in ids}
    prefixes = []
    for _ in range(32):
      step = engine.step_overlap() if overlap else engine.step()
      prefixes.extend(step.prefill_prefix_lens.values())
      for out in step.outputs:
        observed[out.request_id].append(out.token_id)
      if engine.is_finished():
        break
    assert engine.is_finished()
    assert 256 in prefixes
    for request, prompt in zip(ids, prompts):
      assert observed[request] == hf_greedy(hf, prompt, 4)
    if graph:
      assert engine.model_runner.cuda_graph_runner.is_available()


def test_tiny_engine_chunked_mixed_matches_hf(tiny_qwen, monkeypatch):
  from miniinfer.utils.sampling_params import SamplingParams

  path, hf = tiny_qwen
  first = [3 + i % 61 for i in range(17)]
  second = [5 + i % 59 for i in range(300)]
  params_a = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True)
  params_b = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
  with tiny_engine(path, monkeypatch, chunked=True) as engine:
    request_a = engine.add_requests([first], [params_a])[0]
    initial = engine.step()
    observed = {request_a: [out.token_id for out in initial.outputs]}
    request_b = engine.add_requests([second], [params_b])[0]
    observed[request_b] = []
    mixed_seen = False
    for _ in range(64):
      step = engine.step()
      mixed_seen |= step.num_prefill_tokens > 0 and step.num_decode_tokens > 0
      for out in step.outputs:
        observed[out.request_id].append(out.token_id)
      if engine.is_finished():
        break
    assert engine.is_finished()
    assert mixed_seen, "test must actually schedule a mixed prefill/decode batch"
    assert observed[request_a] == hf_greedy(hf, first, 12)
    assert observed[request_b] == hf_greedy(hf, second, 4)
