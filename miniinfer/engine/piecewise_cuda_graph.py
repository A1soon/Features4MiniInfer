"""Bounded static-shape prefill/mixed CUDA Graph for the V100 paged backend."""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import torch

from miniinfer.engine.cuda_graph_runner import retain_moe_workspaces
from miniinfer.scheduler.scheduler_batch import ForwardBatch, ForwardMode
from miniinfer.utils import get_logger

logger = get_logger(__name__)


class PiecewiseCudaGraphRunner:
  """Capture fixed token buckets for at most a small number of V100 requests.

  Each graph contains ``max_sequences`` request segments and one padding segment.
  The padding segment writes only to pages reserved from the KV allocator; every
  rank receives identical buckets and query lengths so MoE collectives match.
  Unsupported sizes and batch layouts return ``None`` for eager fallback.
  """

  def __init__(self, model_runner: Any, buckets: Optional[list[int]] = None):
    self.model_runner = model_runner
    self.device = model_runner.device
    self.dtype = model_runner.config.dtype
    self.kv_cache_mgr = model_runner.kv_cache_mgr
    self.attn_backend = model_runner.attn_backend
    self.model = model_runner.model
    self.page_size = self.kv_cache_mgr.page_size
    self.max_context_len = model_runner.config.max_context_len
    if (
      self.attn_backend.type() != "flash_attn_v100"
      or not hasattr(self.attn_backend, "init_extend_cuda_graph_metadata")
    ):
      raise RuntimeError("prefill CUDA Graph currently supports the V100 paged backend only")

    requested_sequences = int(os.getenv("MINIINFER_PREFILL_CG_MAX_SEQS", "2"))
    self.max_sequences = min(model_runner.config.max_num_seqs, requested_sequences)
    if self.max_sequences <= 0:
      raise ValueError("MINIINFER_PREFILL_CG_MAX_SEQS must be positive")
    if self.max_sequences > self.page_size:
      raise ValueError("graph dummy rows exceed the reserved padding page capacity")
    max_tokens = min(
      model_runner.config.max_extend_len,
      int(os.getenv("MINIINFER_PREFILL_CG_MAX_TOKENS", "512")),
    )
    self.buckets = buckets or self.capture_buckets(self.page_size, max_tokens)
    if not self.buckets:
      raise ValueError("no prefill CUDA Graph token buckets fit the configured limit")
    if any(bucket <= self.max_sequences for bucket in self.buckets):
      raise ValueError("each token bucket must leave room for one query per graph sequence")
    self.static_batch_size = self.max_sequences + 1
    self.max_bucket = max(self.buckets)
    self.max_blocks_per_seq = math.ceil(max(self.max_context_len, self.max_bucket) / self.page_size)
    self.graphs: dict[int, dict[str, Any]] = {}
    self._captured = False

    main_page_count = math.ceil(self.max_bucket / self.page_size)
    scratch_page_count = main_page_count
    allocator = self.kv_cache_mgr.token_allocator
    free_pages = getattr(allocator, "free_pages", None)
    if free_pages is None or free_pages.numel() < scratch_page_count:
      raise RuntimeError("insufficient free KV pages for prefill CUDA Graph padding")
    scratch_pages = free_pages[:scratch_page_count].clone()
    allocator.free_pages = free_pages[scratch_page_count:]
    self.main_padding_pages = scratch_pages[:main_page_count]

  @staticmethod
  def capture_buckets(page_size: int, max_tokens: int) -> list[int]:
    if (
      not isinstance(page_size, int)
      or isinstance(page_size, bool)
      or page_size <= 0
      or not isinstance(max_tokens, int)
      or isinstance(max_tokens, bool)
      or max_tokens <= 0
    ):
      raise ValueError("page_size and max_tokens must be positive integers")
    first = min(page_size, max_tokens)
    buckets = []
    bucket = first
    while bucket < max_tokens:
      buckets.append(bucket)
      bucket *= 2
    buckets.append(max_tokens)
    return sorted(set(buckets))

  @staticmethod
  def padded_query_lengths(
    query_lengths: list[int], bucket: int, max_sequences: int
  ) -> Optional[list[int]]:
    if (
      not query_lengths
      or len(query_lengths) > max_sequences
      or any(length <= 0 for length in query_lengths)
      or bucket <= 0
    ):
      return None
    padding = bucket - sum(query_lengths)
    dummy_count = max_sequences + 1 - len(query_lengths)
    if padding < dummy_count:
      return None
    return query_lengths + [padding - dummy_count + 1] + [1] * (dummy_count - 1)

  def is_available(self) -> bool:
    return self._captured and bool(self.graphs)

  def _trace(self, phase: str, bucket: int) -> None:
    if os.getenv("MINIINFER_PREFILL_CG_TRACE") == "1":
      torch.cuda.synchronize(self.device)
      print(f"PREFILL_CG device={self.device} bucket={bucket} {phase}", flush=True)

  def _cu_seqlens(self, query_lengths: list[int]) -> torch.Tensor:
    values = [0]
    for length in query_lengths:
      values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=self.device)

  def _padding_slots(self, dummy_index: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if dummy_index == 0:
      pages = self.main_padding_pages
      slots = (pages[:, None] * self.page_size + torch.arange(
        self.page_size, dtype=torch.int64, device=pages.device
      )).reshape(-1)[:length]
      used_pages = math.ceil(length / self.page_size)
      return slots, pages[:used_pages]
    return (
      torch.tensor([dummy_index - 1], dtype=torch.int64, device=self.device),
      torch.tensor([0], dtype=torch.int32, device=self.device),
    )

  def _make_capture_batch(self, bucket: int, layout: list[int]) -> tuple[ForwardBatch, dict]:
    device = self.device
    input_ids = torch.zeros(bucket, dtype=torch.int64, device=device)
    positions = torch.zeros(bucket, dtype=torch.int64, device=device)
    out_cache_loc = torch.empty(bucket, dtype=torch.int64, device=device)
    req_pool_indices = torch.zeros(self.static_batch_size, dtype=torch.int64, device=device)
    seq_lens = torch.tensor(layout, dtype=torch.int64, device=device)
    seq_lens_int32 = seq_lens.to(torch.int32)
    extend_seq_lens = seq_lens.clone()
    extend_prefix_lens = torch.zeros(self.static_batch_size, dtype=torch.int64, device=device)
    cu_seqlens_q = self._cu_seqlens(layout)
    last_token_indices = cu_seqlens_q[1:] - 1
    block_table = torch.zeros(
      (self.static_batch_size, self.max_blocks_per_seq), dtype=torch.int32, device=device
    )

    token_offset = 0
    for row in range(self.max_sequences):
      block_table[row, 0] = 0
      out_cache_loc[token_offset] = row
      token_offset += 1
    dummy_slots, dummy_pages = self._padding_slots(0, layout[-1])
    block_table[-1, : dummy_pages.numel()] = dummy_pages.to(torch.int32)
    out_cache_loc[token_offset:] = dummy_slots
    positions[-layout[-1] :] = torch.arange(layout[-1], dtype=torch.int64, device=device)

    batch = ForwardBatch(
      forward_mode=ForwardMode.MIXED,
      batch_size=self.static_batch_size,
      input_ids=input_ids,
      positions=positions,
      req_pool_indices=req_pool_indices,
      out_cache_loc=out_cache_loc,
      attn_backend=self.attn_backend,
      seq_lens=seq_lens,
      seq_lens_int32=seq_lens_int32,
      seq_lens_cpu=torch.tensor(layout),
      max_seq_len=max(layout),
      extend_seq_lens=extend_seq_lens,
      extend_prefix_lens=extend_prefix_lens,
      extend_seq_lens_cpu=layout,
      extend_prefix_lens_cpu=[0] * self.static_batch_size,
      last_token_indices=last_token_indices,
    )
    self.attn_backend.init_extend_cuda_graph_metadata(
      bucket=bucket,
      cu_seqlens_q=cu_seqlens_q,
      cache_seqlens=seq_lens_int32,
      block_table=block_table,
      max_seqlen_q=bucket,
    )
    buffers = {
      "input_ids": input_ids,
      "positions": positions,
      "out_cache_loc": out_cache_loc,
      "req_pool_indices": req_pool_indices,
      "seq_lens": seq_lens,
      "seq_lens_int32": seq_lens_int32,
      "extend_seq_lens": extend_seq_lens,
      "extend_prefix_lens": extend_prefix_lens,
      "cu_seqlens_q": cu_seqlens_q,
      "last_token_indices": last_token_indices,
      "block_table": block_table,
      "batch": batch,
    }
    return batch, buffers

  def capture(self) -> None:
    for bucket in self.buckets:
      try:
        layout = self.padded_query_lengths([1] * self.max_sequences, bucket, self.max_sequences)
        if layout is None:
          continue
        self._trace("capture_begin", bucket)
        batch, buffers = self._make_capture_batch(bucket, layout)
        output_logits = torch.empty(
          (self.static_batch_size, self.model_runner.model_config.vocab_size),
          dtype=self.dtype,
          device=self.device,
        )
        torch.cuda.synchronize()
        with torch.inference_mode():
          self.model.forward(
            buffers["input_ids"], buffers["positions"], batch, return_hidden_states=False
          )
        torch.cuda.synchronize()
        buffers["moe_workspaces"] = retain_moe_workspaces(self.model)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
          output = self.model.forward(
            buffers["input_ids"], buffers["positions"], batch, return_hidden_states=False
          )
          output_logits.copy_(output.logits)
        buffers["graph"] = graph
        buffers["output_logits"] = output_logits
        self.graphs[bucket] = buffers
        self._trace("capture_end", bucket)
      except Exception as exc:
        if getattr(self.model_runner.config, "expert_parallel_size", 1) > 1:
          raise
        logger.warning(f"PiecewiseCudaGraph: capture bucket={bucket} failed: {exc!r}; skip")
    self._captured = True
    logger.info(f"PiecewiseCudaGraph captured V100 buckets={list(self.graphs)}")

  def _bucket_layout(self, query_lengths: list[int]) -> tuple[Optional[int], Optional[list[int]]]:
    for bucket in self.buckets:
      layout = self.padded_query_lengths(query_lengths, bucket, self.max_sequences)
      if layout is not None and bucket in self.graphs:
        return bucket, layout
    return None, None

  def replay(self, forward_batch) -> Any:
    if not forward_batch.forward_mode.is_extend():
      return None
    if forward_batch.batch_size > self.max_sequences:
      return None
    if forward_batch.seq_lens_cpu is None or not forward_batch.extend_seq_lens_cpu:
      return None

    from miniinfer.models.base import BaseModelOutput

    query_lengths = [int(length) for length in forward_batch.extend_seq_lens_cpu]
    bucket, layout = self._bucket_layout(query_lengths)
    if bucket is None:
      return None
    buffers = self.graphs[bucket]
    actual_tokens = int(forward_batch.input_ids.shape[0])
    if sum(query_lengths) != actual_tokens:
      return None

    actual_batch = forward_batch.batch_size
    seq_lens_cpu = forward_batch.seq_lens_cpu
    if hasattr(seq_lens_cpu, "tolist"):
      seq_lens_cpu = seq_lens_cpu.tolist()
    actual_lengths = [int(length) for length in seq_lens_cpu]
    prefixes = (
      [int(length) for length in forward_batch.extend_prefix_lens_cpu]
      if forward_batch.extend_prefix_lens_cpu is not None
      else [seq_len - q_len for seq_len, q_len in zip(actual_lengths, query_lengths)]
    )
    if len(actual_lengths) != actual_batch or len(prefixes) != actual_batch:
      return None
    eager_reference = None
    eager_top1 = None
    self._trace(f"replay_begin queries={query_lengths} lengths={actual_lengths}", bucket)
    if os.getenv("MINIINFER_PREFILL_CG_VERIFY") == "1":
      self._trace("eager_begin", bucket)
      self.attn_backend.init_forward_metadata(forward_batch)
      with torch.inference_mode():
        eager_output = self.model.forward(
          forward_batch.input_ids,
          forward_batch.positions,
          forward_batch,
          return_hidden_states=False,
        )
      torch.cuda.synchronize()
      eager_reference = eager_output.logits.detach().float().cpu().clone()
      eager_top1 = eager_reference.argmax(dim=-1)
      self._trace("eager_end", bucket)

    max_actual_len = max(actual_lengths)
    slots = self.kv_cache_mgr.get_page_table(forward_batch, max_actual_len)
    actual_blocks = (slots[:, :: self.page_size] // self.page_size).to(torch.int32)

    buffers["input_ids"].zero_()
    buffers["positions"].zero_()
    buffers["out_cache_loc"].zero_()
    buffers["req_pool_indices"].zero_()
    buffers["seq_lens"].zero_()
    buffers["extend_seq_lens"].copy_(torch.tensor(layout, dtype=torch.int64, device=self.device))
    buffers["extend_prefix_lens"].zero_()
    buffers["block_table"].zero_()

    buffers["input_ids"][:actual_tokens].copy_(forward_batch.input_ids)
    buffers["positions"][:actual_tokens].copy_(forward_batch.positions)
    buffers["out_cache_loc"][:actual_tokens].copy_(forward_batch.out_cache_loc)
    buffers["req_pool_indices"][:actual_batch].copy_(forward_batch.req_pool_indices)
    buffers["seq_lens"][:actual_batch].copy_(torch.tensor(actual_lengths, dtype=torch.int64, device=self.device))
    buffers["extend_prefix_lens"][:actual_batch].copy_(
      torch.tensor(prefixes, dtype=torch.int64, device=self.device)
    )

    for row, seq_len in enumerate(actual_lengths):
      pages = math.ceil(seq_len / self.page_size)
      if pages > self.max_blocks_per_seq:
        return None
      buffers["block_table"][row, :pages].copy_(actual_blocks[row, :pages])

    token_offset = actual_tokens
    row_offset = actual_batch
    for dummy_index, q_len in enumerate(layout[actual_batch:]):
      dummy_slots, dummy_pages = self._padding_slots(dummy_index, q_len)
      buffers["input_ids"][token_offset : token_offset + q_len].zero_()
      buffers["positions"][token_offset : token_offset + q_len].copy_(
        torch.arange(q_len, dtype=torch.int64, device=self.device)
      )
      buffers["out_cache_loc"][token_offset : token_offset + q_len].copy_(dummy_slots)
      buffers["seq_lens"][row_offset] = q_len
      buffers["block_table"][row_offset, : dummy_pages.numel()].copy_(dummy_pages.to(torch.int32))
      token_offset += q_len
      row_offset += 1

    cu_seqlens_q = self._cu_seqlens(layout)
    buffers["cu_seqlens_q"].copy_(cu_seqlens_q)
    buffers["seq_lens_int32"].copy_(buffers["seq_lens"].to(torch.int32))
    buffers["last_token_indices"].copy_(cu_seqlens_q[1:] - 1)
    self.attn_backend.update_extend_cuda_graph_metadata(bucket)
    self._trace("metadata_ready", bucket)
    buffers["graph"].replay()
    self._trace("replay_end", bucket)
    logits = buffers["output_logits"][:actual_batch].clone()
    self._trace("logits_clone_end", bucket)
    if eager_reference is not None:
      graph_logits_cpu = logits.detach().float().cpu()
      self._trace("logits_cpu_snapshot_end", bucket)
      if os.getenv("MINIINFER_PREFILL_CG_PADDED_VERIFY") == "1":
        self.attn_backend.update_extend_cuda_graph_metadata(bucket)
        with torch.inference_mode():
          padded_output = self.model.forward(
            buffers["input_ids"], buffers["positions"], buffers["batch"],
            return_hidden_states=False,
          )
        padded_reference = padded_output.logits[:actual_batch].detach().float().cpu().clone()
        print(
          f"PREFILL_CG device={self.device} bucket={bucket} "
          f"graph_vs_padded_eager={(graph_logits_cpu - padded_reference).abs().max().item()} "
          f"padded_vs_unpadded_eager={(padded_reference - eager_reference).abs().max().item()} "
          f"top1_mismatches={(padded_reference.argmax(-1) != eager_top1).sum().item()}",
          flush=True,
        )
        torch.testing.assert_close(graph_logits_cpu, padded_reference, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(padded_reference.argmax(-1), eager_top1, rtol=0, atol=0)
      try:
        if os.getenv("MINIINFER_PREFILL_CG_PADDED_VERIFY") != "1":
          torch.testing.assert_close(graph_logits_cpu, eager_reference, rtol=1e-3, atol=1e-3)
      except AssertionError as exc:
        print(f"PREFILL_CG device={self.device} bucket={bucket} parity_failed: {exc}", flush=True)
        raise
      logger.info(
        f"PiecewiseCudaGraph verify bucket={bucket} batch={actual_batch} "
        f"max_abs_error={(graph_logits_cpu - eager_reference).abs().max().item()}"
      )
      self._trace("parity_pass", bucket)
    self._trace("return", bucket)
    return BaseModelOutput(
      logits=logits,
      last_hidden_state=None,
      hidden_states=None,
    )
