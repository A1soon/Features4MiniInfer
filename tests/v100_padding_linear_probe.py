"""Measure padding sensitivity using only real embedding rows and first-layer QKV."""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def attention_padding_probe(read, config, inputs, padding, hidden):
  from miniinfer.kvcache.kv_cache_manager import KVCacheManager
  from miniinfer.layers.attention_backend import FlashAttentionV100Backend
  from miniinfer.layers.layernorm import RMSNorm
  from miniinfer.models.fused_qwen3 import Qwen3Attention
  from miniinfer.scheduler.scheduler_batch import ForwardBatch, ForwardMode

  heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
  head_dim = config.get("head_dim", config["hidden_size"] // heads)
  attention = Qwen3Attention(
    config["hidden_size"], heads, kv_heads, head_dim, layer_id=0,
    max_position_embeddings=512, rope_theta=config["rope_theta"],
    rms_norm_eps=config["rms_norm_eps"],
  ).to(device="cuda:0", dtype=torch.float16)
  with torch.no_grad():
    attention.qkv_proj.weight.copy_(torch.cat([
      read(f"model.layers.0.self_attn.{projection}.weight")
      for projection in ("q_proj", "k_proj", "v_proj")
    ]))
    attention.o_proj.weight.copy_(read("model.layers.0.self_attn.o_proj.weight"))
  attention.q_norm.weight = read("model.layers.0.self_attn.q_norm.weight")
  attention.k_norm.weight = read("model.layers.0.self_attn.k_norm.weight")
  post_norm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
  post_norm.weight = read("model.layers.0.post_attention_layernorm.weight")
  router = read("model.layers.0.mlp.gate.weight")
  manager = KVCacheManager(
    size=2048, max_requests=3, max_context_len=512, num_layers=1, num_heads=kv_heads,
    head_dim=head_dim, dtype=torch.float16, device="cuda:0", page_size=256,
    max_extend_tokens=512,
  )
  backend = FlashAttentionV100Backend(manager)
  snapshots = {}
  current = None

  def hook(name, rows):
    def collect(module, args, output):
      tensors = output if isinstance(output, tuple) else (output,)
      snapshots[current][name] = tuple(tensor[:rows].float().cpu().clone() for tensor in tensors)
    return collect

  handles = []
  for name, rows in (("qkv_proj", 16), ("q_norm", 16 * heads),
                     ("k_norm", 16 * kv_heads), ("rope", 16),
                     ("attn", 16), ("o_proj", 16)):
    handles.append(getattr(attention, name).register_forward_hook(hook(name, rows)))
  try:
    for tokens in (16, 256, 512):
      current = tokens
      snapshots[tokens] = {}
      q_lens = [16] if tokens == 16 else [16, tokens - 17, 1]
      slots = [torch.arange(256, 272, device="cuda:0")]
      if tokens > 16:
        slots += [torch.arange(512, 512 + tokens - 17, device="cuda:0"),
                  torch.tensor([0], device="cuda:0")]
      for row, locations in enumerate(slots):
        manager.request_pool.req_to_token[row, :locations.numel()] = locations
      lengths = torch.tensor(q_lens, device="cuda:0")
      batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND, batch_size=len(q_lens),
        input_ids=torch.zeros(tokens, dtype=torch.int64, device="cuda:0"),
        positions=torch.cat([torch.arange(length, device="cuda:0") for length in q_lens]),
        req_pool_indices=torch.arange(len(q_lens), device="cuda:0"),
        out_cache_loc=torch.cat(slots), seq_lens=lengths,
        seq_lens_int32=lengths.int(), seq_lens_cpu=torch.tensor(q_lens),
        max_seq_len=max(q_lens), extend_seq_lens_cpu=q_lens,
        extend_prefix_lens_cpu=[0] * len(q_lens), attn_backend=backend,
      )
      backend.init_forward_metadata(batch)
      if tokens > 16:
        backend.forward_metadata.max_seqlen_q = tokens
        backend.forward_metadata.max_seqlen_k = 512
      if tokens == 16:
        x = hidden
      else:
        norm = read("model.layers.0.input_layernorm.weight")
        pad_float = padding.float()
        pad_hidden = norm * (pad_float * torch.rsqrt(
          pad_float.square().mean(-1, keepdim=True) + config["rms_norm_eps"]
        )).half()
        x = torch.cat([hidden, pad_hidden.expand(tokens - 16, -1)])
      with torch.inference_mode():
        output = attention(batch.positions, x, batch)
        residual = inputs if tokens == 16 else torch.cat([
          inputs, padding.expand(tokens - 16, -1)
        ])
        post_hidden, _ = post_norm(output, residual)
        router_logits = F.linear(post_hidden, router)
        ids = router_logits.float().softmax(-1).topk(config["num_experts_per_tok"], -1).indices
        for name, tensor in (("post_attention_norm", post_hidden),
                             ("router_logits", router_logits), ("topk_ids", ids)):
          snapshots[tokens][name] = (tensor[:16].float().cpu().clone(),)
      torch.cuda.synchronize()
  finally:
    for handle in handles:
      handle.remove()
  return [{
    "tokens": tokens,
    "max_abs_errors_by_stage": {
      name: max((a - b).abs().max().item() for a, b in zip(values, snapshots[16][name]))
      for name, values in snapshots[tokens].items()
    },
    "different_topk_rows": (snapshots[tokens]["topk_ids"][0]
                           != snapshots[16]["topk_ids"][0]).any(-1).sum().item(),
  } for tokens in (256, 512)]


def main():
  root = Path(os.environ.get("MINIINFER_MOE_MODEL_PATH", "/model"))
  mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
  config = json.loads((root / "config.json").read_text())

  def read(name, rows=None):
    with safe_open(root / mapping[name], framework="pt", device="cpu") as shard:
      tensor = shard.get_tensor(name) if rows is None else shard.get_slice(name)[rows]
      return tensor.to(device="cuda:0", dtype=torch.float16)

  inputs = read("model.embed_tokens.weight", slice(1000, 1016))
  padding = read("model.embed_tokens.weight", slice(0, 1))
  norm = read("model.layers.0.input_layernorm.weight")
  weight = torch.cat([
    read(f"model.layers.0.self_attn.{projection}.weight")
    for projection in ("q_proj", "k_proj", "v_proj")
  ])

  def normalize(x):
    fp32 = x.float()
    return norm * (fp32 * torch.rsqrt(
      fp32.square().mean(-1, keepdim=True) + config["rms_norm_eps"]
    )).half()

  hidden = normalize(inputs)
  records = []
  with torch.inference_mode():
    for reduced_precision in (True, False):
      torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced_precision
      eager = F.linear(hidden, weight).float()
      for tokens in (256, 512):
        padded = normalize(torch.cat([inputs, padding.expand(tokens - len(inputs), -1)]))
        torch.testing.assert_close(padded[:len(inputs)], hidden, rtol=0, atol=0)
        padded_projection = F.linear(padded, weight)[:len(inputs)].float()
        difference = (eager - padded_projection).abs()
        records.append({
          "dtype": "float16",
          "reduced_precision_reduction": reduced_precision,
          "tokens": tokens,
          "max_abs_error": difference.max().item(),
          "different_elements": (difference != 0).sum().item(),
          "numel": difference.numel(),
        })
    eager_fp32 = F.linear(hidden.float(), weight.float())
    padded_fp32 = torch.cat([hidden, normalize(padding).expand(240, -1)]).float()
    fp32_difference = (eager_fp32 - F.linear(padded_fp32, weight.float())[:16]).abs()
    records.append({"dtype": "float32", "tokens": 256,
                    "max_abs_error": fp32_difference.max().item()})
  torch.cuda.synchronize()
  attention_records = attention_padding_probe(read, config, inputs, padding, hidden)
  print(json.dumps({"status": "DIAGNOSTIC_COMPLETE", "checks": records,
                    "attention": attention_records,
                    "loaded": "16 embedding rows, one padding row, layer0 attention/norm only"}),
        flush=True)


if __name__ == "__main__":
  main()
