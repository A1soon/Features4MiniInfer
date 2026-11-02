"""Small independent, per-token MoE oracle and synthetic HF checkpoint."""

import json

import torch
from safetensors.torch import save_file


def moe_reference_with_routes(hidden, expert_gate, expert_up, expert_down, topk_ids, topk_weights):
  flat = hidden.reshape(-1, hidden.shape[-1]).double()
  output = torch.zeros_like(flat)
  for token in range(len(flat)):
    for choice in range(topk_ids.shape[1]):
      expert = int(topk_ids[token, choice])
      gated = torch.nn.functional.silu(expert_gate[expert].double() @ flat[token])
      up = expert_up[expert].double() @ flat[token]
      output[token] += topk_weights[token, choice].double() * (
        expert_down[expert].double() @ (gated * up)
      )
  return output.reshape(hidden.shape).to(hidden.dtype)


def moe_reference(hidden, gate, expert_gate, expert_up, expert_down, top_k, normalize=True):
  probabilities = (hidden.reshape(-1, hidden.shape[-1]).double() @ gate.double().T).softmax(-1)
  weights, ids = probabilities.topk(top_k, dim=-1)
  if normalize:
    weights = weights / weights.sum(-1, keepdim=True)
  return moe_reference_with_routes(hidden, expert_gate, expert_up, expert_down, ids, weights)


def make_checkpoint(root, *, tied=False, dense_layer=False, dtype=torch.float32):
  config = dict(
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=24,
    max_position_embeddings=64,
    tie_word_embeddings=tied,
    mlp_only_layers=[1] if dense_layer else [],
  )
  generator = torch.Generator().manual_seed(2026)
  state = {}

  def add(name, shape):
    state[name] = (torch.randn(shape, generator=generator) * 0.1).to(dtype)

  add("model.embed_tokens.weight", (32, 16))
  add("model.norm.weight", (16,))
  if not tied:
    add("lm_head.weight", (32, 16))
  for layer in range(2):
    prefix = f"model.layers.{layer}"
    for norm in ("input_layernorm", "post_attention_layernorm"):
      add(f"{prefix}.{norm}.weight", (16,))
    for name, shape in (
      ("q_proj", (32, 16)),
      ("k_proj", (16, 16)),
      ("v_proj", (16, 16)),
      ("o_proj", (16, 32)),
      ("q_norm", (8,)),
      ("k_norm", (8,)),
    ):
      add(f"{prefix}.self_attn.{name}.weight", shape)
    if dense_layer and layer == 1:
      for name, shape in (("gate_proj", (32, 16)), ("up_proj", (32, 16)), ("down_proj", (16, 32))):
        add(f"{prefix}.mlp.{name}.weight", shape)
    else:
      add(f"{prefix}.mlp.gate.weight", (8, 16))
      for expert in range(8):
        for name, shape in (
          ("gate_proj", (24, 16)),
          ("up_proj", (24, 16)),
          ("down_proj", (16, 24)),
        ):
          add(f"{prefix}.mlp.experts.{expert}.{name}.weight", shape)
  names = sorted(state)
  mapping = {}
  for i in range(2):
    filename = f"model-{i + 1:05d}-of-00002.safetensors"
    shard = {name: state[name] for name in names[i::2]}
    save_file(shard, root / filename)
    mapping.update({name: filename for name in shard})
  (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
  return config, state
