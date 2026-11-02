"""Stream standard HF Qwen3-MoE weights directly into one rank's parameters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass
class MoELoadReport:
    loaded_tensors: int = 0
    skipped_tensors: int = 0
    source_bytes: int = 0
    destination_bytes: int = 0
    largest_source_tensor_bytes: int = 0


@dataclass
class _Target:
    destination: torch.Tensor
    source_shape: tuple[int, ...]
    axis: int | None = None
    start: int = 0
    end: int = 0


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint index key: {key}")
        result[key] = value
    return result


def _checkpoint_map(root: Path) -> dict[str, str]:
    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text(), object_pairs_hook=_unique_object)["weight_map"]
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Checkpoint weight_map must be a nonempty object")
        for filename in set(mapping.values()):
            if (Path(filename).name != filename or not filename.endswith(".safetensors")
                    or not (root / filename).is_file()):
                raise ValueError(f"Missing or invalid checkpoint shard: {filename}")
        return mapping
    mapping = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():
                if name in mapping:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                mapping[name] = path.name
    if not mapping:
        raise ValueError(f"No safetensors checkpoint in {root}")
    return mapping


def _targets(model) -> dict[str, _Target]:
    core = model.qwen3_moe
    config = model.config
    device, dtype = model.device, model.precision
    targets = {}

    def weight(name, module, shape):
        if module.weight is None:
            module.weight = torch.empty(shape, device=device, dtype=dtype)
        targets[name] = _Target(module.weight, tuple(shape))

    weight("model.embed_tokens.weight", core.embedding, (config.vocab_size, config.hidden_size))
    weight("model.norm.weight", core.norm, (config.hidden_size,))
    weight("lm_head.weight", model.lm_head, (config.vocab_size, config.hidden_size))
    for i, layer in enumerate(core.layers):
        prefix = f"model.layers.{i}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            weight(f"{prefix}.{norm}.weight", getattr(layer, norm), (config.hidden_size,))
        attention = layer.self_attn
        offset = 0
        for projection, size in (("q_proj", attention.q_size), ("k_proj", attention.kv_size),
                                 ("v_proj", attention.kv_size)):
            targets[f"{prefix}.self_attn.{projection}.weight"] = _Target(
                attention.qkv_proj.weight[offset:offset + size], (size, config.hidden_size),
            )
            offset += size
        targets[f"{prefix}.self_attn.o_proj.weight"] = _Target(
            attention.o_proj.weight, (config.hidden_size, config.num_attention_heads * config.head_dim),
        )
        for norm in ("q_norm", "k_norm"):
            weight(f"{prefix}.self_attn.{norm}.weight", getattr(attention, norm), (config.head_dim,))
        if not layer.is_moe:
            mlp = layer.mlp
            size = config.intermediate_size
            for j, projection in enumerate(("gate_proj", "up_proj")):
                targets[f"{prefix}.mlp.{projection}.weight"] = _Target(
                    mlp.gate_up_proj.weight[j * size:(j + 1) * size], (size, config.hidden_size),
                )
            targets[f"{prefix}.mlp.down_proj.weight"] = _Target(
                mlp.down_proj.weight, (config.hidden_size, size),
            )
            continue
        moe = layer.mlp.experts
        experts = moe.experts
        weight(f"{prefix}.mlp.gate.weight", moe.gate, (config.num_experts, config.hidden_size))
        local_size = experts.intermediate_size_local
        start = moe.parallel.tp_rank * local_size
        for expert in range(experts.local_expert_start, experts.local_expert_end):
            local = expert - experts.local_expert_start
            for j, projection in enumerate(("gate_proj", "up_proj")):
                targets[f"{prefix}.mlp.experts.{expert}.{projection}.weight"] = _Target(
                    experts.w13_weight[local, j * local_size:(j + 1) * local_size],
                    (config.moe_intermediate_size, config.hidden_size), 0, start, start + local_size,
                )
            targets[f"{prefix}.mlp.experts.{expert}.down_proj.weight"] = _Target(
                experts.w2_weight[local], (config.hidden_size, config.moe_intermediate_size),
                1, start, start + local_size,
            )
    return targets


def _remote_expert(name, model):
    match = re.fullmatch(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name,
    )
    if not match:
        return False
    layer, expert = map(int, match.groups()[:2])
    core = model.qwen3_moe
    if layer >= len(core.layers) or not core.layers[layer].is_moe or expert >= model.config.num_experts:
        return False
    begin, end = core.parallel.expert_range(model.config.num_experts)
    return not begin <= expert < end


@torch.no_grad()
def load_qwen3_moe_weights(model, model_path: str | Path) -> MoELoadReport:
    """Load local experts and replicated dense weights, without a full state_dict.

    Supports unquantized HF per-expert gate/up/down tensors. Only selected
    tensors (or TP slices) are materialized on CPU, one at a time. No full-shard
    prefetch is performed by each rank. Missing/unexpected keys fail explicitly.
    """
    root = Path(model_path)
    mapping = _checkpoint_map(root)
    targets = _targets(model)
    tied = model.config.tie_word_embeddings
    if tied and "lm_head.weight" not in mapping:
        del targets["lm_head.weight"]
    missing = set(targets) - mapping.keys()
    unexpected = {name for name in mapping if name not in targets and not _remote_expert(name, model)}
    if missing or unexpected:
        raise ValueError(f"Checkpoint keys mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    report = MoELoadReport(skipped_tensors=len(mapping) - len(targets))
    for filename in sorted({mapping[name] for name in targets}):
        with safe_open(root / filename, framework="pt", device="cpu") as source:
            for name in sorted(name for name in targets if mapping[name] == filename):
                target = targets[name]
                sliced = source.get_slice(name)
                if tuple(sliced.get_shape()) != target.source_shape:
                    raise ValueError(f"Wrong checkpoint shape for {name}: {sliced.get_shape()}")
                if target.axis is None:
                    tensor = source.get_tensor(name)
                else:
                    selection = [slice(None)] * len(target.source_shape)
                    selection[target.axis] = slice(target.start, target.end)
                    tensor = sliced[tuple(selection)]
                if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-floating or non-finite weight: {name}")
                converted = tensor.to(device=target.destination.device, dtype=target.destination.dtype)
                if not torch.isfinite(converted).all():
                    raise ValueError(f"Weight overflows destination dtype: {name}")
                target.destination.copy_(converted)
                size = tensor.numel() * tensor.element_size()
                report.loaded_tensors += 1
                report.source_bytes += size
                report.destination_bytes += converted.numel() * converted.element_size()
                report.largest_source_tensor_bytes = max(report.largest_source_tensor_bytes, size)
                del tensor, converted
    if tied:
        if "lm_head.weight" in mapping and not torch.equal(model.lm_head.weight, model.qwen3_moe.embedding.weight):
            raise ValueError("tie_word_embeddings requires matching embedding and lm_head weights")
        model.lm_head.weight = model.qwen3_moe.embedding.weight
    return report
