"""Synthetic safetensors verify real Qwen3-MoE destinations and selective reads."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from miniinfer.config.model.qwen3_moe import Qwen3MoeConfig
from miniinfer.layers.fused_moe import MoEParallelConfig
from miniinfer.loader import moe as loader
from miniinfer.models import Qwen3MoeForCausalLM
from tests.moe_reference import make_checkpoint


def model_for(config, parallel=None, dtype=torch.float32):
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(**config), device="cpu", precision=dtype, parallel=parallel)


@pytest.mark.parametrize("tp,ep,rank", [(1, 1, 0), (1, 4, 2), (2, 2, 3), (4, 1, 2)])
@pytest.mark.parametrize("tied,dense", [(False, False), (True, False), (False, True)])
def test_selective_stream_load(tmp_path, monkeypatch, tp, ep, rank, tied, dense):
    config, state = make_checkpoint(tmp_path, tied=tied, dense_layer=dense, dtype=torch.bfloat16)
    parallel = MoEParallelConfig(tp_size=tp, tp_rank=rank % tp, ep_size=ep, ep_rank=rank // tp)
    model = model_for(config, parallel, torch.float16)
    reads = []
    original = loader.safe_open

    class Spy:
        def __init__(self, *args, **kwargs):
            self.source = original(*args, **kwargs)
        def __enter__(self):
            self.source.__enter__()
            return self
        def __exit__(self, *args):
            return self.source.__exit__(*args)
        def get_slice(self, name):
            reads.append(name)
            return self.source.get_slice(name)
        def get_tensor(self, name):
            return self.source.get_tensor(name)

    monkeypatch.setattr(loader, "safe_open", Spy)
    report = loader.load_qwen3_moe_weights(model, tmp_path)
    begin, end = parallel.expert_range(8)
    assert all(begin <= int(name.split(".experts.")[1].split(".")[0]) < end
               for name in reads if ".experts." in name)
    assert len(reads) == report.loaded_tensors
    assert report.skipped_tensors == (2 - int(dense)) * (8 - (end - begin)) * 3
    assert report.source_bytes == report.destination_bytes
    assert report.largest_source_tensor_bytes < report.source_bytes
    torch.testing.assert_close(model.qwen3_moe.embedding.weight, state["model.embed_tokens.weight"].half())
    if tied:
        assert model.lm_head.weight is model.qwen3_moe.embedding.weight
    else:
        torch.testing.assert_close(model.lm_head.weight, state["lm_head.weight"].half())
    for i, layer in enumerate(model.qwen3_moe.layers):
        prefix = f"model.layers.{i}"
        expected_qkv = torch.cat([state[f"{prefix}.self_attn.{p}_proj.weight"] for p in ("q", "k", "v")])
        torch.testing.assert_close(layer.self_attn.qkv_proj.weight, expected_qkv.half())
        torch.testing.assert_close(layer.self_attn.q_norm.weight, state[f"{prefix}.self_attn.q_norm.weight"].half())
        if layer.is_moe:
            expert = layer.mlp.experts.experts
            size = 24 // tp
            start = parallel.tp_rank * size
            for global_id in range(begin, end):
                name = f"{prefix}.mlp.experts.{global_id}"
                expected = torch.cat([state[f"{name}.{p}_proj.weight"][start:start + size] for p in ("gate", "up")])
                torch.testing.assert_close(expert.w13_weight[global_id - begin], expected.half())
                torch.testing.assert_close(expert.w2_weight[global_id - begin], state[f"{name}.down_proj.weight"][:, start:start + size].half())
        else:
            expected = torch.cat([state[f"{prefix}.mlp.{p}_proj.weight"] for p in ("gate", "up")])
            torch.testing.assert_close(layer.mlp.gate_up_proj.weight, expected.half())


@pytest.mark.parametrize("fault", ["missing", "unexpected", "bad_shape", "nan", "overflow", "missing_shard", "path", "duplicate_index"])
def test_bad_checkpoint_fails_explicitly(tmp_path, fault):
    config, _ = make_checkpoint(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    shard = tmp_path / index["weight_map"][name]
    if fault == "missing":
        del index["weight_map"][name]
    elif fault == "unexpected":
        index["weight_map"]["model.layers.0.mlp.experts.999.gate_proj.weight"] = shard.name
    elif fault in ("bad_shape", "nan", "overflow"):
        state = load_file(shard)
        state[name] = torch.zeros(1, 16) if fault == "bad_shape" else torch.full((24, 16), float("nan") if fault == "nan" else 1e9)
        save_file(state, shard)
    elif fault == "missing_shard":
        index["weight_map"][name] = "missing.safetensors"
    elif fault == "path":
        index["weight_map"][name] = "../outside.safetensors"
    else:
        index_path.write_text('{"weight_map":{},"weight_map":{}}')
    if fault != "duplicate_index":
        index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        loader.load_qwen3_moe_weights(model_for(config, dtype=torch.float16), tmp_path)


def test_unindexed_checkpoint_and_legacy_copy_parity(tmp_path):
    config, state = make_checkpoint(tmp_path)
    (tmp_path / "model.safetensors.index.json").unlink()
    model = model_for(config)
    loader.load_qwen3_moe_weights(model, tmp_path)
    from miniinfer.loader.weight import _merge_state_dict

    legacy = model_for(config)
    legacy.load_weights(_merge_state_dict(dict(state)))
    for left, right in zip(model.parameters(), legacy.parameters(), strict=True):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(model.qwen3_moe.embedding.weight, legacy.qwen3_moe.embedding.weight)
