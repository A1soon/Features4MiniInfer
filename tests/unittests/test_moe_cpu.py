"""MoE arithmetic and rank layout checks requiring neither CUDA nor Triton."""

import os
import subprocess
import sys

import pytest
import torch

from miniinfer.layers.fused_moe import FusedMoE, MoEParallelConfig
from tests.moe_reference import moe_reference


@pytest.mark.parametrize("tokens,top_k,normalize", [(1, 1, True), (7, 2, True), (3, 8, True),
                                                    (7, 2, False), (0, 2, True)])
@pytest.mark.parametrize("shape3d", [False, True])
def test_moe_matches_independent_reference(tokens, top_k, normalize, shape3d):
    torch.manual_seed(47)
    layer = FusedMoE(8, top_k, 16, 24, norm_topk_prob=normalize, dtype=torch.float32)
    gate, up, down = [torch.randn(8, *shape) * 0.1 for shape in [(24, 16), (24, 16), (16, 24)]]
    layer.gate.weight.data.normal_(0, 0.1)
    for expert in range(8):
        for name, value in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
            layer.load_expert_weight(expert, name, value[expert])
    hidden = torch.randn(tokens, 16)
    if shape3d:
        hidden = hidden.reshape(1, tokens, 16)
    torch.testing.assert_close(layer(hidden), moe_reference(
        hidden, layer.gate.weight, gate, up, down, top_k, normalize,
    ), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tp,ep", [(1, 4), (4, 1), (2, 2)])
def test_weight_slices_reconstruct_full_experts(tp, ep):
    gate = torch.arange(8 * 24 * 16).reshape(8, 24, 16).float()
    up, down = gate + 9000, gate.transpose(1, 2).contiguous()
    for ep_rank in range(ep):
        for tp_rank in range(tp):
            layer = FusedMoE(8, 2, 16, 24, parallel=MoEParallelConfig(
                tp_size=tp, tp_rank=tp_rank, ep_size=ep, ep_rank=ep_rank,
            ))
            for expert in range(8):
                for name, value in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
                    layer.load_expert_weight(expert, name, value[expert])
            first, last = layer.parallel.expert_range(8)
            start, end = tp_rank * (24 // tp), (tp_rank + 1) * (24 // tp)
            expected = torch.cat((gate[first:last, start:end], up[first:last, start:end]), dim=1)
            torch.testing.assert_close(layer.experts.w13_weight, expected)
            torch.testing.assert_close(layer.experts.w2_weight, down[first:last, :, start:end])


@pytest.mark.parametrize("options", [{"ep_size": 0}, {"tp_size": -1}, {"ep_size": 4, "ep_rank": 4},
                                     {"tp_rank": -1}, {"dp_size": True}])
def test_reject_invalid_rank_layout(options):
    with pytest.raises(ValueError):
        MoEParallelConfig(**options)


@pytest.mark.parametrize("args", [(8, 9, 16, 24), (7, 2, 16, 24), (8, 2, 16, 25)])
def test_reject_invalid_expert_layout(args):
    with pytest.raises(ValueError):
        FusedMoE(*args, parallel=MoEParallelConfig(tp_size=2, ep_size=2))


def test_missing_distributed_init_fails_before_collective():
    layer = FusedMoE(8, 2, 16, 24, parallel=MoEParallelConfig(ep_size=4))
    with pytest.raises(RuntimeError, match="Initialize"):
        layer.dispatcher.combine(torch.zeros(2, 16))


def test_cpu_import_does_not_load_gpu_kernels_or_initialize_cuda():
    env = dict(os.environ, MINIINFER_EAGER_IMPORT="0")
    subprocess.run([sys.executable, "-c", "import sys, torch; from miniinfer.layers.fused_moe import FusedMoE; "
                    "from miniinfer.models import Qwen3MoeForCausalLM; "
                    "assert 'miniinfer.layers.fused_moe.fused_moe_kernel' not in sys.modules; "
                    "assert not any(n.startswith('kernels.triton.') for n in sys.modules); "
                    "assert not torch.cuda.is_initialized(); "
                    "assert 'flashinfer' not in sys.modules; assert 'flash_attn' not in sys.modules"],
                   check=True, env=env, timeout=30)
