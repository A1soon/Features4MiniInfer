"""FP16 Triton MoE checks for the next assigned V100 validation window."""

import pytest
import torch

from miniinfer.layers.fused_moe import FusedMoE, MoEParallelConfig
from tests.moe_reference import moe_reference_with_routes


pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available(), reason="requires an assigned CUDA GPU"
)


@pytest.mark.parametrize("tokens,top_k", [(1, 1), (7, 2), (17, 8), (257, 2)])
@torch.no_grad()
def test_triton_moe_matches_independent_reference(tokens, top_k):
  torch.manual_seed(77)
  hidden = torch.randn(tokens, 64, dtype=torch.float16)
  gate, up, down = [
    (torch.randn(8, *shape) * 0.05).half() for shape in [(96, 64), (96, 64), (64, 96)]
  ]
  layer = FusedMoE(8, top_k, 64, 96, device="cuda", dtype=torch.float16)
  router = (torch.randn(8, 64) * 0.1).half()
  layer.gate.weight.copy_(router.cuda())
  for expert in range(8):
    for name, value in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
      layer.load_expert_weight(expert, name, value[expert])
  # Repeated calls exercise reused workspace with different values.
  for factor in (1.0, 0.5):
    x = (hidden * factor).cuda()
    routed_ids, routed_weights, _ = layer._route(x)
    actual = layer.experts(x, routed_ids, routed_weights).cpu()
    ref_probs = (x.cpu().double() @ router.double().T).softmax(-1)
    ref_values, ref_ids = ref_probs.topk(top_k, dim=-1)
    ref_weights = ref_values / ref_values.sum(-1, keepdim=True)
    routed_ids_cpu = routed_ids.cpu()
    routed_weights_cpu = routed_weights.cpu()
    if top_k < ref_probs.size(-1):
      runner_up = ref_probs.topk(top_k + 1, dim=-1).values[:, -1]
      margins = ref_values[:, -1] - runner_up
      stable = margins > 1e-4
      assert torch.equal(routed_ids_cpu[stable], ref_ids[stable])
      torch.testing.assert_close(
        routed_weights_cpu[stable].float(),
        ref_weights[stable].float(),
        rtol=0.02,
        atol=0.002,
      )
    else:
      torch.testing.assert_close(
        routed_weights_cpu.sort(-1).values.float(),
        ref_weights.sort(-1).values.float(),
        rtol=0.02,
        atol=0.002,
      )
    expected = moe_reference_with_routes(
      hidden * factor, gate, up, down, routed_ids_cpu, routed_weights_cpu
    )
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.03, atol=0.002)


@pytest.mark.parametrize("ep_rank", [0, 1, 2, 3])
@torch.no_grad()
def test_triton_local_experts_and_zero_remote_contribution(ep_rank):
  torch.manual_seed(79)
  layer = FusedMoE(
    8,
    2,
    64,
    96,
    parallel=MoEParallelConfig(ep_size=4, ep_rank=ep_rank),
    device="cuda",
    dtype=torch.float16,
  )
  for parameter in layer.experts.parameters():
    parameter.normal_(0, 0.05)
  # Only global experts 0/1 are selected. EP ranks 1..3 must write zeros,
  # including after a call that previously filled every workspace row.
  hidden = torch.randn(17, 64, device="cuda", dtype=torch.float16)
  weights = torch.full((17, 2), 0.5, device="cuda", dtype=torch.float16)
  begin = ep_rank * 2
  local = torch.tensor([begin, begin + 1], device="cuda").expand(17, -1)
  layer.experts(hidden, local, weights)
  ids = torch.tensor([0, 1], device="cuda").expand(17, -1)
  actual = layer.experts(hidden, ids, weights)
  if ep_rank:
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
  else:
    expected = layer.experts._eager_forward(hidden, ids, weights)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
