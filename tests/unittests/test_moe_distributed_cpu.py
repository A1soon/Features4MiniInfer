"""Four real Gloo processes validate EP, TP and their orthogonal groups."""

import datetime
import json
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miniinfer.config.model.qwen3_moe import Qwen3MoeConfig
from miniinfer.layers.fused_moe import MoEParallelConfig, create_moe_parallel_config
from miniinfer.loader.moe import load_qwen3_moe_weights
from miniinfer.models import Qwen3MoeForCausalLM
from tests.moe_reference import make_checkpoint, moe_reference


def _worker(rank, rendezvous, checkpoint, result_dir):
    from pathlib import Path
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=datetime.timedelta(seconds=40))
    try:
        config = json.loads((Path(checkpoint) / "config.json").read_text())
        full = {}
        for shard in sorted(Path(checkpoint).glob("*.safetensors")):
            full.update(load_file(shard))
        prefix = "model.layers.0.mlp"
        gate = torch.stack([full[f"{prefix}.experts.{e}.gate_proj.weight"] for e in range(8)])
        up = torch.stack([full[f"{prefix}.experts.{e}.up_proj.weight"] for e in range(8)])
        down = torch.stack([full[f"{prefix}.experts.{e}.down_proj.weight"] for e in range(8)])
        records = []
        for tp, ep in ((1, 4), (4, 1), (2, 2)):
            parallel = create_moe_parallel_config(tp_size=tp, ep_size=ep)
            model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**config), "cpu", torch.float32, parallel)
            report = load_qwen3_moe_weights(model, checkpoint)
            layer = model.qwen3_moe.layers[0].mlp.experts
            for concentrated in (False, True):
                hidden = torch.zeros(7, 16)
                if rank == 0:
                    generator = torch.Generator().manual_seed(99)
                    hidden.copy_(torch.randn(7, 16, generator=generator))
                dist.broadcast(hidden, src=0)
                if concentrated:
                    # All selected experts belong to EP rank 0; other EP ranks
                    # must contribute zeros and still enter the same collective.
                    layer.gate.weight.data.zero_()
                    hidden.zero_()
                    hidden[:, 0] = 1
                    layer.gate.weight.data[0, 0] = 3
                    layer.gate.weight.data[1, 0] = 2
                output = layer(hidden)
                expected = moe_reference(hidden, layer.gate.weight, gate, up, down, 2)
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
                replicas = [torch.empty_like(output) for _ in range(4)]
                dist.all_gather(replicas, output)
                for replica in replicas:
                    torch.testing.assert_close(replica, output, rtol=0, atol=0)
                # The orchestrator's sampled token is authoritative on all ranks.
                token = output[0].argmax().reshape(1) if rank == 0 else torch.full((1,), -1)
                dist.broadcast(token, src=0)
                assert token.item() == expected[0].argmax().item()
                records.append(dict(tp=tp, ep=ep, concentrated=concentrated,
                                    loaded_tensors=report.loaded_tensors,
                                    local_experts=layer.experts.num_local_experts,
                                    max_abs_error=(output - expected).abs().max().item()))
            dist.barrier()
            for group in (parallel.ep_group, parallel.tp_group):
                if group is not None:
                    dist.destroy_process_group(group)
        with pytest.raises(ValueError, match="process group"):
            MoEParallelConfig(ep_size=2, ep_rank=rank % 2).validate_group("ep")
        with pytest.raises(ValueError, match="WORLD"):
            create_moe_parallel_config(tp_size=1, ep_size=2)
        (Path(result_dir) / f"rank-{rank}.json").write_text(json.dumps(records, indent=2))
    finally:
        dist.destroy_process_group()


def test_four_rank_cpu_moe(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch build does not include Gloo")
    config, _ = make_checkpoint(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(config))
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    processes = mp.spawn(_worker, args=(rendezvous, str(tmp_path), str(tmp_path)),
                         nprocs=4, join=False)
    deadline = time.monotonic() + 120
    try:
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("CPU distributed test exceeded 120 seconds")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
        for process in processes.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    for rank in range(4):
        records = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert len(records) == 6
