#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${MOE_CPU_PYTHON:-python3}
results=${MOE_CPU_RESULTS:-${TMPDIR:-/tmp}/miniinfer-moe-cpu}
mkdir -p "$results"
cd "$root"
export MINIINFER_EAGER_IMPORT=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
"$python" -c 'import platform,torch; print(platform.platform()); print("PyTorch",torch.__version__,"Gloo",torch.distributed.is_gloo_available())' \
  | tee "$results/environment.log"
"$python" -m pytest -q -p no:cacheprovider \
  tests/unittests/test_moe_cpu.py tests/unittests/test_moe_loader_cpu.py \
  tests/unittests/test_moe_distributed_cpu.py \
  --junitxml="$results/junit.xml" 2>&1 | tee "$results/pytest.log"
