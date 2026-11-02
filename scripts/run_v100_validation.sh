#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?Usage: bash scripts/run_v100_validation.sh ASSIGNED_GPU_INDEX}
source_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${V100_IMAGE:-miniinfer-v100:py312-torch210-cu126-fa}
results=${V100_RESULTS_DIR:-${TMPDIR:-/tmp}/miniinfer-v100-validation-${UID}}
mkdir -p "$results"
docker image inspect "$image" > "$results/image.json"
docker run --rm --network=none --gpus "device=$gpu" --workdir /workspace \
  -e PYTHONPATH=/workspace -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$source_dir:/workspace:ro" -v "$results:/results" \
  --entrypoint /opt/python312/bin/python3 "$image" \
  -m pytest -p no:cacheprovider -q tests/unittests/test_flashattention_v100.py \
  --junitxml=/results/junit.xml 2>&1 | tee "$results/pytest.log"
printf 'Validation records: %s\n' "$results"
