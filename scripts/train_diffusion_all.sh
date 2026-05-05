#!/usr/bin/env bash
# Diffusion training attempt pipeline (no-memory -> memory -> uncertainty eval).
# Kept for reproducibility of attempted runs; in this project snapshot ConvGRU results are primary.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.venv/bin/activate"
fi
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

DEVICE="${DEVICE:-auto}"
PREFIX="${PREFIX:-outputs/local_movia200_64_diffusion}"

echo "=== Diffusion pipeline (attempt) ==="
echo "repo: ${REPO_ROOT}"
echo "device: ${DEVICE}"
echo "outputs: ${PREFIX}_*"
python -c "import torch; print('torch', torch.__version__); print('mps', __import__('torch.backends.mps', fromlist=['']).is_available()); print('cuda', torch.cuda.is_available())"

python scripts/train_diffusion.py \
  --variant no_memory \
  --manifest data/processed/movi_a_128_subset200/manifest.json \
  --epochs 100 \
  --steps 20000 \
  --batch-size 8 \
  --image-size 64 \
  --context-frames 4 \
  --predict-frames 4 \
  --model-channels 64 \
  --diffusion-steps 100 \
  --sample-steps-eval 50 \
  --eval-max-batches 8 \
  --max-train-windows-per-clip 16 \
  --max-val-windows-per-clip 8 \
  --device "${DEVICE}" \
  --output-dir "${PREFIX}_nomemory_v1"

python scripts/train_diffusion.py \
  --variant memory \
  --manifest data/processed/movi_a_128_subset200/manifest.json \
  --epochs 100 \
  --steps 20000 \
  --batch-size 8 \
  --image-size 64 \
  --context-frames 4 \
  --predict-frames 4 \
  --model-channels 64 \
  --diffusion-steps 100 \
  --sample-steps-eval 50 \
  --eval-max-batches 8 \
  --max-train-windows-per-clip 16 \
  --max-val-windows-per-clip 8 \
  --memory-grid-resolution 48 40 48 \
  --memory-stride 1 \
  --memory-splat-radius 1 \
  --device "${DEVICE}" \
  --output-dir "${PREFIX}_memory_v1"

python scripts/eval_diffusion_uncertainty.py \
  --checkpoint "${PREFIX}_memory_v1/diffusion_model_best.pt" \
  --manifest data/processed/movi_a_128_subset200/manifest.json \
  --context-frames 4 \
  --predict-frames 4 \
  --image-size 64 \
  --max-val-windows-per-clip 8 \
  --sample-steps 50 \
  --uncertainty-samples 8 \
  --write-confidence-threshold 0.99 \
  --confidence-gamma 4.0 \
  --memory-grid-resolution 48 40 48 \
  --memory-stride 1 \
  --memory-splat-radius 1 \
  --device "${DEVICE}" \
  --output-dir "${PREFIX}_uncertainty_v1"

echo "=== Diffusion pipeline finished ==="
