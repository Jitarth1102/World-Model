#!/usr/bin/env bash
# Full local ConvGRU training pipeline (no-memory -> memory -> uncertainty) with --device auto
# (MPS on Apple Silicon, else CUDA/CPU).

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
PREFIX="${PREFIX:-outputs/local_movia200_64_convgru}"

echo "=== ConvGRU pipeline ==="
echo "repo: ${REPO_ROOT}"
echo "device: ${DEVICE}"
echo "outputs: ${PREFIX}_*"
python -c "import torch; print('torch', torch.__version__); print('mps', __import__('torch.backends.mps', fromlist=['']).is_available()); print('cuda', torch.cuda.is_available())"

python scripts/train_nomemory.py \
  --source npz \
  --manifest data/processed/movi_a_128_subset200/manifest.json \
  --epochs 50 \
  --batch-size 16 \
  --image-size 64 \
  --context-frames 4 \
  --predict-frames 4 \
  --hidden-channels 128 \
  --max-train-windows-per-clip 16 \
  --max-val-windows-per-clip 8 \
  --motion-threshold 0.03 \
  --dynamic-loss-weight 3.0 \
  --device "${DEVICE}" \
  --output-dir "${PREFIX}_nomemory_v1"

python scripts/train_memory_model.py \
  --manifest data/processed/movi_a_128_subset200/manifest.json \
  --epochs 50 \
  --batch-size 16 \
  --image-size 64 \
  --context-frames 4 \
  --predict-frames 4 \
  --hidden-channels 128 \
  --max-train-windows-per-clip 16 \
  --max-val-windows-per-clip 8 \
  --motion-threshold 0.03 \
  --dynamic-loss-weight 3.0 \
  --depth-loss-weight 0.1 \
  --memory-covered-loss-weight 1.0 \
  --memory-render-loss-weight 0.0 \
  --memory-grid-resolution 48 40 48 \
  --memory-stride 1 \
  --memory-splat-radius 1 \
  --warm-start-nomemory-checkpoint "${PREFIX}_nomemory_v1/nomemory_model_best.pt" \
  --device "${DEVICE}" \
  --output-dir "${PREFIX}_memory_v1"

python scripts/train_memory_model.py \
  --manifest data/processed/movi_a_128_subset200/manifest.json \
  --epochs 40 \
  --batch-size 16 \
  --image-size 64 \
  --context-frames 4 \
  --predict-frames 4 \
  --hidden-channels 128 \
  --max-train-windows-per-clip 16 \
  --max-val-windows-per-clip 8 \
  --motion-threshold 0.03 \
  --dynamic-loss-weight 3.0 \
  --depth-loss-weight 0.1 \
  --memory-covered-loss-weight 1.0 \
  --memory-render-loss-weight 0.0 \
  --memory-grid-resolution 48 40 48 \
  --memory-stride 1 \
  --memory-splat-radius 1 \
  --enable-uncertainty \
  --uncertainty-loss-weight 0.25 \
  --write-confidence-threshold 0.99 \
  --confidence-gamma 4.0 \
  --warm-start-memory-checkpoint "${PREFIX}_memory_v1/memory_model_best.pt" \
  --device "${DEVICE}" \
  --output-dir "${PREFIX}_uncertainty_v1"

echo "=== ConvGRU pipeline finished ==="
