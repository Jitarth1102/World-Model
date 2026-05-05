#!/usr/bin/env bash
# Full local training: ConvGRU only (3 stages).
# Run: bash scripts/train_full_pipeline.sh

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

echo "=== Full pipeline: ConvGRU only ==="
echo "device: ${DEVICE}"

bash "${REPO_ROOT}/scripts/train_convgru_all.sh"

echo "=== Full pipeline complete ==="
