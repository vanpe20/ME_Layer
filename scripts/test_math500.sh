#!/usr/bin/env bash
# Math-500 evaluation script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL=${MODEL:-Qwen/Qwen3-4B}
DEVICE=${DEVICE:-cuda:0}
DATASET_ROOT=${DATASET_ROOT:-${REPO_ROOT}/dataset/Math-500}
MASK_RATE=${MASK_RATE:-0.0}

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Error: Math-500 dataset not found at ${DATASET_ROOT}" >&2
  exit 1
fi

echo "[Math-500] Testing model=${MODEL} device=${DEVICE} mask_rate=${MASK_RATE}"

cd "${REPO_ROOT}"
python "${REPO_ROOT}/test/math500.py" \
  --model_name_or_path "${MODEL}" \
  --device "${DEVICE}" \
  --mask_rate "${MASK_RATE}"

echo "[Math-500] Test completed."
