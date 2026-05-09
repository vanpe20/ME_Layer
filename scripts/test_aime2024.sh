#!/usr/bin/env bash
# AIME2024 evaluation script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL=${MODEL:-Qwen/Qwen3-14B}
DEVICE=${DEVICE:-cuda:0}
DATA=${DATA:-${REPO_ROOT}/data/AIME24/train-00000-of-00001.parquet}
LOG_PATH=${LOG_PATH:-${REPO_ROOT}/res_log_aime24/aime2024_metrics.log}
MASK_RATE=${MASK_RATE:-0.0}
BATCH_SIZE=${BATCH_SIZE:-1}

echo "[AIME2024] Testing model=${MODEL} device=${DEVICE} mask_rate=${MASK_RATE}"

python "${REPO_ROOT}/test/aime2024_test.py" \
  --model_name_or_path "${MODEL}" \
  --device "${DEVICE}" \
  --data_path "${DATA}" \
  --log_path "${LOG_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --mask_rate "${MASK_RATE}"

echo "[AIME2024] Test completed. Results logged to ${LOG_PATH}"
