#!/usr/bin/env bash
# GSM8K evaluation script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL=${MODEL:-Qwen/Qwen3-4B}
DEVICE=${DEVICE:-cuda:0}
DATA=${DATA:-${REPO_ROOT}/data/gsm8k_cot_verl/test.parquet}
LOG_PATH=${LOG_PATH:-${REPO_ROOT}/res_log_gsm8k/gsm8k_metrics.log}
MASK_RATE=${MASK_RATE:-0.0}
BATCH_SIZE=${BATCH_SIZE:-1}

echo "[GSM8K] Testing model=${MODEL} device=${DEVICE} mask_rate=${MASK_RATE}"

python "${REPO_ROOT}/test/gsm8k.py" \
  --model_name_or_path "${MODEL}" \
  --device "${DEVICE}" \
  --data_path "${DATA}" \
  --log_path "${LOG_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --mask_rate "${MASK_RATE}"

echo "[GSM8K] Test completed. Results logged to ${LOG_PATH}"
