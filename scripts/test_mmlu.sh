#!/usr/bin/env bash
# MMLU evaluation script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL=${MODEL:-Qwen/Qwen3-14B}
DEVICE=${DEVICE:-cuda:0}
DATASET=${DATASET:-cais/mmlu}
LOG_PATH=${LOG_PATH:-${REPO_ROOT}/res_log_mmlu/mmlu_metrics.log}
MASK_RATE=${MASK_RATE:-0.0}
MAX_SAMPLES=${MAX_SAMPLES:-100}

echo "[MMLU] Testing model=${MODEL} device=${DEVICE} mask_rate=${MASK_RATE}"

python "${REPO_ROOT}/test/mmlu_eval.py" \
  --model_name_or_path "${MODEL}" \
  --device "${DEVICE}" \
  --dataset_name "${DATASET}" \
  --eval_split test \
  --max_samples "${MAX_SAMPLES}" \
  --log_path "${LOG_PATH}" \
  --mask_rate "${MASK_RATE}"

echo "[MMLU] Test completed. Results logged to ${LOG_PATH}"
