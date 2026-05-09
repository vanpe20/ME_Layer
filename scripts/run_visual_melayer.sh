#!/usr/bin/env bash
# Run the ME Layer decoder-output L2 norm visualization.
#
# Example:
#   MODEL=Qwen/Qwen3-4B DEVICE=cuda:0 bash scripts/run_visual_melayer.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
PYTHON=${PYTHON:-python}

MODEL=${MODEL:-Qwen/Qwen3-4B}
DEVICE=${DEVICE:-cuda:0}
SAVE_PATH=${SAVE_PATH:-${REPO_ROOT}/visual_res/me_layer_l2_by_dim.png}

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

"${PYTHON}" "${REPO_ROOT}/test/visual_melayer.py" \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --save-path "${SAVE_PATH}" \
  "$@"
