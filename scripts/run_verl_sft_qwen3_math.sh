#!/usr/bin/env bash
# Minimal wrapper to run verl's FSDP SFT trainer for math SFT data.
# Usage:
#   bash scripts/run_verl_sft_qwen3_math.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"

# Math SFT dataset. Expected columns default to question/response.
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/gsm8k_cot}
TRAIN=${TRAIN:-${DATA_DIR}/train.parquet}
VAL=${VAL:-${DATA_DIR}/val.parquet}
PROMPT_KEY=${PROMPT_KEY:-question}
RESPONSE_KEY=${RESPONSE_KEY:-response}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

MODEL=${MODEL:-Qwen/Qwen3-4B}
NUM_GPUS=${NUM_GPUS:-1}
GBZ=${GBZ:-64}         # global batch size
MBZ=${MBZ:-1}           # micro batch size per GPU
EPOCHS=${EPOCHS:-3}
LR=${LR:-2e-5}
MAX_LEN=${MAX_LEN:-4096}
EXP_NAME=${EXP_NAME:-qwen3-sft-math}
PROJECT_NAME=${PROJECT_NAME:-qwen3-sft}
CKPT_DIR=${CKPT_DIR:-${REPO_ROOT}/checkpoints/${EXP_NAME}}
SAVE_FREQ=${SAVE_FREQ:-100}
MAX_STEPS=${MAX_STEPS:-}
RESUME_MODE=${RESUME_MODE:-auto}
LOGGER=${LOGGER:-[console]}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-True}
MODEL_EXTERNAL_LIB=${MODEL_EXTERNAL_LIB:-modeling_qwen3}
MODELING_FILE=${MODELING_FILE:-${REPO_ROOT}/src/modeling_qwen3.py}
COPY_MODELING_TO_MODEL_DIR=${COPY_MODELING_TO_MODEL_DIR:-1}

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  export WANDB_PROJECT
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  export WANDB_ENTITY
fi
if [[ -n "${WANDB_NAME:-}" ]]; then
  export WANDB_NAME
fi

if [[ ! -f "${TRAIN}" ]]; then
  echo "Missing TRAIN parquet file: ${TRAIN}" >&2
  exit 1
fi
if [[ -n "${VAL}" && ! -f "${VAL}" ]]; then
  echo "Missing VAL parquet file: ${VAL}" >&2
  exit 1
fi

# Ensure local modeling files are available inside the MODEL directory for save/merge.
if [[ "${COPY_MODELING_TO_MODEL_DIR}" == "1" && -d "${MODEL}" && -f "${MODELING_FILE}" ]]; then
  if [[ ! -f "${MODEL}/modeling_qwen3.py" ]]; then
    cp "${MODELING_FILE}" "${MODEL}/modeling_qwen3.py"
  fi
  if [[ ! -f "${MODEL}/modeling_rope_utils.py" ]]; then
    ROPE_SRC=$(python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("transformers.modeling_rope_utils")
print(spec.origin if spec else "")
PY
)
    if [[ -n "${ROPE_SRC}" && -f "${ROPE_SRC}" ]]; then
      cp "${ROPE_SRC}" "${MODEL}/"
    fi
  fi
fi

TRAIN_ARGS=(
  data.train_files="${TRAIN}"
  data.prompt_key="${PROMPT_KEY}"
  data.response_key="${RESPONSE_KEY}"
)
if [[ -n "${VAL}" ]]; then
  TRAIN_ARGS+=(data.val_files="${VAL}")
fi

MODEL_ARGS=(
  model.partial_pretrain="${MODEL}"
  model.trust_remote_code="${TRUST_REMOTE_CODE}"
  model.fsdp_config.model_dtype=bf16
  model.enable_gradient_checkpointing=True
  model.fsdp_config.cpu_offload=False
)
if [[ -n "${MODEL_EXTERNAL_LIB}" ]]; then
  MODEL_ARGS+=(model.external_lib="${MODEL_EXTERNAL_LIB}")
fi

TRAINER_STEP_ARGS=()
if [[ -n "${MAX_STEPS}" ]]; then
  TRAINER_STEP_ARGS+=(trainer.total_training_steps="${MAX_STEPS}")
fi

mkdir -p "${CKPT_DIR}"

echo "[SFT] model=${MODEL}"
echo "[SFT] train=${TRAIN}"
echo "[SFT] val=${VAL:-<none>}"
echo "[SFT] gpus=${NUM_GPUS} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[SFT] checkpoints=${CKPT_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" -m verl.trainer.fsdp_sft_trainer \
  "${TRAIN_ARGS[@]}" \
  data.train_batch_size="${GBZ}" \
  data.micro_batch_size_per_gpu="${MBZ}" \
  data.max_length="${MAX_LEN}" \
  data.truncation=left \
  "${MODEL_ARGS[@]}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.total_epochs="${EPOCHS}" \
  "${TRAINER_STEP_ARGS[@]}" \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.logger="${LOGGER}" \
  optim.lr="${LR}"
