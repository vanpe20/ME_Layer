#!/usr/bin/env bash
# GRPO training for Qwen/Qwen3-4B on GSM8K-CoT parquet.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
PYTHON=${PYTHON:-python}
if [[ ! -x "$(command -v "${PYTHON}" 2>/dev/null)" ]]; then
  echo "PYTHON not found or not executable: ${PYTHON}" >&2
  echo "Set PYTHON to a Python with pandas/pyarrow and verl dependencies." >&2
  exit 1
fi

RAW_TRAIN_PARQUET=${RAW_TRAIN_PARQUET:-}
RAW_VAL_PARQUET=${RAW_VAL_PARQUET:-}

DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/gsm8k_cot_verl}
TRAIN_PARQUET=${TRAIN_PARQUET:-${DATA_DIR}/train.parquet}
VAL_PARQUET=${VAL_PARQUET:-${DATA_DIR}/test.parquet}

MODEL=${MODEL:-Qwen/Qwen3-4B}
MODEL_EXTERNAL_LIB=${MODEL_EXTERNAL_LIB:-modeling_qwen3_train}

NUM_GPUS=${NUM_GPUS:-2}
NNODES=${NNODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-16}
ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE:-16}

MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-256}
ROLLOUT_N=${ROLLOUT_N:-4}
GEN_TP=${GEN_TP:-1}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k}
EXP_NAME=${EXP_NAME:-qwen3_4b_grpo_gsm8k_cot_if_our}
OUTPUT_DIR=${OUTPUT_DIR:-/common/users/zs618/qw_rlmath_IF_our_new}
# Save HF model weights as safetensors under each checkpoint's huggingface/ directory.
CKPT_SAVE_CONTENTS=${CKPT_SAVE_CONTENTS:-"['hf_model']"}
# RESUME_PATH=${RESUME_PATH:-/common/users/zs618/qw_rlmath/verl_grpo_gsm8k/qwen3_4b_grpo_gsm8k_cot/global_step_40}
EPOCHS=${EPOCHS:-5}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:--1}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export WANDB_ENTITY=${WANDB_ENTITY:-shizeru77-rutgers-university}
export WANDB_PROJECT=${WANDB_ENTITY:-massive_value}

mkdir -p "${DATA_DIR}"

if [[ "${FORCE_PREP:-0}" == "1" || ! -f "${TRAIN_PARQUET}" || ! -f "${VAL_PARQUET}" ]]; then
  if [[ -z "${RAW_TRAIN_PARQUET}" || -z "${RAW_VAL_PARQUET}" ]]; then
    echo "Missing TRAIN/VAL parquet and no RAW_TRAIN_PARQUET/RAW_VAL_PARQUET provided." >&2
    echo "Please set TRAIN_PARQUET/VAL_PARQUET for existing data or set RAW_TRAIN_PARQUET/RAW_VAL_PARQUET to prepare from raw GSM8K parquet." >&2
    exit 1
  fi

  echo "Preparing GSM8K-CoT parquet for verl..."
  export RAW_TRAIN_PARQUET RAW_VAL_PARQUET TRAIN_PARQUET VAL_PARQUET INSTRUCTION=${INSTRUCTION:-}
  "${PYTHON}" - <<'PY'
import os
import pandas as pd

train_src = os.environ["RAW_TRAIN_PARQUET"]
val_src = os.environ["RAW_VAL_PARQUET"]
train_dst = os.environ["TRAIN_PARQUET"]
val_dst = os.environ["VAL_PARQUET"]
instruction = os.environ.get("INSTRUCTION") or 'Let\'s think step by step and output the final answer after "####".'

def convert(src, dst, split):
    df = pd.read_parquet(src)
    records = []
    for idx, row in df.iterrows():
        question = row.get("question")
        if question is None:
            raise ValueError(f"Missing 'question' in {src}")
        answer = row.get("answer")
        if answer is None:
            raise ValueError(f"Missing 'answer' in {src}")
        answer = str(answer).strip()
        prompt = [{"role": "user", "content": f"{question} {instruction}"}]
        records.append(
            {
                "data_source": "openai/gsm8k",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": int(idx),
                    "question": question,
                    "answer": answer,
                    "response": row.get("response"),
                },
            }
        )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(dst, index=False)

convert(train_src, train_dst, "train")
convert(val_src, val_dst, "test")
PY
  if [[ "${PREP_ONLY:-0}" == "1" ]]; then
    exit 0
  fi
fi

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.prompt_key=prompt \
  data.max_prompt_length="${MAX_PROMPT_LEN}" \
  data.max_response_length="${MAX_RESPONSE_LEN}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.model.external_lib="${MODEL_EXTERNAL_LIB}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.checkpoint.save_contents="${CKPT_SAVE_CONTENTS}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ROLLOUT_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${EPOCHS}" \
  "$@"
