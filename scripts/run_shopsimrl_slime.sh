#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
HF_CHECKPOINT="${HF_CHECKPOINT:?Set HF_CHECKPOINT to the Qwen3.5-4B Hugging Face checkpoint}"
TRAIN_LOAD="${TRAIN_LOAD:?Set TRAIN_LOAD to the converted actor checkpoint}"
TRAIN_SAVE="${TRAIN_SAVE:?Set TRAIN_SAVE to the checkpoint output directory}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
TASK_DATA="${TASK_DATA:-${PROJECT_ROOT}/data/shopsim_train.jsonl}"
CUSTOM_CONFIG="${CUSTOM_CONFIG:-${PROJECT_ROOT}/configs/slime_shopsimrl.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_ARGS=()

case "${USE_WANDB}" in
  0) ;;
  1)
    WANDB_ARGS=(
      --use-wandb
      --wandb-mode "${WANDB_MODE:-online}"
      --wandb-project "${WANDB_PROJECT:-shopsimrl}"
      --wandb-group "${WANDB_GROUP:-shopsimrl-training}"
      --wandb-dir "${WANDB_DIR:-${PROJECT_ROOT}/runs/wandb}"
      --disable-wandb-random-suffix
    )
    [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_ARGS+=(--wandb-team "${WANDB_ENTITY}")
    ;;
  *) echo "USE_WANDB must be 0 or 1" >&2; exit 2 ;;
esac

[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_GPUS must be positive" >&2; exit 2; }
(( NUM_GPUS % 2 == 0 )) || {
  echo "NUM_GPUS must be divisible by the current TP=2 training and rollout topology" >&2
  exit 2
}
[[ "${GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "GLOBAL_BATCH_SIZE must be positive" >&2; exit 2; }
(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT % GLOBAL_BATCH_SIZE == 0 )) || {
  echo "rollout batch must be divisible by GLOBAL_BATCH_SIZE (count unique episodes, not trajectory segments)" >&2
  exit 2
}

[[ -f "${SLIME_DIR}/train.py" ]] || { echo "missing slime checkout: ${SLIME_DIR}" >&2; exit 2; }
[[ -f "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh" ]] || { echo "missing Qwen3.5 model args" >&2; exit 2; }
[[ -d "${HF_CHECKPOINT}" ]] || { echo "missing HF_CHECKPOINT: ${HF_CHECKPOINT}" >&2; exit 2; }
[[ -d "${TRAIN_LOAD}" ]] || { echo "missing TRAIN_LOAD: ${TRAIN_LOAD}" >&2; exit 2; }
[[ -f "${TASK_DATA}" ]] || { echo "missing TASK_DATA: ${TASK_DATA}; run training prepare-data" >&2; exit 2; }
[[ -f "${CUSTOM_CONFIG}" ]] || { echo "missing CUSTOM_CONFIG: ${CUSTOM_CONFIG}" >&2; exit 2; }
command -v ray >/dev/null || { echo "ray is not on PATH" >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 is not on PATH" >&2; exit 2; }

cd "${PROJECT_ROOT}"
source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

SHOPSIMRL_PROJECT_ROOT="${PROJECT_ROOT}" python3 scripts/run_shopsimrl.py training check \
  "${CUSTOM_CONFIG}" --task-data "${TASK_DATA}"

ray start \
  --head \
  --node-ip-address "${MASTER_ADDR:-127.0.0.1}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host 0.0.0.0 \
  --dashboard-port 8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PROJECT_ROOT}:${SLIME_DIR}:${MEGATRON_PATH}\",
    \"SHOPSIMRL_PROJECT_ROOT\": \"${PROJECT_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\"
  }
}"

ray job submit --address http://127.0.0.1:8265 \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_DIR}/train.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --load "${TRAIN_LOAD}" \
  --save "${TRAIN_SAVE}" \
  --save-interval 10 \
  --prompt-data "${TASK_DATA}" \
  --input-key prompt \
  --metadata-key metadata \
  --apply-chat-template \
  --rollout-shuffle \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --rollout-max-response-len 2048 \
  --rollout-max-context-len 32768 \
  --rollout-temperature 0.6 \
  --rollout-top-p 1.0 \
  --rollout-top-k -1 \
  --reward-key reward \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --balance-data \
  --advantage-estimator grpo \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --tensor-model-parallel-size 2 \
  --sequence-parallel \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 32768 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --rollout-num-gpus-per-engine 2 \
  --sglang-mem-fraction-static 0.7 \
  --router-policy consistent_hashing \
  --sglang-tool-call-parser qwen3_coder \
  --sglang-reasoning-parser qwen3 \
  --custom-config-path "${CUSTOM_CONFIG}" \
  --custom-generate-function-path shopsimrl.slime_runtime.generate \
  --custom-rollout-log-function-path shopsimrl.slime_metrics.enrich_rollout_metrics \
  --dynamic-sampling-filter-path shopsimrl.slime_runtime.fully_scored_group_filter \
  --custom-reward-post-process-path shopsimrl.slime_runtime.normalize_grpo_by_prompt_and_rollout \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  "${WANDB_ARGS[@]}"
