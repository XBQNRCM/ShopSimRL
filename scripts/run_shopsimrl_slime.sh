#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/setup_sys_cache.sh"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
HF_CHECKPOINT="${HF_CHECKPOINT:?Set HF_CHECKPOINT to the Qwen3.5-4B Hugging Face checkpoint}"
TRAIN_LOAD="${TRAIN_LOAD:?Set TRAIN_LOAD to the converted actor checkpoint}"
TRAIN_SAVE="${TRAIN_SAVE:?Set TRAIN_SAVE to the checkpoint output directory}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
TASK_DATA="${TASK_DATA:-${PROJECT_ROOT}/data/shopsim_train.jsonl}"
CUSTOM_CONFIG="${CUSTOM_CONFIG:-${PROJECT_ROOT}/configs/slime_shopsimrl.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
TP_SIZE="${TP_SIZE:-2}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-21}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
# Caps any single training Sample: a long episode is one micro-batch that
# MAX_TOKENS_PER_GPU cannot split, so this is what bounds peak logits memory.
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-24576}"
# Per-turn generation cap. 2048 cuts ~7% of episodes mid-tool-call (reward 0);
# 4096 was smoke-verified on 4x80GB without changing train step time.
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-1024}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.7}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
SAVE_HF="${SAVE_HF:-}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_ARGS=()
SAVE_HF_ARGS=()
if [[ -n "${SAVE_HF}" ]]; then
  SAVE_HF_ARGS=(--save-hf "${SAVE_HF}")
fi

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
[[ "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "TP_SIZE must be positive" >&2; exit 2; }
[[ "${ROLLOUT_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "ROLLOUT_TP_SIZE must be positive" >&2; exit 2; }
(( NUM_GPUS % TP_SIZE == 0 )) || {
  echo "NUM_GPUS must be divisible by TP_SIZE=${TP_SIZE}" >&2
  exit 2
}
(( NUM_GPUS % ROLLOUT_TP_SIZE == 0 )) || {
  echo "NUM_GPUS must be divisible by ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE}" >&2
  exit 2
}
[[ "${GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "GLOBAL_BATCH_SIZE must be positive" >&2; exit 2; }
[[ "${OVER_SAMPLING_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "OVER_SAMPLING_BATCH_SIZE must be positive" >&2; exit 2; }
(( OVER_SAMPLING_BATCH_SIZE >= ROLLOUT_BATCH_SIZE )) || {
  echo "OVER_SAMPLING_BATCH_SIZE must be >= ROLLOUT_BATCH_SIZE" >&2
  exit 2
}
(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT % GLOBAL_BATCH_SIZE == 0 )) || {
  echo "rollout batch must be divisible by GLOBAL_BATCH_SIZE (count unique episodes, not trajectory segments)" >&2
  exit 2
}
(( GLOBAL_BATCH_SIZE % N_SAMPLES_PER_PROMPT == 0 )) || {
  echo "GLOBAL_BATCH_SIZE must be a multiple of N_SAMPLES_PER_PROMPT so each train step keeps whole groups" >&2
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

NVIDIA_LIB_ROOT="${CONDA_PREFIX:-/root/miniconda3/envs/slime}/lib/python3.12/site-packages/nvidia"
NVIDIA_LDPATH=""
for _nvlib in cudnn cublas nccl nvshmem cusparse cusolver cufft curand cuda_runtime cuda_nvrtc nvjitlink; do
  if [[ -d "${NVIDIA_LIB_ROOT}/${_nvlib}/lib" ]]; then
    NVIDIA_LDPATH="${NVIDIA_LIB_ROOT}/${_nvlib}/lib${NVIDIA_LDPATH:+:${NVIDIA_LDPATH}}"
  fi
done
unset _nvlib
export LD_LIBRARY_PATH="${NVIDIA_LDPATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

SEQUENCE_PARALLEL_ARGS=()
if (( TP_SIZE > 1 )); then
  SEQUENCE_PARALLEL_ARGS=(--sequence-parallel)
fi

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

export SHOPSIMRL_PROJECT_ROOT="${PROJECT_ROOT}"
export RAY_PYTHONPATH="${PROJECT_ROOT}:${SLIME_DIR}:${MEGATRON_PATH}"
RUNTIME_ENV_JSON="$(python3 -c '
import json, os
env = {
    "PYTHONPATH": os.environ["RAY_PYTHONPATH"],
    "SHOPSIMRL_PROJECT_ROOT": os.environ["SHOPSIMRL_PROJECT_ROOT"],
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
}
for key in (
    "WANDB_API_KEY",
    "WANDB_PROJECT",
    "WANDB_GROUP",
    "WANDB_MODE",
    "WANDB_DIR",
    "WANDB_ENTITY",
    "TMPDIR",
    "TEMP",
    "TMP",
    "RAY_TMPDIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "XDG_CACHE_HOME",
):
    value = os.environ.get(key)
    if value:
        env[key] = value
print(json.dumps({"env_vars": env}))
')"

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
  --save-interval "${SAVE_INTERVAL}" \
  "${SAVE_HF_ARGS[@]}" \
  --prompt-data "${TASK_DATA}" \
  --input-key prompt \
  --metadata-key metadata \
  --apply-chat-template \
  --rollout-shuffle \
  --num-rollout "${NUM_ROLLOUT}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}" \
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
  --override-opt-param-scheduler \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --tensor-model-parallel-size "${TP_SIZE}" \
  "${SEQUENCE_PARALLEL_ARGS[@]}" \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}" \
  --recompute-loss-function \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}" \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}" \
  --router-policy consistent_hashing \
  --sglang-tool-call-parser qwen3_coder \
  --sglang-reasoning-parser qwen3 \
  --custom-config-path "${CUSTOM_CONFIG}" \
  --custom-generate-function-path shopsimrl.slime_runtime.generate \
  --custom-rollout-log-function-path shopsimrl.slime_metrics.enrich_rollout_metrics \
  --rollout-all-samples-process-path shopsimrl.slime_metrics.record_generated_rollout_groups \
  --dynamic-sampling-filter-path shopsimrl.slime_runtime.fully_scored_group_filter \
  --custom-reward-post-process-path shopsimrl.slime_runtime.normalize_grpo_by_prompt_and_rollout \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  "${WANDB_ARGS[@]}"
