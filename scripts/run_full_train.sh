#!/usr/bin/env bash
# 4-GPU training: 80 rollout steps, gate (bare + masked val) every 20.
# Keeps only the latest Megatron checkpoint.
# Resume after a finished slime round (ckpt saved, val/gate not run), e.g.:
#   RESUME_ROUND=1 bash scripts/run_full_train.sh
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate slime
fi
# Overlay is 30G; keep inductor / Triton / Ray / Python tmp on autodl-tmp.
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/setup_sys_cache.sh"

ENV_FILE="${PROJECT_ROOT}/.env"
[[ -f "${ENV_FILE}" ]] || { echo "missing ${ENV_FILE}" >&2; exit 2; }
while IFS= read -r line || [[ -n "${line}" ]]; do
  line="${line%$'\r'}"
  [[ -z "${line}" || "${line}" == \#* ]] && continue
  export "${line?}"
done < "${ENV_FILE}"
: "${WANDB_API_KEY:?WANDB_API_KEY missing in .env}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export TP_SIZE="${TP_SIZE:-2}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
# Submit 32 groups, keep 16; GBS 64 → one rollout yields two optimizer steps.
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
# Qwen3.5-4B vocab=248320。峰值由 24k 独苗 Sample 决定，不是打包上限。
# 16384 让两条典型 ~7k Sample 能进同一箱，且不超过已跑通的 24k 峰值。见 docs/archive/oom_qwen35_4b.md。
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-24576}"
# Shared by slime rollouts and the val/gate evaluator so both sides generate
# under the same budget; emit_training_loop_round.py reads it for max_tokens.
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-4096}"
# 超长单条 episode 会单独成批；不分块 softmax 会再复制一份 [T,V] fp32。
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-1024}"
# slime 在每个 job 的最后一步必存；interval 取大数，避免 rollout_id+1
# 整除 20 时在第 19 步多存一份。
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-shopsimrl}"
export WANDB_GROUP="${WANDB_GROUP:-qwen35-4b-train-80}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/runs/wandb}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/root/autodl-tmp/models/Qwen3.5-4B}"
export TRAIN_SAVE="${TRAIN_SAVE:-${PROJECT_ROOT}/runs/slime-qwen35-4b}"
export SAVE_HF="${SAVE_HF:-${PROJECT_ROOT}/runs/slime-qwen35-4b-hf}"
export MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
export SHOPSIM_ENV_SLOTS="${SHOPSIM_ENV_SLOTS:-48}"
export SHOPSIM_PORT="${SHOPSIM_PORT:-5700}"
STEPS_PER_ROUND="${STEPS_PER_ROUND:-20}"
NUM_ROUNDS="${NUM_ROUNDS:-4}"
RESUME_ROUND="${RESUME_ROUND:-0}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SHOPSIM_HEALTH="http://127.0.0.1:${SHOPSIM_PORT}/healthz"
SHOPSIM_PYTHON="${SHOPSIM_PYTHON:-/root/miniconda3/envs/shopsim/bin/python}"
SHOP_ENV_ROOT="${PROJECT_ROOT}/ShopSimulator/shop_env"
INITIAL_SKILLBANK="${INITIAL_SKILLBANK:-${PROJECT_ROOT}/artifacts/cold-start/selected_skillbank.json}"
INITIAL_CURRICULUM="${INITIAL_CURRICULUM:-${PROJECT_ROOT}/artifacts/cold-start/curriculum.json}"

visible="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
[[ "${visible}" -ge "${NUM_GPUS}" ]] || {
  echo "need ${NUM_GPUS} GPUs, nvidia-smi saw ${visible}" >&2
  exit 2
}

mkdir -p "${TRAIN_SAVE}" "${SAVE_HF}" "${WANDB_DIR}" "${PROJECT_ROOT}/runs"

latest_iter() {
  local tracker="${TRAIN_SAVE}/latest_checkpointed_iteration.txt"
  [[ -f "${tracker}" ]] || { echo 0; return; }
  local raw
  raw="$(tr -d '[:space:]' < "${tracker}")"
  if [[ "${raw}" =~ ^iter_([0-9]+)$ ]]; then
    echo "$((10#${BASH_REMATCH[1]}))"
  elif [[ "${raw}" =~ ^[0-9]+$ ]]; then
    echo "$((10#${raw}))"
  else
    echo 0
  fi
}

restore_loop_inputs() {
  local start="$1"
  if (( start == 0 )); then
    skillbank="${INITIAL_SKILLBANK}"
    curriculum="${INITIAL_CURRICULUM}"
    history_ledger=""
    return
  fi
  local prev
  prev="$(printf 'round-%03d' "$((start - 1))")"
  local selected="${PROJECT_ROOT}/runs/${prev}-online-gate/selected_skillbank.json"
  [[ -f "${selected}" ]] || {
    echo "RESUME_ROUND=${start} missing ${selected}" >&2
    exit 2
  }
  skillbank="${selected}"
  curriculum="${PROJECT_ROOT}/runs/$(printf 'round-%03d' "${start}")/curriculum.json"
  [[ -f "${curriculum}" ]] || {
    echo "RESUME_ROUND=${start} missing ${curriculum}" >&2
    exit 2
  }
  local gate_ledger="${PROJECT_ROOT}/runs/${prev}-online-gate/proposal_ledger.jsonl"
  local analysis_ledger="${PROJECT_ROOT}/runs/${prev}-online-analysis/proposal_ledger.jsonl"
  if [[ -f "${gate_ledger}" ]]; then
    history_ledger="${gate_ledger}"
  elif [[ -f "${analysis_ledger}" ]]; then
    history_ledger="${analysis_ledger}"
  else
    history_ledger=""
  fi
}

[[ "${RESUME_ROUND}" =~ ^[0-9]+$ ]] || {
  echo "RESUME_ROUND must be a non-negative integer" >&2
  exit 2
}
(( RESUME_ROUND < NUM_ROUNDS )) || {
  echo "RESUME_ROUND=${RESUME_ROUND} is past NUM_ROUNDS=${NUM_ROUNDS}" >&2
  exit 2
}
existing_iter="$(latest_iter)"
if (( RESUME_ROUND == 0 )); then
  if (( existing_iter > 0 )); then
    echo "TRAIN_SAVE already has checkpoint ${existing_iter}; clear ${TRAIN_SAVE} or set RESUME_ROUND" >&2
    exit 2
  fi
else
  min_iter=$((RESUME_ROUND * STEPS_PER_ROUND))
  if (( existing_iter < min_iter )); then
    echo "RESUME_ROUND=${RESUME_ROUND} needs iter >= ${min_iter}, have ${existing_iter}" >&2
    exit 2
  fi
fi

ensure_shopsim() {
  if curl -sf -m 3 "${SHOPSIM_HEALTH}" >/dev/null; then
    return
  fi
  [[ -x "${SHOPSIM_PYTHON}" ]] || { echo "missing shopsim python: ${SHOPSIM_PYTHON}" >&2; exit 2; }
  echo "starting ShopSimulator on :${SHOPSIM_PORT} slots=${SHOPSIM_ENV_SLOTS}"
  PATH="$(dirname "${SHOPSIM_PYTHON}"):${PATH}" \
    SHOPSIM_ENV_SLOTS="${SHOPSIM_ENV_SLOTS}" \
    SHOPSIM_PORT="${SHOPSIM_PORT}" \
    SHOPSIM_RUNS_ROOT="${PROJECT_ROOT}/runs" \
    nohup bash "${SHOP_ENV_ROOT}/start.sh" \
    > "${PROJECT_ROOT}/runs/shopsim-env.log" 2>&1 &
  echo $! > "${PROJECT_ROOT}/runs/shopsim-env.pid"
  for _ in $(seq 1 60); do
    curl -sf -m 3 "${SHOPSIM_HEALTH}" >/dev/null && return
    sleep 2
  done
  echo "ShopSimulator did not become ready; see ${PROJECT_ROOT}/runs/shopsim-env.log" >&2
  exit 1
}

prune_checkpoints() {
  python3 - <<'PY'
import os, shutil
from pathlib import Path
save = Path(os.environ["TRAIN_SAVE"])
tracker = save / "latest_checkpointed_iteration.txt"
if not tracker.is_file():
    raise SystemExit(f"missing {tracker}")
raw = tracker.read_text(encoding="utf-8").strip()
if raw.startswith("iter_"):
    keep = raw
elif raw.isdigit():
    keep = f"iter_{int(raw):07d}"
else:
    raise SystemExit(f"unrecognized tracker value {raw!r} in {tracker}")
removed = []
for path in save.iterdir():
    if path.is_dir() and path.name.startswith("iter_") and path.name != keep:
        shutil.rmtree(path)
        removed.append(path.name)
print(f"kept {keep}; removed {removed or 'nothing'}")
PY
}

stop_sglang() {
  if [[ -f "${PROJECT_ROOT}/runs/sglang-val.pid" ]]; then
    kill "$(cat "${PROJECT_ROOT}/runs/sglang-val.pid")" >/dev/null 2>&1 || true
    rm -f "${PROJECT_ROOT}/runs/sglang-val.pid"
  fi
  pkill -f "sglang.launch_server" >/dev/null 2>&1 || true
}

stop_analyst() {
  local pid_file="${PROJECT_ROOT}/runs/analyst.pid"
  local stop_file="${PROJECT_ROOT}/runs/analyst.stop"
  if [[ -f "${pid_file}" ]]; then
    touch "${stop_file}"
    local pid
    pid="$(tr -d '[:space:]' < "${pid_file}")"
    if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${pid_file}"
  fi
  rm -f "${stop_file}"
}

serve_checkpoint() {
  stop_sglang
  [[ -d "${SAVE_HF}" && -f "${SAVE_HF}/config.json" ]] || {
    echo "missing HF export at ${SAVE_HF}; slime --save-hf should have written it" >&2
    exit 1
  }
  echo "serving ${SAVE_HF} on :${SGLANG_PORT} tp=${ROLLOUT_TP_SIZE} dp=$((NUM_GPUS / ROLLOUT_TP_SIZE))"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" nohup python3 -m sglang.launch_server \
    --model-path "${SAVE_HF}" \
    --host 127.0.0.1 \
    --port "${SGLANG_PORT}" \
    --served-model-name Qwen/Qwen3.5-4B \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --mem-fraction-static "${SGLANG_MEM_FRACTION:-0.7}" \
    --tp "${ROLLOUT_TP_SIZE}" \
    --dp-size "$((NUM_GPUS / ROLLOUT_TP_SIZE))" \
    > "${PROJECT_ROOT}/runs/sglang-val.log" 2>&1 &
  echo $! > "${PROJECT_ROOT}/runs/sglang-val.pid"
  for _ in $(seq 1 120); do
    curl -sf -m 3 "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null && return
    sleep 2
  done
  echo "sglang val server did not become ready; see ${PROJECT_ROOT}/runs/sglang-val.log" >&2
  exit 1
}

ensure_shopsim
trap 'stop_analyst; stop_sglang; ray stop --force >/dev/null 2>&1 || true' EXIT

skillbank="${INITIAL_SKILLBANK}"
curriculum="${INITIAL_CURRICULUM}"
history_ledger=""
restore_loop_inputs "${RESUME_ROUND}"
if (( RESUME_ROUND > 0 )); then
  export TRAIN_LOAD="${TRAIN_SAVE}"
  echo "===== resume from ${RESUME_ROUND} skillbank=${skillbank} curriculum=${curriculum} iter=$(latest_iter) ====="
  prune_checkpoints
else
  export TRAIN_LOAD="${TRAIN_LOAD:-/root/autodl-tmp/models/Qwen3.5-4B_torch_dist}"
fi

for (( i = RESUME_ROUND; i < NUM_ROUNDS; i++ )); do
  round_id="$(printf 'round-%03d' "${i}")"
  next_id="$(printf 'round-%03d' "$((i + 1))")"
  export NUM_ROLLOUT="$(( (i + 1) * STEPS_PER_ROUND + 1 ))"
  emit_args=(
    --project-root "${PROJECT_ROOT}"
    --round-id "${round_id}"
    --skillbank "${skillbank}"
    --curriculum "${curriculum}"
  )
  if [[ -n "${history_ledger}" ]]; then
    emit_args+=(--history-ledger "${history_ledger}")
  fi
  python3 "${PROJECT_ROOT}/scripts/emit_training_loop_round.py" "${emit_args[@]}"
  export CUSTOM_CONFIG="${PROJECT_ROOT}/runs/loop/${round_id}/slime.yaml"
  # Re-apply in case this bash process started before setup_sys_cache.sh existed.
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/scripts/setup_sys_cache.sh"

  expected_iter=$(( (i + 1) * STEPS_PER_ROUND ))
  if (( $(latest_iter) >= expected_iter )); then
    echo "===== ${round_id} slime already saved iter $(latest_iter) (>= ${expected_iter}); skip ====="
    export TRAIN_LOAD="${TRAIN_SAVE}"
    prune_checkpoints
  else
    echo "===== ${round_id} slime NUM_ROLLOUT=${NUM_ROLLOUT} load=${TRAIN_LOAD} ====="
    rm -f "${PROJECT_ROOT}/runs/analyst.stop" "${PROJECT_ROOT}/runs/analyst.pid"
    python3 scripts/run_shopsimrl.py training watch-analyze \
      "${PROJECT_ROOT}/runs/loop/${round_id}/analysis.yaml" \
      --stop-file "${PROJECT_ROOT}/runs/analyst.stop" \
      > "${PROJECT_ROOT}/runs/loop/${round_id}/analyst.log" 2>&1 &
    echo $! > "${PROJECT_ROOT}/runs/analyst.pid"
    ray stop --force >/dev/null 2>&1 || true
    bash "${PROJECT_ROOT}/scripts/run_shopsimrl_slime.sh"
    ray stop --force >/dev/null 2>&1 || true
    rm -rf /tmp/ray
    prune_checkpoints
    export TRAIN_LOAD="${TRAIN_SAVE}"
    stop_analyst
  fi

  echo "===== ${round_id} compile cards / bare val / gate ====="
  # Analyst failure writes a current-skill-only pool and still continues to val/gate.
  python3 scripts/run_shopsimrl.py training analyze \
    "${PROJECT_ROOT}/runs/loop/${round_id}/analysis.yaml" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-dir "${WANDB_DIR}"
  serve_checkpoint
  python3 scripts/run_shopsimrl.py run \
    "${PROJECT_ROOT}/runs/loop/${round_id}/val.yaml"
  set +e
  python3 scripts/run_shopsimrl.py training online-gate \
    "${PROJECT_ROOT}/runs/loop/${round_id}/gate.yaml" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-dir "${WANDB_DIR}"
  gate_status=$?
  set -e
  stop_sglang

  selected="${PROJECT_ROOT}/runs/${round_id}-online-gate/selected_skillbank.json"
  gate_ledger="${PROJECT_ROOT}/runs/${round_id}-online-gate/proposal_ledger.jsonl"
  analysis_ledger="${PROJECT_ROOT}/runs/${round_id}-online-analysis/proposal_ledger.jsonl"
  if [[ -f "${gate_ledger}" ]]; then
    history_ledger="${gate_ledger}"
  elif [[ -f "${analysis_ledger}" ]]; then
    history_ledger="${analysis_ledger}"
  fi
  if (( gate_status == 0 )) && [[ -f "${selected}" ]]; then
    skillbank="${selected}"
    echo "${round_id} gate complete; next skillbank ${selected}"
  else
    echo "${round_id} gate incomplete; keep current skillbank for ${next_id}" >&2
  fi
  if (( i + 1 < NUM_ROUNDS )); then
    next_curriculum="${PROJECT_ROOT}/runs/${next_id}/curriculum.json"
    mkdir -p "$(dirname "${next_curriculum}")"
    python3 scripts/run_shopsimrl.py training build-curriculum \
      "${skillbank}" \
      "${next_curriculum}" \
      --round-id "${next_id}" \
      --skill-free-probability 0.20 \
      --rho-min 0.20 \
      --rho-max 0.90
    curriculum="${next_curriculum}"
  fi
done

echo "finished ${NUM_ROUNDS} rounds, ${STEPS_PER_ROUND} steps each"
