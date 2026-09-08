#!/usr/bin/env bash
# Tear down a training run: outer loop, Ray job, sglang engines, ShopSimulator.
# Orphaned sglang schedulers survive `ray stop` and keep holding VRAM.
set -uo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RAY_BIN="${RAY_BIN:-/root/miniconda3/envs/slime/bin/ray}"

pkill -9 -f 'run_full_train.sh' 2>/dev/null
pkill -9 -f 'training watch-analyze' 2>/dev/null
pkill -9 -f 'ray job submit' 2>/dev/null
sleep 2
"${RAY_BIN}" stop --force >/dev/null 2>&1
pkill -9 -f 'sglang' 2>/dev/null
if [[ -f "${PROJECT_ROOT}/runs/shopsim-env.pid" ]]; then
  kill -9 "$(cat "${PROJECT_ROOT}/runs/shopsim-env.pid")" 2>/dev/null
  rm -f "${PROJECT_ROOT}/runs/shopsim-env.pid"
fi
pkill -9 -f 'shop_env.pack_api' 2>/dev/null

for _ in $(seq 1 30); do
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && break
  sleep 2
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
