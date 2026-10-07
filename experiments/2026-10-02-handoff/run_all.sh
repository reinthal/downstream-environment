#!/usr/bin/env bash
# 2026-10-02 auditor-handoff test, from repo root (GPUs 4-7, two lanes).
set -euo pipefail
cd "$(dirname "$0")/../.."
D=experiments/2026-10-02-handoff
VLLM=(env PATH="$PWD/.venv-vllm/bin:$PATH" .venv-vllm/bin/python)
.venv/bin/python $D/calibrate.py
CUDA_VISIBLE_DEVICES=4,5 "${VLLM[@]}" -m auditor.run --per-group 10 --seeds 5 \
    --config $D/config.json --arm base > $D/logs/audit_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 "${VLLM[@]}" -m auditor.run --per-group 10 --seeds 5 \
    --config $D/config.json --arm steered > $D/logs/audit_steered.log 2>&1 &
wait
