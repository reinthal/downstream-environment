#!/usr/bin/env bash
# 2026-10-01 follow-up ("did you lie") probe pipeline, from repo root.
#   bash experiments/2026-10-01/run_all.sh gen      # GPUs 4,5: on-policy replies (vLLM)
#   bash experiments/2026-10-01/run_all.sh games    # GPUs 6,7: follow-up activations on game turns
#   bash experiments/2026-10-01/run_all.sh rest     # after gen: grade -> activations -> train -> score
#   bash experiments/2026-10-01/run_all.sh audit    # GPUs 4,5: auditor with follow-up grades
set -euo pipefail
cd "$(dirname "$0")/../.."
D=experiments/2026-10-01
VLLM=(env PATH="$PWD/.venv-vllm/bin:$PATH" .venv-vllm/bin/python)
PY=.venv/bin/python

gen() {
  export CUDA_VISIBLE_DEVICES=4,5
  "${VLLM[@]}" $D/generate.py --task roleplaying --thinking off
  "${VLLM[@]}" $D/generate.py --task roleplaying --thinking on
  "${VLLM[@]}" $D/generate.py --task alpaca
}

games() {
  (for s in base steered instructed; do CUDA_VISIBLE_DEVICES=6 $PY $D/collect.py --set $s; done) &
  (for s in nothink noparsnip; do CUDA_VISIBLE_DEVICES=7 $PY $D/collect.py --set $s; done)
  wait
}

rest() {
  # grader: local Qwen3.5-27B (no API key); drop --grader local for the paper's GPT-4o
  for v in off on; do CUDA_VISIBLE_DEVICES=4,5 "${VLLM[@]}" $D/grade.py --thinking $v --grader local; done
  CUDA_VISIBLE_DEVICES=6 $PY $D/collect.py --set roleplaying_off
  CUDA_VISIBLE_DEVICES=6 $PY $D/collect.py --set roleplaying_on
  CUDA_VISIBLE_DEVICES=6 $PY $D/collect.py --set alpaca
  $PY $D/train.py --thinking off
  $PY $D/train.py --thinking on
  $PY $D/score.py
}

audit() {
  export CUDA_VISIBLE_DEVICES=4,5
  for arm in base steered nothink; do
    "${VLLM[@]}" -m auditor.run --per-group 10 --seeds 5 --config $D/audit/config.json --arm $arm
  done
}

"$1"
