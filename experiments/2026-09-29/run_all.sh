#!/usr/bin/env bash
# 2026-09-29 pipeline, two independent lanes (vLLM TP=2 each), from repo root:
#   bash experiments/2026-09-29/run_all.sh a   # GPUs 4,5: new games, their judges + scores, nothink audit, 10k steered audit
#   bash experiments/2026-09-29/run_all.sh b   # GPUs 6,7: instructed pairs, judge ablations, 2k audits, 10k base audit
# then: .venv/bin/python experiments/2026-09-29/analyze.py
set -euo pipefail
cd "$(dirname "$0")/../.."
D=experiments/2026-09-29
VLLM=(env PATH="$PWD/.venv-vllm/bin:$PATH" .venv-vllm/bin/python)
AUDIT=(-m auditor.run --per-group 10 --seeds 5)

lane_a() {
  export CUDA_VISIBLE_DEVICES=4,5
  for arm in noparsnip nothink; do "${VLLM[@]}" $D/play_games.py --arm $arm; done
  "${VLLM[@]}" $D/judge.py --arm noparsnip --view full
  "${VLLM[@]}" $D/judge.py --arm noparsnip --view public
  "${VLLM[@]}" $D/judge.py --arm nothink --view public
  until [ -f $D/transcripts_instructed.jsonl ]; do sleep 60; done     # lane b
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python $D/score_messages.py --arm noparsnip &
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python $D/score_messages.py --arm nothink
  wait
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python $D/score_messages.py --arm instructed
  "${VLLM[@]}" "${AUDIT[@]}" --config $D/audit_2k/config.json --arm nothink \
      --conditions none,lr,dom,judge_public
  "${VLLM[@]}" "${AUDIT[@]}" --config $D/audit_10k/config.json --arm steered
}

lane_b() {
  export CUDA_VISIBLE_DEVICES=6,7
  "${VLLM[@]}" $D/instructed_pairs.py
  for view in redacted public; do "${VLLM[@]}" $D/judge.py --arm base --view $view; done
  for arm in base steered; do
    "${VLLM[@]}" "${AUDIT[@]}" --config $D/audit_2k/config.json --arm $arm
  done
  "${VLLM[@]}" "${AUDIT[@]}" --config $D/audit_10k/config.json --arm base
}

"lane_$1"
