#!/usr/bin/env bash
# 2026-10-12 layer sweep, non-pilot run, from the repo root, with the Qwen3.8
# server already up on cfg.server_url (see README). Roleplaying generation /
# grading run alongside the games (both only need the server); activation
# collection (GPUs 4,5, the whole model) starts once both are done.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/2026-10-12-qwen38-layer-sweep
export HF_HUB_OFFLINE=1 PATH="$HOME/.local/bin:$PATH"
ENGINE=(uv run --no-project --with-requirements environments/carrot_parsnip/requirements.txt python)

(.venv/bin/python $E/generate_roleplay.py && .venv/bin/python $E/grade.py) &
roleplay=$!

"${ENGINE[@]}" $E/play_games.py --arm base
"${ENGINE[@]}" $E/extract.py --arm base
.venv/bin/python $E/instructed.py
wait $roleplay

for s in roleplaying games instructed; do
  CUDA_VISIBLE_DEVICES=4,5 .venv/bin/python $E/collect.py --set $s
done
.venv/bin/python $E/analyze.py
