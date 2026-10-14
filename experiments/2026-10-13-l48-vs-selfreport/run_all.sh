#!/usr/bin/env bash
# 2026-10-13 run, from the repo root, in two stages so the server and the
# activation collection can share the same two GPUs on a busy box:
#   bash run_all.sh games            # server up on cfg.server_url: games, extraction, P(Yes)
#   bash run_all.sh probe 1,2        # server stopped: activations on GPUs 1,2, then analysis
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/2026-10-13-l48-vs-selfreport
export HF_HUB_OFFLINE=1 PATH="$HOME/.local/bin:$PATH"
ENGINE=(uv run --no-project --with-requirements environments/carrot_parsnip/requirements.txt python)

case "$1" in
  games)
    for arm in normal deny; do
      "${ENGINE[@]}" $E/play_games.py --arm $arm
      "${ENGINE[@]}" $E/extract.py --arm $arm
      .venv/bin/python $E/pyes.py --arm $arm
    done ;;
  probe)
    for arm in normal deny; do
      CUDA_VISIBLE_DEVICES="$2" .venv/bin/python $E/collect.py --arm $arm
    done
    .venv/bin/python $E/analyze.py ;;
  *) echo "usage: $0 games | probe <gpu,gpu>" >&2; exit 2 ;;
esac
