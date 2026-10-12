"""Play 3-player Carrot-Parsnip games with the engine's own orchestrator
(as experiments/2026-10-09-3p-game/play_games.py). The game is game.yaml;
the players are Qwen3.8-27B on the local server (inspect's openai-api/vllm
provider, pointed at cfg.server_url). The engine writes one Inspect log per
game; they are copied to inspect_logs_{arm}/ and the batch summary to
summary_{arm}.json. extract.py turns the logs into transcripts_/games_ files.

Run (server up):
      uv run --no-project --with-requirements environments/carrot_parsnip/requirements.txt \
          python experiments/2026-10-12-qwen38-layer-sweep/play_games.py --arm base
Pilot (seeds cfg.pilot_seed.., never analysed): --arm pilot --games 2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from common import OUT, REPO, load_config

from Carrot_Parsnip_Agents import run_games  # noqa: E402  (engine, via common's sys.path)
from game_config import GameConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["base", "pilot"], required=True)
    ap.add_argument("--games", type=int, default=None, help="pilot only")
    args = ap.parse_args()
    cfg = load_config()
    if args.arm == "base":
        if args.games is not None:
            ap.error("--games is for the pilot; the base arm runs cfg.n_games")
        n, seed = cfg.n_games, cfg.game_seed
        cfg.save(OUT)
    else:
        if args.games is None:
            ap.error("--arm pilot requires --games")
        n, seed = args.games, cfg.pilot_seed
    log_dir = OUT / f"inspect_logs_{args.arm}"
    if log_dir.exists():
        raise SystemExit(f"{log_dir} exists — move it away before re-running this arm")

    game_cfg = GameConfig.from_yaml(REPO / cfg.game_config)
    if game_cfg.eval_model != f"openai-api/vllm/{cfg.served_model}":
        raise SystemExit(f"game.yaml eval_model {game_cfg.eval_model!r} is not the served model")
    os.environ["VLLM_BASE_URL"] = cfg.server_url
    os.environ["VLLM_API_KEY"] = "local"          # the local server checks no key
    summary = run_games(num_games=n, base_seed=seed, config=game_cfg)
    logs = summary.pop("eval_logs")
    log_dir.mkdir()
    for i, log in enumerate(logs):
        if log.eval.task != f"carrot_parsnip_game_{i}":
            raise RuntimeError(f"log {i} is task {log.eval.task!r}")
        shutil.copy(log.location, log_dir / f"game_{i:03d}.eval")
    summary["base_seed"] = seed
    (OUT / f"summary_{args.arm}.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"copied {len(logs)} logs to {log_dir}")


if __name__ == "__main__":
    main()
