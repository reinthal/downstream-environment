"""Play 3-player Carrot-Parsnip games for one arm with the engine's own
orchestrator (as experiments/2026-10-12-qwen38-layer-sweep/play_games.py).
Both arms use seeds game_seed.. so role assignments are paired; speaking
orders are not (the engine's order RNG is unseeded). Players: Qwen3.8-27B
on the local server. Logs -> inspect_logs_{arm}/, summary -> summary_{arm}.json.

Run (server up):
      uv run --no-project --with-requirements environments/carrot_parsnip/requirements.txt \
          python experiments/2026-10-13-l48-vs-selfreport/play_games.py --arm normal
Pilot (never analysed): --arm deny_pilot --games 1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from common import ARMS, OUT, arm_seed, arm_yaml, load_config

from Carrot_Parsnip_Agents import run_games  # noqa: E402  (engine, via common's sys.path)
from game_config import GameConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--games", type=int, default=None, help="pilot arms only")
    args = ap.parse_args()
    cfg = load_config()
    pilot = args.arm.endswith("_pilot")
    if pilot == (args.games is None):
        ap.error("--games is required for pilot arms and not allowed otherwise")
    n = args.games if pilot else cfg.n_games
    if not pilot:
        cfg.save(OUT)
    log_dir = OUT / f"inspect_logs_{args.arm}"
    if log_dir.exists():
        raise SystemExit(f"{log_dir} exists — move it away before re-running this arm")

    game_cfg = GameConfig.from_yaml(arm_yaml(cfg, args.arm))
    if game_cfg.eval_model != f"openai-api/vllm/{cfg.served_model}":
        raise SystemExit(f"eval_model {game_cfg.eval_model!r} is not the served model")
    os.environ["VLLM_BASE_URL"] = cfg.server_url
    os.environ["VLLM_API_KEY"] = "local"          # the local server checks no key
    seed = arm_seed(cfg, args.arm)
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
