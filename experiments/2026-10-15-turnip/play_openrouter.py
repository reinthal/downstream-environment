"""Play Turnip games on OpenRouter with NO live probe (interrogation only).

A variant of play_games.py for the OpenRouter matchup (DeepSeek-V4.1-Flash
suspects vs a GLM-5.3-Flash Turnip, game_or_interrog.yaml): no Modal proxy
token, no endpoint waking (OpenRouter has none), and detector=None — the
game YAML must set interrogation.detector: off. Everything else (batching,
.eval log capture, the replacement loop, summary format) is play_games'
play_batch, so extract.py reads the output unchanged.

Run (repo root; OPENROUTER_API_KEY is read from .env, not sourced):
  uv run --no-project --with-requirements environments/carrot_parsnip/requirements.txt --with modal --with numpy \
    python experiments/2026-10-15-turnip/play_openrouter.py --arm or_interrog [--pilot 3]
"""
from __future__ import annotations

import argparse
import json
import os

from common import OUT, REPO, load_config, tag

import extract  # noqa: E402

from play_games import play_batch  # noqa: E402  (play_batch is Modal-free)
from game_config import GameConfig  # noqa: E402


def set_openrouter_env() -> None:
    """Export OPENROUTER_API_KEY (+ base URL) from .env into the process env
    where the engine and Inspect's openrouter provider read it. No Modal SDK
    runs here, so sourcing OpenRouter keys is safe (unlike the Modal arms)."""
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    os.environ["OPENROUTER_API_KEY"] = env["OPENROUTER_API_KEY"]
    if env.get("OPENROUTER_BASE_URL"):
        os.environ["OPENROUTER_BASE_URL"] = env["OPENROUTER_BASE_URL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="or_interrog")
    ap.add_argument("--pilot", type=int, default=None, metavar="N", help="play N pilot games and stop")
    args = ap.parse_args()
    cfg = load_config()
    pilot = args.pilot is not None
    t = tag(args.arm, pilot)
    set_openrouter_env()
    game_cfg = GameConfig.from_yaml(REPO / cfg.game_configs[args.arm])
    if game_cfg.interrogation.detector != "off":
        raise ValueError(f"{args.arm}: interrogation.detector must be 'off' for the no-probe run, "
                         f"got {game_cfg.interrogation.detector!r}")
    detector = None
    seed = cfg.pilot_seed if pilot else cfg.arm_seeds[args.arm]
    log_dir = OUT / f"inspect_logs_{t}"
    log_dir.mkdir(exist_ok=True)
    summary_path = OUT / f"summary_{t}.json"
    batches = json.loads(summary_path.read_text()) if summary_path.exists() else []

    if pilot:
        if extract.game_logs(log_dir):
            raise SystemExit(f"{log_dir} already has logs — move it away first")
        batches.append(play_batch(game_cfg, detector, args.pilot, seed, log_dir, 0))
        summary_path.write_text(json.dumps(batches, indent=1, default=str) + "\n")
        print(f"pilot: {args.pilot} games in {log_dir}")
        return

    while True:
        played, valid = extract.count(batches)
        print(f"{t}: {played} played, {valid} valid (target {cfg.n_valid_games})", flush=True)
        if valid >= cfg.n_valid_games:
            break
        if played - valid > cfg.max_replacements:
            raise SystemExit(f"{played - valid} replacements exceed max_replacements="
                             f"{cfg.max_replacements}: stop and debug")
        deficit = cfg.n_valid_games - valid
        batches.append(play_batch(game_cfg, detector, deficit, seed + played, log_dir, played))
        summary_path.write_text(json.dumps(batches, indent=1, default=str) + "\n")
    print(f"done: {len(extract.game_logs(log_dir))} logs in {log_dir}")


if __name__ == "__main__":
    main()
