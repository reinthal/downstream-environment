"""Play Carrot-Parsnip games on the local Qwen3.5-27B vLLM server until the
arm has cfg.n_valid_games valid games (preregistered replacement rule).

Each batch plays the deficit at the next unused seeds (``arm_seed + index``),
copies the engine's Inspect logs to inspect_logs_{tag}/game_{index:03d}.eval
and re-counts the valid games with extract.count_valid. More than
cfg.max_replacements extra games aborts the arm: that is a pipeline problem
to debug and record under Deviations, not something to play through.

Run:  uv run --no-sync python experiments/2026-10-12-fibs-probes/play_games.py --arm 3p
Pilot (seeds from pilot_seed, never analysed): --arm 3p --pilot 2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from common import OUT, REPO, load_config, tag

import extract  # noqa: E402

from Carrot_Parsnip_Agents import run_games  # noqa: E402  (engine, via common's sys.path)
from game_config import GameConfig  # noqa: E402


def play_batch(game_cfg: GameConfig, n: int, base_seed: int, log_dir, first_index: int) -> dict:
    summary = run_games(num_games=n, base_seed=base_seed, config=game_cfg)
    logs = summary.pop("eval_logs")
    if len(logs) != n:
        raise RuntimeError(f"engine returned {len(logs)} logs for {n} games")
    for i, log in enumerate(logs):
        if log.eval.task != f"carrot_parsnip_game_{i}":
            raise RuntimeError(f"log {i} is task {log.eval.task!r}")
        shutil.copy(log.location, log_dir / f"game_{first_index + i:03d}.eval")
        for sample in log.samples or []:
            if sample.error is not None and "ReasoningNotExposedError" in sample.error.message:
                raise SystemExit(f"game {first_index + i}: {sample.error.message.splitlines()[0][:300]}\n"
                                 "the provider does not expose the model's reasoning; fix the server "
                                 "or the model before playing on")
    summary["base_seed"], summary["first_index"], summary["n"] = base_seed, first_index, n
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pilot", type=int, default=None, metavar="N", help="play N pilot games and stop")
    args = ap.parse_args()
    cfg = load_config()
    pilot = args.pilot is not None
    t = tag(args.arm, pilot)
    os.environ.setdefault("VLLM_BASE_URL", cfg.vllm_base_url)
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")
    if not pilot:
        cfg.save(OUT)
    game_cfg = GameConfig.from_yaml(REPO / cfg.game_configs[args.arm])
    seed = cfg.pilot_seed if pilot else cfg.arm_seeds[args.arm]
    log_dir = OUT / f"inspect_logs_{t}"
    log_dir.mkdir(exist_ok=True)
    summary_path = OUT / f"summary_{t}.json"
    batches = json.loads(summary_path.read_text()) if summary_path.exists() else []

    if pilot:
        if extract.game_logs(log_dir):
            raise SystemExit(f"{log_dir} already has logs — move it away first")
        batches.append(play_batch(game_cfg, args.pilot, seed, log_dir, 0))
        summary_path.write_text(json.dumps(batches, indent=1, default=str) + "\n")
        print(f"pilot: {args.pilot} games in {log_dir}")
        return

    while True:
        logs = extract.game_logs(log_dir)
        played, valid = len(logs), extract.count_valid(logs, game_cfg)
        print(f"{t}: {played} played, {valid} valid (target {cfg.n_valid_games})", flush=True)
        if valid >= cfg.n_valid_games:
            break
        if played - valid > cfg.max_replacements:
            raise SystemExit(f"{played - valid} replacements exceed max_replacements="
                             f"{cfg.max_replacements}: stop and debug (see PREREGISTRATION.md)")
        deficit = cfg.n_valid_games - valid
        batches.append(play_batch(game_cfg, deficit, seed + played, log_dir, played))
        summary_path.write_text(json.dumps(batches, indent=1, default=str) + "\n")
    print(f"done: {len(extract.game_logs(log_dir))} logs in {log_dir}")


if __name__ == "__main__":
    main()
