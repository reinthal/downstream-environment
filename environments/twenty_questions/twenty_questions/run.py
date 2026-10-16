"""Play a batch of games and write the experiment files:

    {out}/games_{arm}.json         one summary per game (secret, winner, exchanges)
    {out}/transcripts_{arm}.jsonl  one row per player action (records.Turn)
    {out}/game_{arm}.yaml          the full GameConfig that produced them
    {out}/inspect_logs_{arm}/      the Inspect eval log (inspect view)

    uv run --directory environments/twenty_questions python -m twenty_questions.run \
        --config configs/default.yaml --n-games 4 --out /path/to/experiment --arm pilot

Games that error (a player failed to act within max_attempts, an API
failure) are listed in games_{arm}.json under "errors" and make the run exit
non-zero; they are never silently dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inspect_ai import eval as inspect_eval

from .config import GameConfig
from .task import make_task


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--n-games", type=int, required=True)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    args = ap.parse_args(argv)

    cfg = GameConfig.from_yaml(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    games_path = args.out / f"games_{args.arm}.json"
    if games_path.exists():
        raise SystemExit(f"{games_path} exists — pick a new --arm or move it away")
    cfg.to_yaml(args.out / f"game_{args.arm}.yaml")

    task = make_task(cfg, args.n_games, args.base_seed, name=f"twenty_questions_{args.arm}")
    [log] = inspect_eval(task, log_dir=str(args.out / f"inspect_logs_{args.arm}"),
                         max_samples=cfg.max_concurrent_games, max_connections=cfg.max_connections,
                         fail_on_error=False)
    games, errors = [], []
    with open(args.out / f"transcripts_{args.arm}.jsonl", "w") as f:
        for sample in sorted(log.samples or [], key=lambda s: s.metadata["seed"]):
            if sample.error is not None:
                errors.append({"game_id": sample.id, "seed": sample.metadata["seed"],
                               "error": sample.error.message})
                continue
            games.append(sample.store["game"])
            for turn in sample.store["turns"]:
                f.write(json.dumps(turn) + "\n")
    games_path.write_text(json.dumps({"games": games, "errors": errors, "log": log.location}, indent=1) + "\n")
    won = sum(g["winner"] == "auditor" for g in games)
    print(f"{args.arm}: {len(games)} games ({won} auditor wins), {len(errors)} errors -> {args.out}")
    for e in errors:
        print(f"  ERROR {e['game_id']}: {e['error'].splitlines()[0][:300]}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
