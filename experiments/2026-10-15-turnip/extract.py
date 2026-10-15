"""Turn play_games.py's summaries into the analysis files (CPU).

The engine records each game (result dict: roles, Turnip, speaking orders,
announcements and QA in order, the QA records with their live detector
scores, votes, outcome). Preregistered exclusions (PREREGISTRATION.md §4):
engine/provider errors — the game failed or never resolved, which includes a
QA turn with no valid answer or question after 3 attempts (the engine
raises) — and malformed turns (two announcements by one player in one
round). Silent discussion turns are not an exclusion; they are counted.

  games_{tag}.json      per valid game: seed, roles, turnip/carrot/parsnip,
                        qa targets per round, votes, winner, outcome, flags
  qa_{tag}.jsonl        every QA answer of the valid games (+ detector scores)
  transcripts_{tag}.jsonl  announcements, questions and answers in order
  excluded_{tag}.json   dropped games with the reason

Run:  $R python experiments/2026-10-15-turnip/extract.py --arm interrog [--pilot]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import OUT, load_config, tag, write_jsonl


class GameExcluded(Exception):
    """Excluded by the preregistered rules."""


def game_logs(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("game_*.eval"))


def games_of(batches: list[dict]) -> list[tuple[int, int, dict]]:
    """(global index, seed, engine result) for every game played."""
    out = []
    for b in batches:
        games = b["games"]
        if len(games) != b["n"]:
            raise ValueError(f"batch at {b['first_index']}: {len(games)} results for {b['n']} games")
        for g in games:
            i = g["game_id"]
            out.append((b["first_index"] + i, b["base_seed"] + i, g))
    idx = [i for i, _, _ in out]
    if idx != list(range(len(out))):
        raise ValueError(f"game indices not contiguous: {idx}")
    return out


def check_game(g: dict, n_rounds: int, exchanges: int) -> dict:
    if g["outcome"] in ("failed", "not_run"):
        raise GameExcluded(f"{g['outcome']}: {g.get('failure_reason', '')[:300]}")
    if g["outcome"] not in ("carrot_win", "parsnip_win_other_eliminated", "parsnip_win_nobody_eliminated"):
        raise ValueError(f"unexpected outcome {g['outcome']!r}")
    ann = [m for m in g["announcements"] if m["kind"] == "announcement"]
    per_turn = Counter((m["round"], m["speaker"]) for m in ann)
    doubled = sorted(k for k, c in per_turn.items() if c > 1)
    if doubled:
        raise GameExcluded(f"two announcements in one turn: {doubled}")
    qa = g["qa"]
    if len(qa) != n_rounds * exchanges:
        raise ValueError(f"{len(qa)} QA answers, expected {n_rounds * exchanges}")
    names = list(g["roles"])
    silent = len(names) * n_rounds - len(ann)
    if silent < 0:
        raise ValueError("more announcements than turns")
    by_role = {r: n for n, r in g["roles"].items()}
    targets = [qa[r * exchanges]["target"] for r in range(n_rounds)]
    votes = {v["name"]: v["target"] for v in g["votes"]}
    return {
        "roles": g["roles"], "turnip": by_role["Turnip"], "carrot": by_role["Carrot"],
        "parsnip": by_role["Parsnip"], "orders": g["orders"], "qa_targets": targets,
        "votes": votes, "winner": g["winner"], "outcome": g["outcome"],
        "carrot_team_win": g["outcome"] == "carrot_win",
        "silent_turns": silent,
        "qa_attempts_max": max(q["attempts"] for q in qa),
    }


def count(batches: list[dict]) -> tuple[int, int]:
    """(played, valid) — what play_games.py uses for the replacement rule."""
    cfg = load_config()
    from game_config import GameConfig  # noqa: E402
    from common import REPO
    played = games_of(batches)
    gc = GameConfig.from_yaml(REPO / cfg.game_configs["interrog"])   # both arms: same rounds / exchanges
    valid = 0
    for _, _, g in played:
        try:
            check_game(g, gc.num_rounds_discussion, gc.interrogation.exchanges)
            valid += 1
        except GameExcluded:
            pass
    return len(played), valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    from game_config import GameConfig  # noqa: E402
    from common import REPO
    gc = GameConfig.from_yaml(REPO / cfg.game_configs[args.arm])
    t = tag(args.arm, args.pilot)
    batches = json.loads((OUT / f"summary_{t}.json").read_text())
    games, qa_rows, transcript, excluded = [], [], [], []
    for idx, seed, g in games_of(batches):
        ident = {"game_id": f"g{idx:03d}", "game_idx": idx, "game_seed": seed, "arm": t}
        try:
            meta = check_game(g, gc.num_rounds_discussion, gc.interrogation.exchanges)
        except GameExcluded as exc:
            excluded.append({**ident, "reason": str(exc)})
            continue
        games.append({**ident, **meta})
        qa_rows += [{**ident, **q} for q in g["qa"]]
        transcript += [{**ident, "role": g["roles"][m["speaker"]], **m} for m in g["announcements"]]
    write_jsonl(OUT / f"qa_{t}.jsonl", qa_rows)
    write_jsonl(OUT / f"transcripts_{t}.jsonl", transcript)
    (OUT / f"games_{t}.json").write_text(json.dumps(
        {"arm": t, "game_config": cfg.game_configs[args.arm], "games": games}, indent=1) + "\n")
    (OUT / f"excluded_{t}.json").write_text(json.dumps(excluded, indent=1) + "\n")
    print(f"{len(games)} valid games ({len(qa_rows)} QA answers), {len(excluded)} excluded")
    for x in excluded:
        print("  excluded", x["game_id"], "-", x["reason"])


if __name__ == "__main__":
    main()
