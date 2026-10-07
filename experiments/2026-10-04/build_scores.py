"""Raw per-message detector scores for the analysing auditor (CPU).

For each detector, every base game's 15 messages with round, speaking
position, player and the detector's raw score (log-odds; higher = more likely
deceptive). Nothing else: no other games, no roles, no processing.

  raw_fu   follow-up probe (2026-10-01)
  raw_lr   instructed-pairs LR probe (2026-09-22)
  raw_rand random direction on the follow-up activations (seed 0; null)

Output: raw_{fu,lr,rand}_base.json ({game_id: [{round, position, player, score}]})

Run:  .venv/bin/python experiments/2026-10-04/build_scores.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "2026-10-03"))
from data import messages, scores  # noqa: E402  (2026-10-03 loaders)

DETECTORS = {"raw_fu": ("fu", {}), "raw_lr": ("lr", {}), "raw_rand": ("rand_fu", {"seed": 0})}


def main():
    msgs = messages("base")
    for name, (det, kw) in DETECTORS.items():
        s = scores(det, "base", **kw)
        games = defaultdict(list)
        for m, v in zip(msgs, s):
            games[m["game_id"]].append({"round": m["round"] + 1, "position": m["position"] + 1,
                                        "player": m["speaker"], "score": round(float(v), 2)})
        for rows in games.values():
            rows.sort(key=lambda r: (r["round"], r["position"]))
        (OUT / f"{name}_base.json").write_text(json.dumps(games, indent=1) + "\n")
        print(name, len(games), "games; example s0g0:", games["s0g0"][:3])


if __name__ == "__main__":
    main()
