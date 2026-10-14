"""What the pilot must show before the non-pilot run (CPU). Fails loudly.

  - each pilot arm has at least one valid game and one scored Parsnip turn;
  - the deny line is in every deny-arm Parsnip prompt and in no other prompt;
  - pyes_{arm}.csv has one finite score per scored turn.

Run:  .venv/bin/python experiments/2026-10-13-l48-vs-selfreport/pilot_check.py
"""
from __future__ import annotations

import csv
import math

import yaml

from common import OUT, arm_yaml, load_config, scored_turns


def main():
    cfg = load_config()
    deny_line = yaml.safe_load(arm_yaml(cfg, "deny").read_text())["parsnip"]["private_knowledge"]
    for arm in ("normal_pilot", "deny_pilot"):
        turns = scored_turns(arm)
        parsnip = [r for r in turns if r["role"] == "Parsnip"]
        assert parsnip, f"{arm}: no scored Parsnip turn"
        for r in turns:
            has = deny_line in r["input_messages"][1]["content"]
            want = arm == "deny_pilot" and r["role"] == "Parsnip"
            assert has == want, f"{arm} {r['game_id']} {r['speaker']}: deny line present={has}, expected {want}"
        with open(OUT / f"pyes_{arm}.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(turns), f"{arm}: {len(rows)} P(Yes) rows for {len(turns)} turns"
        assert all(math.isfinite(float(x["score"])) for x in rows), f"{arm}: non-finite P(Yes)"
        by_role = {role: [round(float(x["score"]), 2) for x in rows if x["role"] == role] for role in ("Parsnip", "Carrot")}
        print(f"{arm}: {len(turns)} turns ok; log P(Yes) - log P(No) by role: {by_role}")
    print("pilot OK")


if __name__ == "__main__":
    main()
