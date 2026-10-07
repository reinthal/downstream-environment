"""Results of the analysing auditor (CPU): raw detector scores, with / without
a Python tool, this game's data only.

Per condition: accuracy (95% game-cluster CI); the detector-only benchmarks on
the same games (highest raw average; highest average after a within-game
speaking-position correction); how often the auditor picks the raw-average
top player; overrides; tool use and whether its code looked at position/round.

Output: results.json

Run:  .venv/bin/python experiments/2026-10-04/analyze_audits.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
RAW = {"fu_tool": "raw_fu", "fu_notool": "raw_fu", "lr_tool": "raw_lr", "rand_tool": "raw_rand"}


def ci(rows, n=2000):
    by = defaultdict(list)
    for r in rows:
        by[r["game_id"]].append(r["correct"])
    s = np.array([sum(v) for v in by.values()])
    c = np.array([len(v) for v in by.values()])
    idx = np.random.default_rng(0).integers(0, len(s), (n, len(s)))
    return [round(float(x), 3) for x in np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])]


def picks(rows):
    """raw-average top player and within-game position-corrected top player"""
    raw = defaultdict(list)
    for r in rows:
        raw[r["player"]].append(r["score"])
    pos = defaultdict(list)
    for r in rows:
        pos[r["position"]].append(r["score"])
    corr = defaultdict(list)
    for r in rows:
        corr[r["player"]].append(r["score"] - np.mean(pos[r["position"]]))
    top = lambda d: max(d, key=lambda p: np.mean(d[p]))  # noqa: E731
    return top(raw), top(corr)


def main():
    rows = [json.loads(l) for d in ("audit_a", "audit_b") for l in open(OUT / d / "auditor_base.jsonl")]
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    out = {}
    for cond, rs in by.items():
        d = {"n": len(rs), "accuracy": round(float(np.mean([r["correct"] for r in rs])), 3), "ci95": ci(rs)}
        if cond in RAW:
            data = json.loads((OUT / f"{RAW[cond]}_base.json").read_text())
            pk = {g: picks(v) for g, v in data.items()}
            par = {r["game_id"]: r["parsnip"] for r in rs}
            d["rule_raw_average"] = round(float(np.mean([pk[g][0] == par[g] for g in par])), 3)
            d["rule_position_corrected"] = round(float(np.mean([pk[g][1] == par[g] for g in par])), 3)
            follow = [r["guess"] == pk[r["game_id"]][0] for r in rs]
            d["picks_raw_average_top"] = round(float(np.mean(follow)), 3)
            over = [r for r, f in zip(rs, follow) if not f]
            d["overrides"] = len(over)
            d["overrides_correct"] = int(sum(r["correct"] for r in over))
            d["overrides_where_rule_was_right"] = int(sum(pk[r["game_id"]][0] == r["parsnip"] for r in over))
            tools = [r["tool"] for r in rs if r.get("tool")]
            if tools:
                codes = [" ".join(c["code"] for c in t["tool_calls"]) for t in tools]
                d["mean_tool_calls"] = round(float(np.mean([t["n_tool_calls"] for t in tools])), 2)
                d["share_no_tool_call"] = round(float(np.mean([t["n_tool_calls"] == 0 for t in tools])), 3)
                d["code_groups_by_position"] = round(float(np.mean(
                    [bool(re.search(r"groupby\(\s*['\"]position|\['position'\]|\.position", c)) for c in codes])), 3)
                d["code_groups_by_round"] = round(float(np.mean(
                    [bool(re.search(r"groupby\(\s*['\"]round|\['round'\]\s*==", c)) for c in codes])), 3)
                d["code_errors"] = round(float(np.mean(
                    [any("Traceback" in c["output"] or "[error]" in c["output"] for c in t["tool_calls"])
                     for t in tools])), 3)
        out[cond] = d
    (OUT / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    for c, d in out.items():
        print(c, json.dumps(d))


if __name__ == "__main__":
    main()
