"""Step 3 results: auditor accuracy with per-game detector summaries (CPU).

Pools audit_a and audit_b (2 seeds each -> 4 audits per game x condition).
For each condition: accuracy with a 95% game-cluster bootstrap CI, the
detector's own top-1 accuracy on the same games, how often the auditor picks
the detector's top player, and how its overrides turn out.

Output: step3.json

Run:  .venv/bin/python experiments/2026-10-03/analyze_audits.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from data import OUT

SRC = {"fu_prob": "posteriors_fu_base.json", "lr_prob": "posteriors_lr_base.json",
       "rand_prob": "posteriors_rand_fu_base.json", "godmode_prob": "posteriors_godmode_base.json"}


def ci(rows, n=2000):
    by = defaultdict(list)
    for r in rows:
        by[r["game_id"]].append(r["correct"])
    s = np.array([sum(v) for v in by.values()])
    c = np.array([len(v) for v in by.values()])
    idx = np.random.default_rng(0).integers(0, len(s), (n, len(s)))
    return [round(float(x), 3) for x in np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])]


def main():
    rows = [json.loads(l) for d in ("audit_a", "audit_b")
            for l in open(OUT / d / "auditor_seeds_base.jsonl")]
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    out = {}
    for cond, rs in by.items():
        d = {"n": len(rs), "accuracy": round(float(np.mean([r["correct"] for r in rs])), 3),
             "ci95": ci(rs), "cut_off": round(float(np.mean([r["force_closed"] for r in rs])), 3)}
        if cond in SRC:
            post = json.loads((OUT / SRC[cond]).read_text())
            top = {g: max(p, key=p.get) for g, p in post.items()}
            par = {r["game_id"]: r["parsnip"] for r in rs}
            follow = [r for r in rs if r["guess"] == top[r["game_id"]]]
            over = [r for r in rs if r["guess"] != top[r["game_id"]]]
            d.update({
                "detector_alone_top1": round(float(np.mean([top[g] == par[g] for g in par])), 3),
                "follows_detector_top": round(len(follow) / len(rs), 3),
                "overrides": len(over),
                "overrides_correct": int(sum(r["correct"] for r in over)),
                "overrides_where_detector_was_right": int(sum(top[r["game_id"]] == r["parsnip"] for r in over)),
            })
        out[cond] = d
    (OUT / "step3.json").write_text(json.dumps(out, indent=1) + "\n")
    for c, d in out.items():
        extra = (f" | detector alone {d['detector_alone_top1']:.2f} | follows top pick {d['follows_detector_top']:.0%}"
                 f" | overrides {d['overrides']} (right {d['overrides_correct']}, detector was right in "
                 f"{d['overrides_where_detector_was_right']})") if "detector_alone_top1" in d else ""
        print(f"{c:13s} acc {d['accuracy']:.3f} {d['ci95']} cut-off {d['cut_off']:.0%}{extra}")


if __name__ == "__main__":
    main()
