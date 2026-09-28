"""Analysis of the LOCAL repeated-seed auditor run (auditor_seeds_{arm}.jsonl).

Same stack as the games themselves (local vLLM), 50 games x 10 seeds,
conditions none + godmode: does a genuinely informative detector (the
god-mode text judge, parsnip-vs-carrot AUROC ~1.0) rescue the auditor?
Game-cluster bootstrap CIs; paired delta godmode - none per arm.

Run:  uv run --no-sync python experiments/2026-09-25/analyze_local_seeds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
sys.path.insert(0, str(OUT))

from analyze_auditor_api import (COND_COLOR, COND_LABEL, cluster_ci,  # noqa: E402
                                 per_game_matrix, slide_figure)
from common import read_jsonl  # noqa: E402

sys.path.insert(0, str(OUT.parents[0] / "2026-09-22"))

ARMS = ["base", "steered"]
CONDS = ["none", "godmode"]


def main():
    rng = np.random.default_rng(0)
    results = {"conditions": CONDS, "chance": 0.2, "arms": {}}
    pooled_gm = {c: [] for c in CONDS}
    for arm in ARMS:
        recs = read_jsonl(OUT / f"auditor_seeds_{arm}.jsonl")
        results["arms"][arm] = {"n_audits": len(recs), "conditions": {}}
        mats = {c: per_game_matrix(recs, c) for c in CONDS}
        for c in CONDS:
            gm = mats[c].mean(axis=1)
            pooled_gm[c].append(gm)
            d = gm - mats["none"].mean(axis=1)
            idx = rng.integers(0, len(gm), (10_000, len(gm)))
            results["arms"][arm]["conditions"][c] = {
                "n_games": int(mats[c].shape[0]), "n_seeds": int(mats[c].shape[1]),
                "accuracy": float(mats[c].mean()),
                "ci95_cluster": cluster_ci(gm, rng),
                "majority_vote_accuracy": float((gm > 0.5).mean()),
                "delta_vs_none": float(d.mean()),
                "delta_ci95": [float(q) for q in
                               np.quantile(d[idx].mean(axis=1), [0.025, 0.975])],
            }

    pooled = {}
    for c in CONDS:
        gm = np.concatenate(pooled_gm[c])
        lo, hi = cluster_ci(gm, rng)
        pooled[c] = (float(gm.mean()), lo, hi)
    results["pooled"] = {c: {"accuracy": pooled[c][0],
                             "ci95_cluster": list(pooled[c][1:])} for c in CONDS}

    n = results["arms"]["base"]["conditions"]["none"]
    slide_figure(pooled,
                 f"local vLLM auditor, {n['n_games']}+{n['n_games']} games x "
                 f"{n['n_seeds']} seeds per condition, arms pooled - "
                 "95% game-cluster bootstrap CI",
                 FIG / "slide_godmode_vs_none_local.png",
                 labels=COND_LABEL, colors=COND_COLOR, bar_width=0.45)

    (OUT / "results_local_seeds.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
