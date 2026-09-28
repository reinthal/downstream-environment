"""Analysis of the god-mode text lie detector (judge_godmode_{arm}.jsonl).

Per arm: parsnip-vs-carrot AUROC of the 1-7 score, per-role score
distributions, agreement with the DeepSeek oracle judge (role revealed), and
head-to-head with the activation probes on the same messages. Figure: score
distribution by role per arm (discrete 1-7, grouped bars).

Run:  uv run --no-sync python experiments/2026-09-25/analyze_godmode.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
PREV = OUT.parents[0] / "2026-09-22"
sys.path.insert(0, str(PREV))

from analyze_judge import (DECEPTIVE_C, HONEST_C, INK, MUTED, SEC, SURFACE,  # noqa: E402
                           style_axis)
from common import read_jsonl  # noqa: E402

ARMS = ["base", "steered"]


def main():
    results = {}
    dists = {}
    for arm in ARMS:
        god = read_jsonl(OUT / f"judge_godmode_{arm}.jsonl")
        god = [g for g in god if g["judge_score"] is not None]
        key = lambda r: (r["game_id"], r["round"], r["position"])  # noqa: E731
        gmap = {key(g): g for g in god}
        y_role = np.array([g["role"] == "Parsnip" for g in god])
        s = np.array([g["judge_score"] for g in god], dtype=float)
        res = {"n_scored": len(god),
               "auroc_parsnip_vs_carrot": float(roc_auc_score(y_role, s)),
               "mean_score_by_role": {r: float(s[[g["role"] == r for g in god]].mean())
                                      for r in ("Carrot", "Parsnip")},
               "label_by_role": {r: dict(Counter(g["judge_label"] for g in god
                                                 if g["role"] == r))
                                 for r in ("Carrot", "Parsnip")}}

        # vs oracle judge (knows the role)
        oracle = {key(o): o for o in read_jsonl(PREV / f"judge_{arm}_api_oracle.jsonl")}
        pairs = [(gmap[k]["judge_score"], oracle[k]) for k in gmap if k in oracle]
        ol = [o["judge_label"] for _, o in pairs]
        gs = np.array([g for g, _ in pairs], dtype=float)
        m = np.array([l in ("honest", "deceptive") for l in ol])
        yo = np.array([l == "deceptive" for l in ol])[m]
        res["auroc_vs_oracle_honest_deceptive"] = float(roc_auc_score(yo, gs[m]))
        res["spearman_vs_oracle_score"] = float(
            np.corrcoef(np.argsort(np.argsort(gs)),
                        np.argsort(np.argsort([o["judge_score"] for _, o in pairs])))[0, 1])

        # head-to-head with probes on the same messages
        with open(PREV / f"scores_{arm}.csv") as f:
            rows = list(csv.DictReader(f))
        probe = {}
        for p in ("lr", "dom"):
            sp = np.array([float(r[f"{p}_full"]) for r in rows
                           if (r["game_id"], int(r["round"]), int(r["position"])) in gmap])
            yp = np.array([r["role"] == "Parsnip" for r in rows
                           if (r["game_id"], int(r["round"]), int(r["position"])) in gmap])
            probe[p] = float(roc_auc_score(yp, sp))
        res["probe_auroc_same_messages"] = probe
        results[arm] = res
        dists[arm] = {r: Counter(g["judge_score"] for g in god if g["role"] == r)
                      for r in ("Carrot", "Parsnip")}

    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(1, 8)
    w = 0.4
    for ax, arm in zip(axes, ARMS):
        style_axis(ax)
        for off, (role, color) in enumerate([("Carrot", HONEST_C),
                                             ("Parsnip", DECEPTIVE_C)]):
            c = dists[arm][role]
            n = sum(c.values())
            ax.bar(x + (off - 0.5) * w, [c.get(k, 0) / n for k in x], w * 0.92,
                   color=color, alpha=0.75, label=role, zorder=2)
        ax.set_xticks(x)
        ax.set_xlabel("god-mode suspicion score", color=SEC)
        auroc = results[arm]["auroc_parsnip_vs_carrot"]
        ax.set_title(f"{'non-steered' if arm == 'base' else 'steered'}  "
                     f"(AUROC {auroc:.3f})", color=INK, loc="left")
    axes[0].set_ylabel("fraction of role's messages", color=SEC)
    axes[1].legend(frameon=False, labelcolor=SEC)
    fig.suptitle("God-mode text lie detector: think + announcement, no role access",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG / "godmode_score_by_role.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("figure ->", path)

    (OUT / "results_godmode.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
