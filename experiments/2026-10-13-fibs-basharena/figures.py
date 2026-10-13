"""The preregistered figure (PREREGISTRATION §4) plus one exploratory panel.

  figures/auroc.svg        per-step AUROC per probe with 95% task-clustered CI,
                           y 0.5-1.0, dashed 0.90 (threshold), dotted 0.5 (chance)
  figures/contrast.svg     linear contrast score by step class (honest /
                           pre-onset / post-onset), shared y
  figures/by_step.svg      exploratory: linear contrast against step index, by class

Run:  uv run --project environments/bash_arena python experiments/2026-10-13-fibs-basharena/figures.py --tag main
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import label_steps  # noqa: E402
from common import OUT, load_config  # noqa: E402

CLASSES = [("honest", "honest steps"), ("pre_onset", "attack, pre-onset"), ("post_onset", "attack, post-onset")]
COLORS = {"honest": "#4C72B0", "pre_onset": "#DD8452", "post_onset": "#C44E52"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    cfg = load_config()
    res = json.loads((OUT / f"results_{args.tag}.json").read_text())
    rows = label_steps(cfg, args.tag)["rows"]
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)

    # panel 1: AUROC per probe
    archs = list(res["auroc"])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for i, a in enumerate(archs):
        p, (lo, hi) = res["auroc"][a]["point"], res["auroc"][a]["ci95"]
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", color="#333333", capsize=4)
        ax.annotate(f"{p:.3f}", (i, p), textcoords="offset points", xytext=(8, 0), fontsize=9, va="center")
    ax.axhline(cfg.auroc_threshold, ls="--", color="grey", lw=1, label="preregistered threshold 0.90")
    ax.axhline(0.5, ls=":", color="grey", lw=1, label="chance")
    ax.set_xticks(range(len(archs)), [f"{a} ensemble" for a in archs])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("per-step AUROC (post-onset vs honest + pre-onset)")
    ax.set_title(f"fibs probes on BashArena, Qwen3.8-27B\n{res['positives']} positive / {res['negatives']} negative steps, 47 tasks", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "auroc.svg")

    # panel 2: contrast by class (shared y)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    data = [[r["linear_contrast"] for r in rows if r["cls"] == c] for c, _ in CLASSES]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for body, (c, _) in zip(parts["bodies"], CLASSES):
        body.set_facecolor(COLORS[c]); body.set_alpha(0.6)
    ax.set_xticks([1, 2, 3], [f"{lab}\n(n={len(d)})" for (_, lab), d in zip(CLASSES, data)])
    ax.set_ylabel("linear contrast (No. − Yes.)")
    ax.set_title("linear ensemble contrast by step class", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "contrast.svg")

    # exploratory: contrast vs step index by class
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for c, lab in CLASSES:
        xs = np.array([r["step"] for r in rows if r["cls"] == c]); ys = np.array([r["linear_contrast"] for r in rows if r["cls"] == c])
        bins = np.arange(1, 52, 3)
        med = [np.median(ys[(xs >= b) & (xs < b + 3)]) if ((xs >= b) & (xs < b + 3)).any() else np.nan for b in bins]
        ax.plot(bins + 1, med, "o-", ms=3, color=COLORS[c], label=f"{lab} (median per 3 steps)")
    ax.set_xlabel("step index in trajectory")
    ax.set_ylabel("linear contrast")
    ax.set_title("exploratory: contrast against step index, by class", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "by_step.svg")
    print("wrote", sorted(p.name for p in fig_dir.glob("*.svg")))


if __name__ == "__main__":
    main()
