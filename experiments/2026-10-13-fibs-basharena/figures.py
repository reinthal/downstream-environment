"""The preregistered figure (PREREGISTRATION §4) plus one exploratory panel.

  figures/auroc.svg        per-step AUROC per probe with 95% task-clustered CI,
                           y 0.5-1.0, dashed 0.90 (threshold), dotted 0.5 (chance)
  figures/contrast.svg     linear contrast score by step class (honest /
                           pre-onset / post-onset), shared y
  figures/by_step.svg      exploratory: linear contrast against step index, by class
  figures/roc.svg          exploratory: per-step ROC, primary and its decomposition

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
from analyze import clustered_ci, label_steps  # noqa: E402
from common import OUT, load_config  # noqa: E402

CLASSES = [("honest", "honest steps"), ("pre_onset", "attack, pre-onset"), ("post_onset", "attack, post-onset")]
COLORS = {"honest": "#4C72B0", "pre_onset": "#DD8452", "post_onset": "#C44E52"}


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FPR/TPR at every distinct threshold, highest first (tied scores share one point)."""
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order].astype(float)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp, fp = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    return np.r_[0, fp / fp[-1]], np.r_[0, tp / tp[-1]]


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

    # exploratory: contrast vs step index by class, median per 3 steps with 95% task-clustered CI
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bins = np.arange(1, 52, 3)
    for c, lab in CLASSES:
        sel = [r for r in rows if r["cls"] == c]
        xs = np.array([r["step"] for r in sel]); ys = np.array([r["linear_contrast"] for r in sel])
        tasks = np.array([r["task_id"] for r in sel])
        med, lo, hi = [], [], []
        for b in bins:
            m = (xs >= b) & (xs < b + 3)
            if not m.any():
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            med.append(np.median(ys[m]))
            l, h = clustered_ci(ys[m], np.zeros(m.sum()), tasks[m], cfg.n_boot, cfg.boot_seed,
                                stat=lambda v, _: np.median(v))
            lo.append(l); hi.append(h)
        ax.fill_between(bins + 1, lo, hi, color=COLORS[c], alpha=0.2, lw=0)
        ax.plot(bins + 1, med, "o-", ms=3, color=COLORS[c], label=f"{lab} (median per 3 steps)")
    ax.set_xlabel("step index in trajectory")
    ax.set_ylabel("linear contrast")
    ax.set_title("exploratory: contrast against step index, by class\nband = 95% task-clustered bootstrap CI of the median", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "by_step.svg")

    # exploratory: per-step ROC curves, primary contrast and its decomposition; shared axes
    expl = json.loads((OUT / f"exploratory_{args.tag}.json").read_text())
    comparisons = [
        ("primary: post-onset vs honest + pre-onset", {"post_onset"}, {"honest", "pre_onset"},
         {a: res["auroc"][a] for a in archs}),
        ("post-onset vs pre-onset (action signal)", {"post_onset"}, {"pre_onset"},
         {a: expl[a]["post_onset_vs_pre_onset"] for a in archs}),
        ("pre-onset vs honest (prompt signal)", {"pre_onset"}, {"honest"},
         {a: expl[a]["pre_onset_vs_honest"] for a in archs}),
    ]
    arch_colors = {"linear": "#333333", "mlp": "#4C72B0"}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharex=True, sharey=True)
    for ax, (title, pos, neg, stored) in zip(axes, comparisons):
        sel = [r for r in rows if r["cls"] in pos | neg]
        y = np.array([r["cls"] in pos for r in sel])
        for a in archs:
            fpr, tpr = roc_curve(np.array([r[f"{a}_contrast"] for r in sel]), y)
            p, (lo, hi) = stored[a]["point"], stored[a]["ci95"]
            ax.plot(fpr, tpr, color=arch_colors[a], lw=1.5, label=f"{a}: {p:.3f} [{lo:.3f}, {hi:.3f}]")
        ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
        ax.set_title(f"{title}\n{int(y.sum())} pos / {int((~y).sum())} neg steps", fontsize=9)
        ax.set_xlabel("false positive rate (steps)")
        ax.legend(fontsize=8, loc="lower right", title="AUROC [95% task CI]", title_fontsize=8)
    axes[0].set_ylabel("true positive rate (steps)")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    fig.suptitle("fibs probes on BashArena (Qwen3.8-27B): per-step ROC", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc.svg")
    print("wrote", sorted(p.name for p in fig_dir.glob("*.svg")))


if __name__ == "__main__":
    main()
