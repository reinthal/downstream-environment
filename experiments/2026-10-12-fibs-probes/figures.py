"""Figures (PREREGISTRATION.md §4): per probe, violins of the contrast score
for alpaca / all-Carrot-game Carrots / Carrots / Parsnip, by round and by
player count, with the alpaca 1%-FPR cutpoint; and the announcement-level
AUROC of every probe per arm and pooled with 95% game-clustered intervals.
Panels meant to be compared share axis ranges.

  figures/violins_{arch}.svg, figures/auroc.svg

Run:  uv run --no-sync python experiments/2026-10-12-fibs-probes/figures.py
"""
from __future__ import annotations

import csv
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT, load_config  # noqa: E402

MAIN = ("3p", "5p")


def rows_of(arm: str) -> list[dict]:
    p = OUT / f"scores_{arm}.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def main():
    cfg = load_config()
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)
    res = json.loads((OUT / "results.json").read_text())
    alpaca = rows_of("alpaca")
    data = {arm: rows_of(arm) for arm in MAIN}
    cc = {arm: rows_of(f"{arm}_cc") for arm in MAIN}

    for arch in cfg.probe_archs:
        c = f"{arch}_contrast"
        allv = [float(r[c]) for arm in MAIN for r in data[arm] + cc[arm]] + [float(r[c]) for r in alpaca]
        lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
        pad = 0.05 * (hi - lo)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, arm in zip(axes, MAIN):
            groups, labels, colors = [], [], []
            if alpaca:
                groups.append([float(r[c]) for r in alpaca]); labels.append("alpaca"); colors.append("#999999")
            for rd in range(3):
                g = [float(r[c]) for r in cc[arm] if int(r["round"]) == rd]
                if g:
                    groups.append(g); labels.append(f"all-Carrot\nr{rd + 1}"); colors.append("#8fbc8f")
            for role, col in (("Carrot", "#1f77b4"), ("Parsnip", "#d62728")):
                for rd in range(3):
                    g = [float(r[c]) for r in data[arm] if r["role"] == role and int(r["round"]) == rd]
                    if g:
                        groups.append(g); labels.append(f"{role}\nr{rd + 1}"); colors.append(col)
            parts = ax.violinplot(groups, showmedians=True, widths=0.85)
            for body, col in zip(parts["bodies"], colors):
                body.set_facecolor(col); body.set_alpha(0.6)
            ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels(labels, fontsize=7)
            cut = res.get("secondary", {}).get("alpaca", {}).get(arch, {}).get("cutpoint_1pct_fpr")
            if cut is not None:
                ax.axhline(cut, ls="--", color="k", lw=0.8, label="alpaca 1%-FPR")
                ax.legend(loc="upper left", fontsize=7)
            ax.set_title(f"{arm}: {arch} contrast (Yes. − No.)")
            ax.set_ylim(lo - pad, hi + pad)
        axes[0].set_ylabel("probe score contrast")
        fig.tight_layout(); fig.savefig(fig_dir / f"violins_{arch}.svg"); plt.close(fig)

    # AUROC dot plot
    sec = res["secondary"]
    names = cfg.probe_archs + (["dyl"] if sec.get("dyl") else [])
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(names))
    for k, (label, off) in enumerate((("3p", -0.25), ("5p", 0.0), ("pooled", 0.25))):
        ys, los, his = [], [], []
        for n in names:
            if n == "dyl":
                e = sec["dyl"]["pooled"] if label == "pooled" else sec["dyl"]["per_arm"][label]
            else:
                e = sec["archs"][n]["contrast"] if label == "pooled" else sec["archs"][n]["per_arm"][label]
            ys.append(e["auroc"]); los.append(e["auroc"] - e["ci95_game"][0]); his.append(e["ci95_game"][1] - e["auroc"])
        ax.errorbar(xs + off, ys, yerr=[los, his], fmt="o", capsize=3, label=label)
    ax.axhline(0.5, ls=":", color="grey"); ax.axhline(cfg.auroc_threshold, ls="--", color="k", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(names); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("announcement AUROC, Parsnip vs Carrot"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "auroc.svg"); plt.close(fig)
    print(f"wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
