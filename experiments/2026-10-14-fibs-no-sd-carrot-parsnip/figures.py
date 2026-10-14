"""figures/auroc.svg from results.json: pooled announcement AUROC per
condition (seed dots, mean with 95% game interval; published as reference),
linear and MLP panels on the same y range; right panel the gap with its
interval against 0 and cfg.delta, both archs on one axis.

Run:  uv run --no-sync python experiments/2026-10-14-fibs-no-sd-carrot-parsnip/figures.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT, load_config  # noqa: E402

LABELS = {"no_sd": "no SD / negotiation", "random_drop": "random drop", "published": "published (ref.)"}


def main():
    cfg = load_config()
    res = json.loads((OUT / "results.json").read_text())
    pooled = {cfg.primary_arch: res["primary"]["pooled"], **res["secondary"]["mlp_pooled"]}
    fig, axes = plt.subplots(1, len(cfg.archs) + 1, figsize=(4 * (len(cfg.archs) + 1), 3.6))
    lo_y = min(min(min(d[c]["per_seed"].values()) for c in cfg.conditions) for d in pooled.values())
    lo_y = min(lo_y, min(d["published_reference"]["ci95_game"][0] for d in pooled.values()))
    ylim = (max(0.5, lo_y - 0.01), 1.0)
    for ax, a in zip(axes, cfg.archs):
        d = pooled[a]
        for k, c in enumerate(cfg.conditions):
            seeds = list(d[c]["per_seed"].values())
            ax.scatter([k] * len(seeds), seeds, s=14, color="0.6", zorder=2)
            lo, hi = d[c]["ci95_game"]
            ax.errorbar(k + 0.15, d[c]["mean_auroc"], yerr=[[d[c]["mean_auroc"] - lo], [hi - d[c]["mean_auroc"]]],
                        fmt="o", color="C0", capsize=3, zorder=3)
        p = d["published_reference"]
        k = len(cfg.conditions)
        ax.errorbar(k, p["auroc"], yerr=[[p["auroc"] - p["ci95_game"][0]], [p["ci95_game"][1] - p["auroc"]]],
                    fmt="s", color="0.3", capsize=3)
        ax.set_xticks(range(k + 1), [LABELS[c] for c in cfg.conditions] + [LABELS["published"]], fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_title(f"{a}: announcement AUROC (3p+5p)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    ax = axes[-1]
    for k, a in enumerate(cfg.archs):
        g = pooled[a]["gap"]
        ax.errorbar(k, g["value"], yerr=[[g["value"] - g["ci95_game"][0]], [g["ci95_game"][1] - g["value"]]],
                    fmt="o", color="C0", capsize=3)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.axhline(cfg.delta, color="C3", lw=0.8, ls="--", label=f"δ = {cfg.delta}")
    ax.set_xticks(range(len(cfg.archs)), cfg.archs)
    ax.set_xlim(-0.5, len(cfg.archs) - 0.5)
    ax.set_title("gap = random drop − no SD", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "auroc.svg")
    print("wrote figures/auroc.svg")


if __name__ == "__main__":
    main()
