"""Auditor accuracy: calibrated 1-7 grades (this experiment) vs raw scores
(2026-09-22 baseline), per arm x condition, with Wilson 95% CIs.

Writes results.json and figures/auditor_accuracy_grades_vs_raw.png.

Run:  uv run --no-sync python experiments/2026-09-25/analyze_auditor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
PREV = OUT.parents[0] / "2026-09-22"
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(PREV))

sys.path.insert(0, str(OUT.parents[1]))

from analyze_judge import BASELINE, GRID, INK, MUTED, SEC, SURFACE, style_axis  # noqa: E402
from common import read_jsonl  # noqa: E402

from auditor import AuditorConfig  # noqa: E402

ARMS = ["base", "steered"]
RAW_C, GRADE_C = "#898781", "#2a78d6"          # presentation is the entity
CHANCE = 0.2                                   # 1 parsnip among 5 players


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def seed_group_of(game_id: str) -> int:
    return int(game_id[1:game_id.index("g")])


def accuracy(path: Path, conditions) -> dict:
    recs = read_jsonl(path)
    out = {}
    for cond in conditions:
        r = [x for x in recs if x["condition"] == cond]
        k, n = sum(x["correct"] for x in r), len(r)
        lo, hi = wilson(k, n)
        groups = sorted({seed_group_of(x["game_id"]) for x in r})
        by_seed = {g: [x["correct"] for x in r if seed_group_of(x["game_id"]) == g]
                   for g in groups}
        out[cond] = {"n": n, "correct": k, "accuracy": k / n if n else None,
                     "ci95": [lo, hi],
                     "accuracy_by_seed_group": {g: float(np.mean(v))
                                                for g, v in by_seed.items()},
                     "parse_failed": sum(x["parse_failed"] for x in r)}
    return out


ARM_LABEL = {"base": "non-steered", "steered": "steered"}
ARM_COLOR = {"base": "#2a78d6", "steered": "#d03b3b"}


def grades_figure(results: dict, path: Path):
    """Grades auditor only: accuracy per condition, non-steered vs steered,
    Wilson 95% CI whiskers + per-seed-group accuracies (n=20 each) as dots."""
    import matplotlib.pyplot as plt

    conditions = results["conditions"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)
    x = np.arange(len(conditions))
    w = 0.36
    rng = np.random.default_rng(7)
    for off, arm in enumerate(ARMS):
        a = results["arms"][arm]["grades"]
        pos = x + (off - 0.5) * w
        acc = [a[c]["accuracy"] for c in conditions]
        err = [[a[c]["accuracy"] - a[c]["ci95"][0] for c in conditions],
               [a[c]["ci95"][1] - a[c]["accuracy"] for c in conditions]]
        ax.bar(pos, acc, w * 0.94, color=ARM_COLOR[arm], alpha=0.55,
               label=ARM_LABEL[arm], zorder=2)
        ax.errorbar(pos, acc, yerr=err, fmt="none", ecolor=SEC,
                    elinewidth=1.1, capsize=3, zorder=4)
        for i, c in enumerate(conditions):
            seeds = list(a[c]["accuracy_by_seed_group"].values())
            jitter = rng.uniform(-w * 0.28, w * 0.28, len(seeds))
            ax.scatter(pos[i] + jitter, seeds, s=14, color=ARM_COLOR[arm],
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)
    ax.axhline(CHANCE, color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(len(conditions) - 0.55, CHANCE + 0.015, "chance", color=MUTED,
            fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("auditor accuracy", color=SEC)
    ax.set_title("Auditor accuracy with 1-7 suspicion grades\n"
                 "bars: 100 games, Wilson 95% CI · dots: seed groups (20 games each)",
                 color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right", labelcolor=SEC)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("figure ->", path)


def main():
    cfg = AuditorConfig.load(OUT / "config.json")
    conditions = cfg.conditions
    results = {"conditions": conditions, "chance": CHANCE, "arms": {}}
    for arm in ARMS:
        results["arms"][arm] = {
            "grades": accuracy(OUT / f"auditor_{arm}.jsonl", conditions),
            "raw_scores_2026_09_22": accuracy(PREV / f"auditor_{arm}.jsonl", conditions),
        }

    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(conditions))
    w = 0.36
    for ax, arm in zip(axes, ARMS):
        style_axis(ax)
        for off, (key, label, color) in enumerate(
                [("raw_scores_2026_09_22", "raw scores (09-22)", RAW_C),
                 ("grades", "1-7 grades (09-25)", GRADE_C)]):
            a = results["arms"][arm][key]
            acc = [a[c]["accuracy"] for c in conditions]
            err = [[a[c]["accuracy"] - a[c]["ci95"][0] for c in conditions],
                   [a[c]["ci95"][1] - a[c]["accuracy"] for c in conditions]]
            ax.bar(x + (off - 0.5) * w, acc, w * 0.94, color=color, alpha=0.75,
                   label=label, zorder=2)
            ax.errorbar(x + (off - 0.5) * w, acc, yerr=err, fmt="none",
                        ecolor=SEC, elinewidth=1, capsize=2.5, zorder=3)
        ax.axhline(CHANCE, color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_title(f"{arm} arm", color=INK, loc="left")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("auditor accuracy (n=100 games)", color=SEC)
    axes[0].text(len(conditions) - 0.55, CHANCE + 0.015, "chance",
                 color=MUTED, fontsize=8, ha="right")
    axes[1].legend(frameon=False, loc="upper right", labelcolor=SEC)
    fig.tight_layout()
    path = FIG / "auditor_accuracy_grades_vs_raw.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("figure ->", path)

    grades_figure(results, FIG / "auditor_accuracy_grades.png")

    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
