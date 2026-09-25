"""Aggregate auditor accuracy per condition (and arm), probe diagnostics, figure.

Reads auditor_{arm}.jsonl + scores_{arm}.csv for every arm present, writes
results.json and figures/auditor_accuracy.png.

Run:  uv run --no-sync python experiments/2026-09-22/analyze.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, read_jsonl

FIG = OUT / "figures"
CONDITIONS = ["none", "lr", "dom", "random"]
COND_LABELS = {"none": "no probe", "lr": "logistic regression",
               "dom": "diff of means", "random": "random probe"}

# chart chrome: same reference-palette instance as experiments/2026-09-20
SURFACE, INK, SEC, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SERIES = {"base": "#2a78d6", "steered": "#1baf7a"}   # categorical slots 1-2


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def mcnemar_vs_none(rows: list[dict], cond: str) -> dict:
    """Exact McNemar (binomial) on games where cond and 'none' disagree."""
    from scipy.stats import binomtest

    by_game = {r["game_id"]: r["correct"] for r in rows if r["condition"] == "none"}
    b = c = 0
    for r in rows:
        if r["condition"] != cond or r["game_id"] not in by_game:
            continue
        if r["correct"] and not by_game[r["game_id"]]:
            b += 1                      # cond right, none wrong
        elif not r["correct"] and by_game[r["game_id"]]:
            c += 1
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"cond_right_none_wrong": b, "cond_wrong_none_right": c, "p_value": float(p)}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")


def analyze_arm(arm: str) -> dict:
    rows = read_jsonl(OUT / f"auditor_{arm}.jsonl")
    out = {"n_games": len({r["game_id"] for r in rows}), "conditions": {}}
    for cond in CONDITIONS:
        cr = [r for r in rows if r["condition"] == cond]
        if not cr:
            continue
        k, n = sum(r["correct"] for r in cr), len(cr)
        by_seed = defaultdict(lambda: [0, 0])
        for r in cr:
            g = by_seed[r["game_id"].split("g")[0]]
            g[0] += r["correct"]
            g[1] += 1
        out["conditions"][cond] = {
            "correct": k, "n": n, "accuracy": k / n, "wilson_95": wilson_ci(k, n),
            "per_seed": {s: f"{v[0]}/{v[1]}" for s, v in sorted(by_seed.items())},
            "parse_failures": sum(r["parse_failed"] for r in cr),
            "mcnemar_vs_none": mcnemar_vs_none(rows, cond) if cond != "none" else None,
        }

    # probe diagnostics: can each probe separate parsnip from carrot messages?
    with open(OUT / f"scores_{arm}.csv") as f:
        srows = list(csv.DictReader(f))
    y = np.array([r["role"] == "Parsnip" for r in srows])
    out["message_probe_diagnostics"] = {}
    for col in ("lr_pub", "lr_full", "dom_pub", "dom_full", "rand_pub"):
        s = np.array([float(r[col]) for r in srows])
        out["message_probe_diagnostics"][col] = {
            "auroc_parsnip_vs_carrot": auroc(s, y),
            "mean_parsnip": float(s[y].mean()), "mean_carrot": float(s[~y].mean()),
        }
    return out


def figure(results: dict):
    import matplotlib.pyplot as plt

    arms = list(results)
    conds = [c for c in CONDITIONS if c in results[arms[0]]["conditions"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=SEC)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    width = 0.32 if len(arms) > 1 else 0.5
    for a, arm in enumerate(arms):
        xs, accs, lo, hi = [], [], [], []
        for i, cond in enumerate(conds):
            c = results[arm]["conditions"][cond]
            xs.append(i + (a - (len(arms) - 1) / 2) * (width + 0.03))
            accs.append(c["accuracy"])
            lo.append(c["accuracy"] - c["wilson_95"][0])
            hi.append(c["wilson_95"][1] - c["accuracy"])
        ax.bar(xs, accs, width=width, color=SERIES[arm], label=arm, zorder=2)
        ax.errorbar(xs, accs, yerr=[lo, hi], fmt="none", ecolor=SEC,
                    elinewidth=1.2, capsize=3, zorder=3)
        for x, acc in zip(xs, accs):
            ax.text(x, acc + 0.02, f"{acc:.2f}", ha="center", color=SEC, fontsize=8)
    ax.axhline(0.2, color=BASELINE, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(-0.55, 0.215, "chance (1/5)", color=MUTED, fontsize=8, ha="left")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([COND_LABELS[c] for c in conds])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("auditor accuracy (100 games)", color=SEC)
    ax.set_title("Auditor Parsnip identification by detector condition", color=INK, loc="left")
    if len(arms) > 1:
        ax.legend(frameon=False, labelcolor=SEC, loc="upper right")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "auditor_accuracy.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    results = {}
    for arm in ("base", "steered"):
        if (OUT / f"auditor_{arm}.jsonl").exists():
            results[arm] = analyze_arm(arm)
    if not results:
        raise SystemExit("no auditor results found")
    figure(results)
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    for arm, r in results.items():
        print(f"== {arm} ({r['n_games']} games) ==")
        for cond, c in r["conditions"].items():
            mc = c["mcnemar_vs_none"]
            extra = f"  p={mc['p_value']:.3f} vs none" if mc else ""
            print(f"  {COND_LABELS[cond]:22s} {c['correct']:3d}/{c['n']} "
                  f"({c['accuracy']:.2f}, CI {c['wilson_95'][0]:.2f}-{c['wilson_95'][1]:.2f})"
                  f"{extra}")
    print("wrote", OUT / "results.json", "and", FIG / "auditor_accuracy.png")


if __name__ == "__main__":
    main()
