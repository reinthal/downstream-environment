"""Analysis of the OpenRouter repeated-seed auditor runs.

auditor_api_{arm}.jsonl: 50 games x conditions x 10 auditor seeds. Seeds
within a game share the transcript, so they are correlated — the game is the
sampling unit. Per arm x condition:
  * mean accuracy over all audits, with a game-level cluster-bootstrap 95% CI
  * majority-vote accuracy (per game: correct iff >5/10 seeds correct)
  * paired delta vs the "none" condition (same games, cluster bootstrap)

Figure: bars (mean accuracy + cluster CI) per condition, non-steered vs
steered; small dots = per-game mean over its 10 seeds (50 per bar).

Run:  uv run --no-sync python experiments/2026-09-25/analyze_auditor_api.py
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

from analyze_judge import INK, MUTED, SEC, SURFACE, style_axis  # noqa: E402
from common import read_jsonl  # noqa: E402

ARMS = ["base", "steered"]
ARM_LABEL = {"base": "non-steered", "steered": "steered"}
ARM_COLOR = {"base": "#2a78d6", "steered": "#d03b3b"}
CHANCE = 0.2
N_BOOT = 10_000


def per_game_matrix(recs: list[dict], cond: str) -> np.ndarray:
    """games x seeds matrix of 0/1 correctness for one condition."""
    by_game = {}
    for r in recs:
        if r["condition"] == cond:
            by_game.setdefault(r["game_id"], {})[r["auditor_seed"]] = int(r["correct"])
    gids = sorted(by_game)
    n_seeds = max(len(v) for v in by_game.values())
    return np.array([[by_game[g].get(s, 0) for s in range(n_seeds)] for g in gids])


def cluster_ci(game_means: np.ndarray, rng) -> list[float]:
    idx = rng.integers(0, len(game_means), (N_BOOT, len(game_means)))
    boot = game_means[idx].mean(axis=1)
    return [float(q) for q in np.quantile(boot, [0.025, 0.975])]


COND_LABEL = {"none": "no detector", "lr": "LR probe",
              "dom": "diff-of-means", "random": "random probe",
              "godmode": "god-mode judge"}
COND_COLOR = {"none": "#898781", "lr": "#2a78d6",
              "dom": "#1baf7a", "random": "#c2963f",
              "godmode": "#8a5fd0"}


def slide_figure(pooled: dict, subtitle: str, path: Path,
                 labels: dict | None = None, colors: dict | None = None,
                 bar_width: float = 0.6):
    """One slide-ready 16:9 chart. pooled: key -> (acc, ci_lo, ci_hi);
    keys default to detector conditions, override labels/colors for other
    groupings (e.g. arms)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    labels = labels or COND_LABEL
    colors = colors or COND_COLOR
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#eef3fa")                     # light panel behind the bars
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.yaxis.grid(True, color="#ffffff", linewidth=1.4)
    ax.set_axisbelow(True)
    conds = list(pooled)
    for i, c in enumerate(conds):
        acc, lo, hi = pooled[c]
        ax.bar(i, acc, bar_width, color=colors[c], alpha=0.85, zorder=2)
        ax.errorbar(i, acc, yerr=[[acc - lo], [hi - acc]], fmt="none",
                    ecolor="#0b0b0b", elinewidth=1.8, capsize=6, zorder=4)
        ax.text(i + 0.06, hi + 0.008, f"{100 * acc:.1f}%", ha="left",
                va="bottom", color="#0b0b0b", fontsize=17)
    ax.axhline(0.2, color="#898781", linewidth=1.3, linestyle=(0, (4, 4)))
    ax.text(len(conds) - 0.52, 0.207, "chance (1 of 5)", color="#898781",
            fontsize=13, ha="right")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([labels[c] for c in conds], fontsize=17, color="#0b0b0b")
    top = max(0.55, max(hi for _, _, hi in pooled.values()) + 0.12)
    ax.set_yticks([t for t in (0, 0.25, 0.5, 0.75, 1.0) if t <= top])
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(labelsize=15, colors="#52514e", length=0)
    ax.set_title("% auditor identifies the liar (higher better)",
                 color="#0b0b0b", loc="left", fontsize=21, pad=40)
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, color="#52514e",
            fontsize=13, va="bottom")
    fig.tight_layout()
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    print("figure ->", path)


def main():
    rng = np.random.default_rng(0)
    conditions = ["none", "lr", "dom", "random", "godmode"]
    results = {"conditions": conditions, "chance": CHANCE,
               "note": "cluster bootstrap over games (seeds within a game are correlated)",
               "arms": {}}
    mats = {}
    for arm in ARMS:
        recs = read_jsonl(OUT / f"auditor_api_{arm}.jsonl")
        model = recs[0].get("judge_model", "?")
        results["arms"][arm] = {"model": model, "n_audits": len(recs), "conditions": {}}
        for cond in conditions:
            M = mats[(arm, cond)] = per_game_matrix(recs, cond)
            gm = M.mean(axis=1)
            d = gm - mats[(arm, "none")].mean(axis=1)
            idx = rng.integers(0, len(gm), (N_BOOT, len(gm)))
            results["arms"][arm]["conditions"][cond] = {
                "n_games": int(M.shape[0]), "n_seeds": int(M.shape[1]),
                "accuracy": float(M.mean()),
                "ci95_cluster": cluster_ci(gm, rng),
                "majority_vote_accuracy": float((gm > 0.5).mean()),
                "delta_vs_none": float(d.mean()),
                "delta_ci95": [float(q) for q in np.quantile(d[idx].mean(axis=1),
                                                             [0.025, 0.975])],
                "parse_failed": sum(r["parse_failed"] for r in recs
                                    if r["condition"] == cond),
            }

    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)
    x = np.arange(len(conditions))
    w = 0.36
    for off, arm in enumerate(ARMS):
        pos = x + (off - 0.5) * w
        cs = results["arms"][arm]["conditions"]
        acc = [cs[c]["accuracy"] for c in conditions]
        err = [[cs[c]["accuracy"] - cs[c]["ci95_cluster"][0] for c in conditions],
               [cs[c]["ci95_cluster"][1] - cs[c]["accuracy"] for c in conditions]]
        ax.bar(pos, acc, w * 0.94, color=ARM_COLOR[arm], alpha=0.55,
               label=ARM_LABEL[arm], zorder=2)
        ax.errorbar(pos, acc, yerr=err, fmt="none", ecolor=SEC, elinewidth=1.1,
                    capsize=3, zorder=4)
        for i, c in enumerate(conditions):
            gm = mats[(arm, c)].mean(axis=1)
            jitter = rng.uniform(-w * 0.30, w * 0.30, len(gm))
            ax.scatter(pos[i] + jitter, gm, s=7, color=ARM_COLOR[arm], alpha=0.5,
                       edgecolors="none", zorder=3)
    ax.axhline(CHANCE, color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    ax.text(len(conditions) - 0.55, CHANCE + 0.015, "chance", color=MUTED,
            fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("auditor accuracy", color=SEC)
    n_g = results["arms"]["base"]["conditions"]["none"]["n_games"]
    n_s = results["arms"]["base"]["conditions"]["none"]["n_seeds"]
    ax.set_title(f"Auditor accuracy, 1-7 suspicion grades (OpenRouter, "
                 f"{n_g} games x {n_s} seeds)\n"
                 "bars: mean, game-cluster bootstrap 95% CI · dots: per-game mean",
                 color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right", labelcolor=SEC)
    fig.tight_layout()
    path = FIG / "auditor_accuracy_api_seeds.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("figure ->", path)

    # ── pooled slide: arms combined, one bar per condition ────────────────────
    results["pooled"] = {}
    pooled = {}
    for c in conditions:
        gm = np.concatenate([mats[(arm, c)].mean(axis=1) for arm in ARMS])
        lo, hi = cluster_ci(gm, rng)
        pooled[c] = (float(gm.mean()), lo, hi)
        results["pooled"][c] = {"n_game_clusters": len(gm),
                                "n_audits": int(sum(mats[(a, c)].size for a in ARMS)),
                                "accuracy": pooled[c][0], "ci95_cluster": [lo, hi]}
    n_aud = results["pooled"][conditions[0]]["n_audits"]
    slide_figure(pooled,
                 subtitle=f"{n_aud} audits per condition "
                          "(100 games x 10 auditor seeds, steered + non-steered pooled)",
                 path=FIG / "slide_auditor_pooled.png")

    (OUT / "results_api.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
