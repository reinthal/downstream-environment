"""EXPLORATORY (not preregistered): ROC curves of every probe on the
announcements, Parsnip (positive) vs Carrot, one panel per arm with identical
axes. Solid lines: the probes listed in ``cfg.roc_probes`` (names below), in
that order. Dashed: the linear-matched random direction of seed
``cfg.roc_random_seed`` (random_probe.py). Dotted: chance.

A listed probe whose score file is missing is an error — the list in
config.json says what is drawn. Every probe must cover exactly the
announcements of scores_{arm}.csv. Scores are used as written (higher = more
deceptive, no sign flips).

  figures/roc.svg

Run:  uv run --no-sync python experiments/2026-10-12-fibs-probes/roc.py
"""
from __future__ import annotations

import csv
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze import MAIN, auroc  # noqa: E402
from common import OUT, load_config  # noqa: E402


def aisi_column() -> str:
    return f"aisi_score_l{json.loads((OUT / 'results_aisi_dyl.json').read_text())['chosen_layer_block_index']}"


# name -> (label, score file, column or a function returning it, colour).
# Colours follow the probe, not its position in cfg.roc_probes.
PROBES = {
    "linear": ("fibs linear", "scores_{arm}.csv", "linear_contrast", "#2a78d6"),
    "mlp": ("fibs MLP", "scores_{arm}.csv", "mlp_contrast", "#eb6834"),
    "efc": ("fibs EFC", "scores_{arm}.csv", "efc_contrast", "#1baf7a"),
    "axial": ("fibs axial", "scores_{arm}.csv", "axial_contrast", "#eda100"),
    "dyl": ("in-house DYL (layer 18)", "scores_{arm}.csv", "dyl_logodds", "#e87ba4"),
    "apollo": ("Apollo LR (layer 16)", "apollo_{arm}.csv", "plain_pub_apollo", "#008300"),
    "aisi": ("AISI DYL", "aisi_dyl_{arm}.csv", aisi_column, "#6250d6"),
}
RANDOM_COLOR, CHANCE_COLOR = "#52514e", "#8a8983"


def load(arm: str, file: str, column: str) -> dict[tuple, tuple[float, int]]:
    """{(game_id, round, position): (score, is_parsnip)}; fails if the file,
    the column or a value is missing."""
    out = {}
    for r in csv.DictReader(open(OUT / file.format(arm=arm))):
        out[r["game_id"], int(r["round"]), int(r["position"])] = (float(r[column]), int(r["role"] == "Parsnip"))
    return out


def roc(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(FPR, TPR) over all thresholds, ties moved through together."""
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]     # last index of each run of equal scores
    tpr = np.r_[0.0, np.cumsum(y)[last] / y.sum()]
    fpr = np.r_[0.0, np.cumsum(1 - y)[last] / (1 - y).sum()]
    return fpr, tpr


def main():
    cfg = load_config()
    unknown = [p for p in cfg.roc_probes if p not in PROBES]
    if unknown:
        raise ValueError(f"roc_probes names {unknown} are not in {list(PROBES)}")
    seed = cfg.roc_random_seed
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), sharex=True, sharey=True)
    for ax, arm in zip(axes, MAIN):
        ref = load(arm, "scores_{arm}.csv", f"{cfg.primary_arch}_contrast")
        keys = list(ref)

        def arrays(file: str, column: str, what: str):
            d = load(arm, file, column)
            if set(d) != set(keys):
                raise ValueError(f"{arm} {what}: {file.format(arm=arm)} covers {len(d)} announcements, "
                                 f"{len(set(d) ^ set(keys))} differ from scores_{arm}.csv ({len(keys)})")
            if any(d[k][1] != ref[k][1] for k in keys):
                raise ValueError(f"{arm} {what}: role labels differ from scores_{arm}.csv")
            return np.array([d[k][0] for k in keys]), np.array([d[k][1] for k in keys])

        ax.plot([0, 1], [0, 1], ls=":", lw=0.8, color=CHANCE_COLOR, label="chance", zorder=1)
        rand = {s_: arrays("random_{arm}.csv", f"rand_s{s_}", f"random seed {s_}") for s_ in cfg.random_seeds}
        ra = [auroc(*rand[s_]) for s_ in cfg.random_seeds]
        ax.plot(*roc(*rand[seed]), ls="--", lw=1.6, color=RANDOM_COLOR, zorder=2,
                label=f"random direction, seed {seed}: {auroc(*rand[seed]):.3f}\n"
                      f"({len(ra)} seeds: {min(ra):.3f}–{max(ra):.3f}, median {np.median(ra):.3f})")
        for z, name in enumerate(cfg.roc_probes):
            label, file, column, color = PROBES[name]
            s, y = arrays(file, column if isinstance(column, str) else column(), name)
            ax.plot(*roc(s, y), lw=1.8, color=color, label=f"{label}: {auroc(s, y):.3f}", zorder=3 + z)
        n_pos = sum(v[1] for v in ref.values())
        ax.set_title(f"{arm}: {len(keys)} announcements ({n_pos} Parsnip, {len(keys) - n_pos} Carrot)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("false-positive rate (Carrot announcements)")
        ax.grid(color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = handles[2:] + handles[1:2] + handles[:1], labels[2:] + labels[1:2] + labels[:1]
        ax.legend(handles, labels, loc="lower right", fontsize=8, title="AUROC", title_fontsize=8,
                  frameon=True, framealpha=1.0, edgecolor="#d5d4cf")
    axes[0].set_ylabel("true-positive rate (Parsnip announcements)")
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "roc.svg"); plt.close(fig)
    print(f"wrote figures/roc.svg ({', '.join(cfg.roc_probes)} + random seed {seed})")


if __name__ == "__main__":
    main()
