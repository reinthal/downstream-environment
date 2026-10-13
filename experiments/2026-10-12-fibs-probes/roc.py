"""EXPLORATORY (not preregistered): ROC curves of every probe on the
announcements, Parsnip (positive) vs Carrot, one column per arm: top row the
full ROC, bottom row a zoom on ``cfg.roc_zoom_fpr`` x ``cfg.roc_zoom_tpr`` (marked
by a rectangle on the full panel); each row has identical axes. Solid lines: the probes listed in ``cfg.roc_probes`` (names below), in
that order. Dashed: the linear-matched random direction of seed
``cfg.roc_random_seed`` (random_probe.py), which must be the best of
``cfg.random_seeds`` on the pooled AUROC (the legend says so; error otherwise).
Dotted: chance.

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
from matplotlib.patches import Rectangle  # noqa: E402

from analyze import MAIN, auroc  # noqa: E402
from common import OUT, load_config  # noqa: E402

# ── figure text ──
SUPTITLE = "Probe AUROC (larger area under the curve better)"
PANEL_TITLES = {"3p": "3-player game (~1/3 class balance)", "5p": "5-player game (~1/5 class balance)"}
XLABEL = "FPR (Carrot wrongly flagged)"
YLABEL = "TPR (Parsnip caught)"


def aisi_column() -> str:
    return f"aisi_score_l{json.loads((OUT / 'results_aisi_dyl.json').read_text())['chosen_layer_block_index']}"


# name -> (label, score file, column or a function returning it, colour).
# Colours follow the probe, not its position in cfg.roc_probes.
PROBES = {
    "linear": ("fibs linear", "scores_{arm}.csv", "linear_contrast", "#2a78d6"),
    "mlp": ("fibs MLP", "scores_{arm}.csv", "mlp_contrast", "#eb6834"),
    "efc": ("fibs EFC", "scores_{arm}.csv", "efc_contrast", "#1baf7a"),
    "axial": ("fibs axial", "scores_{arm}.csv", "axial_contrast", "#eda100"),
    "dyl": ("in-house DYL (layer 18, own question)", "scores_{arm}.csv", "dyl_logodds", "#b0527a"),
    "apollo": ("Apollo LR (read at layer 19, announcement tokens)", "apollo_{arm}.csv", "apollo_l19", "#008300"),
    # the DYL probes on the fibs probes' stored activations: same token (full stop) and rule (Yes. - No.) as fibs linear
    "dyl_same": ("in-house DYL (read at layer 19, fibs read)", "sameread_{arm}.csv", "dyl_stop_contrast", "#e87ba4"),
    "aisi_same": ("AISI DYL (read at layer 35, fibs read)", "sameread_{arm}.csv", "aisi_stop_contrast", "#6250d6"),
    "aisi": ("AISI DYL (layer 38, model-card format)", "aisi_dyl_{arm}.csv", aisi_column, "#3a2d8f"),
}
RANDOM_COLOR, CHANCE_COLOR, ZOOM_COLOR = "#52514e", "#8a8983", "#1f1e1c"


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
    # the legend calls this seed the best of the seeds on the pooled AUROC (no seed is best in both arms)
    pooled = {int(k): v["pooled"] for k, v in
              json.loads((OUT / "results_random.json").read_text())["auroc"]["rand"]["per_seed"].items()}
    if sorted(pooled) != sorted(cfg.random_seeds) or pooled[seed] < max(pooled.values()):
        raise ValueError(f"random seed {seed} is not the best of {sorted(pooled)} pooled; the legend would be false")
    (x0, x1), (y0, y1) = cfg.roc_zoom_fpr, cfg.roc_zoom_tpr
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 14.6), layout="constrained")
    for (full, zoom), arm in zip(axes.T, MAIN):
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

        rand = {s_: arrays("random_{arm}.csv", f"rand_s{s_}", f"random seed {s_}") for s_ in cfg.random_seeds}
        ra = {s_: auroc(*rand[s_]) for s_ in cfg.random_seeds}
        # (curve, plot kwargs), drawn identically in the full and the zoomed panel
        curves = [(([0, 1], [0, 1]), dict(ls=":", lw=0.8, color=CHANCE_COLOR, label="chance", zorder=1)),
                  (roc(*rand[seed]), dict(ls="--", lw=1.6, color=RANDOM_COLOR, zorder=2,
                                          label=f"random direction (seed {seed}) (best of {len(ra)} seeds, pooled): "
                                                f"{ra[seed]:.3f}"))]
        for z, name in enumerate(cfg.roc_probes):
            label, file, column, color = PROBES[name]
            s, y = arrays(file, column if isinstance(column, str) else column(), name)
            curves.append((roc(s, y), dict(lw=1.8, color=color, label=f"{label}: {auroc(s, y):.3f}", zorder=3 + z)))
        for ax in (full, zoom):
            for xy, kw in curves:
                ax.plot(*xy, **kw)
            ax.set_aspect("equal")
            ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL)
            ax.grid(color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        full.set_xlim(0, 1); full.set_ylim(0, 1)
        zoom.set_xlim(x0, x1); zoom.set_ylim(y0, y1)
        full.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.9, ec=ZOOM_COLOR, zorder=20,
                                 clip_on=False))
        full.set_title(PANEL_TITLES[arm])
        zoom.set_title(f"zoom on the marked region: FPR {x0:g}–{x1:g}, TPR {y0:g}–{y1:g}", fontsize=10)
        handles, labels = full.get_legend_handles_labels()
        handles, labels = handles[2:] + handles[1:2] + handles[:1], labels[2:] + labels[1:2] + labels[:1]
        # below the column: inside either panel a legend this size would cover curves
        zoom.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, title="AUROC",
                    title_fontsize=9, frameon=True, framealpha=1.0, edgecolor="#d5d4cf")
    fig.suptitle(SUPTITLE, fontsize=14)
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "roc.svg"); plt.close(fig)
    print(f"wrote figures/roc.svg ({', '.join(cfg.roc_probes)} + random seed {seed})")


if __name__ == "__main__":
    main()
