"""Figure for the AIS-404 pre-onset ablation (PREREGISTRATION.md §4): every
condition on a shared probe-score y-axis, one panel per architecture, identical
y-limits across panels. Violin + box + median per condition.

Conditions are the locally re-collected ones (``scores_main_<cond>.jsonl``);
``post_onset`` is the main-run reference (sibling ``scores_main_attack.jsonl``,
steps at/after the frozen onset). Missing conditions are skipped so the figure
can be drawn before the run finishes (a ``(partial)`` tag is added then).

Run (.venv):  uv run --no-sync python experiments/2026-10-14-preonset-ablation/figures.py --tag main
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

# display order, grouped: honest baseline | attack-trajectory arms | honest-trajectory controls
PLOT_ORDER = [
    ("honest", "honest", "#4c72b0"),
    ("swap", "swap\n(attack traj,\ninstr removed)", "#55a868"),
    ("pre_onset", "pre_onset\n(attack traj,\ninstr present)", "#dd8452"),
    ("post_onset", "post_onset\n(overt attack)\n[main-run ref]", "#c44e52"),
    ("ctrl_neutral", "ctrl_neutral\n(honest+instr,\nneutral)", "#8172b3"),
    ("ctrl_good", "ctrl_good\n(honest+instr,\nunethical)", "#937860"),
    ("ctrl_evil", "ctrl_evil\n(honest+instr,\nmonitored)", "#da8bc3"),
    ("honest_attack", "honest_attack\n(honest traj,\nfull attack prompt)", "#8c8c8c"),
]
BASELINE = {"honest": "orig_honest", "pre_onset": "orig_preonset"}


def load_scores(tag: str, cond: str, arch: str):
    """Return the per-step probe scores for a condition, or None if absent."""
    if cond == "post_onset":
        path = C.MAIN / "scores_main_attack.jsonl"
        if not path.exists():
            return None
        onsets = C.load_onsets()
        field = f"{arch}_contrast"
        vals = [r[field] for r in C.read_jsonl(path)
                if not r["dropped"] and r["uid"] in onsets and r["step"] >= onsets[r["uid"]]]
        return np.array(vals, float)
    path = C.OUT / f"scores_{tag}_{BASELINE.get(cond, cond)}.jsonl"
    if not path.exists():
        return None
    field = f"{arch}_score"
    return np.array([r[field] for r in C.read_jsonl(path) if not r["dropped"]], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--archs", default="linear,mlp")
    ap.add_argument("--out", default="fig_conditions.png")
    args = ap.parse_args()
    archs = args.archs.split(",")

    data = {a: {} for a in archs}
    present = []
    for cond, _, _ in PLOT_ORDER:
        got = {a: load_scores(args.tag, cond, a) for a in archs}
        if all(v is not None and len(v) for v in got.values()):
            present.append(cond)
            for a in archs:
                data[a][cond] = got[a]
    labels = {c: lbl for c, lbl, _ in PLOT_ORDER}
    colors = {c: col for c, _, col in PLOT_ORDER}
    partial = [c for c, _, _ in PLOT_ORDER if c not in present]

    # shared y-limits across panels (PREREG §4): common 1-99 pct over all shown data
    allv = np.concatenate([data[a][c] for a in archs for c in present])
    ylo, yhi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
    pad = 0.08 * (yhi - ylo)
    ylim = (ylo - pad, yhi + pad)

    fig, axes = plt.subplots(1, len(archs), figsize=(1.9 * len(present) * len(archs) / 2 + 2,
                                                      6.2), sharey=True)
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        xs = np.arange(len(present))
        vdata = [data[arch][c] for c in present]
        vp = ax.violinplot(vdata, positions=xs, widths=0.8, showextrema=False)
        for body, c in zip(vp["bodies"], present):
            body.set_facecolor(colors[c]); body.set_alpha(0.45); body.set_edgecolor("none")
        bp = ax.boxplot(vdata, positions=xs, widths=0.22, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", lw=1.6),
                        whiskerprops=dict(color="#444"), capprops=dict(color="#444"),
                        boxprops=dict(facecolor="white", edgecolor="#444", alpha=0.9))
        hon_med = float(np.median(data[arch]["honest"])) if "honest" in present else None
        if hon_med is not None:
            ax.axhline(hon_med, ls="--", lw=1, color="#4c72b0", alpha=0.7, zorder=0)
        for x, c in zip(xs, present):
            m = float(np.median(data[arch][c]))
            ax.annotate(f"{m:+.3f}", (x, m), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7.5, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[c] for c in present], fontsize=7.2, rotation=0)
        ax.set_title(f"{arch} probe", fontsize=11, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(0, color="#999", lw=0.8, zorder=0)
    axes[0].set_ylabel("fibs probe score  (lie − truth)", fontsize=10)
    n = {c: len(data[archs[0]][c]) for c in present}
    sub = "AIS-404 pre-onset ablation — probe score by condition  (dashed = honest median; n per condition: " \
          + ", ".join(f"{c}={n[c]}" for c in present) + ")"
    if partial:
        sub += f"\n[still collecting: {', '.join(partial)}]"
    fig.suptitle(sub, fontsize=9.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = C.OUT / args.out
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  ({len(present)} conditions: {', '.join(present)})")


if __name__ == "__main__":
    main()
