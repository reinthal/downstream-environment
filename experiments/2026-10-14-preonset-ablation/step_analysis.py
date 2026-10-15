"""Per-step trajectory analysis for the AIS-404 pre-onset ablation: for each
step index, the median probe score across trajectories with a task-clustered
bootstrap 95% CI, one line per condition, linear and mlp panels (shared y).

A step is kept for a condition only while at least ``--min-n`` trajectories (and
>= 2 tasks, for the cluster bootstrap) still reach it, so the thinning tails of
the pre-onset-only conditions are not drawn as noise.

Run (.venv):
  uv run --no-sync python experiments/2026-10-14-preonset-ablation/step_analysis.py --tag main
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

# condition -> colour (same palette as figures.py; post_onset omitted — it is a
# step-subset of the attack trajectory, not a condition of its own here)
PLOT_ORDER = [
    ("honest", "honest", "#4c72b0"),
    ("swap", "swap", "#55a868"),
    ("pre_onset", "pre_onset", "#dd8452"),
    ("ctrl_neutral", "ctrl_neutral", "#8172b3"),
    ("ctrl_good", "ctrl_good", "#937860"),
    ("ctrl_evil", "ctrl_evil", "#da8bc3"),
    ("honest_attack", "honest_attack", "#8c8c8c"),
]
BASELINE = {"honest": "orig_honest", "pre_onset": "orig_preonset"}


def task_of(uid: str) -> str:
    return uid.split("-")[0]


def clustered_median_ci(values, tasks, n_boot, seed):
    values = np.asarray(values, float)
    tasks = np.asarray(tasks)
    ids = np.unique(tasks)
    members = {c: np.nonzero(tasks == c)[0] for c in ids}
    point = float(np.median(values))
    if len(ids) < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([members[c] for c in draw])
        meds[b] = np.median(values[idx])
    return point, float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def per_step(rows, arch, n_boot, seed, min_n, bin_w):
    """bin_center -> (median, lo, hi, n); steps pooled into width-``bin_w`` bins
    (1-indexed: bin k covers steps [k*bin_w+1 .. (k+1)*bin_w])."""
    by_bin = defaultdict(list)
    for r in rows:
        if not r["dropped"]:
            b = (r["step"] - 1) // bin_w
            by_bin[b].append((r[f"{arch}_score"], task_of(r["uid"])))
    out = {}
    for b in sorted(by_bin):
        vals = [v for v, _ in by_bin[b]]
        tks = [t for _, t in by_bin[b]]
        if len(vals) < min_n:
            continue
        m, lo, hi = clustered_median_ci(vals, tks, n_boot, seed)
        center = b * bin_w + (bin_w + 1) / 2  # e.g. bin_w=5 -> 3, 8, 13, ...
        out[center] = (m, lo, hi, len(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--archs", default="linear,mlp")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--bin", type=int, default=5, help="step bin width")
    ap.add_argument("--out", default="fig_step_trajectory.png")
    args = ap.parse_args()
    archs = args.archs.split(",")

    conds = [(c, lbl, col) for c, lbl, col in PLOT_ORDER if c != "post_onset"]
    rows_by_cond = {}
    for c, _, _ in conds:
        path = C.OUT / f"scores_{args.tag}_{BASELINE.get(c, c)}.jsonl"
        if path.exists():
            rows_by_cond[c] = C.read_jsonl(path)
    conds = [(c, lbl, col) for c, lbl, col in conds if c in rows_by_cond]

    stats = {a: {} for a in archs}
    for a in archs:
        for c, _, _ in conds:
            stats[a][c] = per_step(rows_by_cond[c], a, args.n_boot, args.seed, args.min_n, args.bin)

    # shared y across panels (1-99 pct of the plotted CI bounds)
    allb = [v for a in archs for c, _, _ in conds for (m, lo, hi, n) in stats[a][c].values() for v in (lo, hi)]
    ylo, yhi = min(allb), max(allb)
    pad = 0.08 * (yhi - ylo)
    ylim = (ylo - pad, yhi + pad)

    fig, axes = plt.subplots(1, len(archs), figsize=(8.2 * len(archs), 6.4), sharey=True)
    if len(archs) == 1:
        axes = [axes]
    short = {"honest": "honest", "swap": "swap (atk traj, instr removed)",
             "pre_onset": "pre_onset (atk traj, instr present)",
             "ctrl_neutral": "ctrl_neutral (honest+instr)",
             "ctrl_good": "ctrl_good (honest+instr, unethical)",
             "ctrl_evil": "ctrl_evil (honest+instr, monitored)",
             "honest_attack": "honest_attack (honest traj, attack prompt)"}
    for ax, a in zip(axes, archs):
        for i, (c, _, col) in enumerate(conds):
            d = stats[a][c]
            if not d:
                continue
            xs = np.array(sorted(d))
            m = np.array([d[s][0] for s in xs])
            lo = np.array([d[s][1] for s in xs])
            hi = np.array([d[s][2] for s in xs])
            dx = (i - (len(conds) - 1) / 2) * 0.35  # small x-dodge so caps don't overlap
            ax.errorbar(xs + dx, m, yerr=[m - lo, hi - m], color=col, lw=1.6,
                        marker="o", ms=3, capsize=3, elinewidth=1.1, capthick=1.1,
                        label=short.get(c, c))
        ax.axhline(0, color="#999", lw=0.8, zorder=0)
        ax.set_title(f"{a} probe", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"trajectory step index (binned every {args.bin})", fontsize=10)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("fibs probe score  (lie − truth)", fontsize=10)
    axes[-1].legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle(f"AIS-404 pre-onset ablation — median probe score per {args.bin}-step bin "
                 f"(task-clustered 95% CI bars, {args.n_boot} resamples; bins with ≥{args.min_n} step-observations)",
                 fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = C.OUT / args.out
    fig.savefig(out, dpi=150, bbox_inches="tight")

    dump = {a: {c: {int(s): {"median": v[0], "ci95": [v[1], v[2]], "n": v[3]}
                    for s, v in stats[a][c].items()} for c, _, _ in conds} for a in archs}
    (C.OUT / f"step_stats_{args.tag}.json").write_text(json.dumps(dump, indent=1) + "\n")
    print(f"wrote {out} and step_stats_{args.tag}.json  ({len(conds)} conditions, archs={archs})")


if __name__ == "__main__":
    main()
