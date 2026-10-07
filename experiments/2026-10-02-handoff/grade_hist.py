"""Distribution of the 1-7 grades the auditor was shown (base games, all
messages of the 50 audited games): LR probe and follow-up probe overlaid, each
with its original alpaca-based calibration (CPU).

Output: figures/grade_distributions.png

Run:  .venv/bin/python experiments/2026-10-02-handoff/grade_hist.py   (after calibrate.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
G = json.loads((OUT / "grade_rule.json").read_text())

SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "ytick.left": False, "xtick.bottom": False, "legend.frameon": False})

import csv  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(OUT.parents[1]))
from auditor.scale import grade  # noqa: E402

cuts = json.loads((OUT / "suspicion_scale.json").read_text())["cutpoints"]
with open(OUT / "scores_base.csv") as f:
    rows = [r for r in csv.DictReader(f) if int(r["game_id"].split("g")[1]) < 10]
grades = list(range(1, 8))
series = [("lr", BLUE, "LR probe"), ("fu", ORANGE, "Follow-up probe")]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for col, color, label in series:
    g = [grade(float(r[f"{col}_full"]), cuts[col]) for r in rows]
    share = [g.count(k) / len(g) for k in grades]
    ax.bar(grades, share, width=1.0, color=color, alpha=0.45, edgecolor=color, linewidth=1.5,
           label=label)
for lo, hi, lab in ((0.5, 3.5, "‘very likely honest’"), (3.5, 5.5, "‘unsure’"),
                    (5.5, 7.5, "‘likely dishonest’")):
    ax.text((lo + hi) / 2, 0.42, lab, ha="center", fontsize=9, color=MUTED)
for x in (3.5, 5.5):
    ax.axvline(x, color=INK2, lw=1)
ax.set(xticks=grades, xlim=(0.5, 7.5), ylim=(0, 0.45), xlabel="Grade (1-7)",
       ylabel="Share of messages")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2, fontsize=9)
h = fig.get_figheight()
fig.text(0.02, 1 - 0.14 / h, "Grades the auditor saw", fontsize=13, fontweight="bold", va="top")
fig.text(0.02, 1 - 0.44 / h, "Base games, all 750 messages of the 50 audited games. Labels above "
         "are the legend wording given to the auditor.", fontsize=9.5, color=INK2, va="top")
fig.subplots_adjust(left=0.09, right=0.98, top=0.8, bottom=0.28)
(OUT / "figures").mkdir(exist_ok=True)
fig.savefig(OUT / "figures" / "grade_distributions.png", dpi=170)
print("wrote", OUT / "figures" / "grade_distributions.png")

# ── same grades split by role: what separates the Parsnip ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (col, title) in zip(axes, [("lr", "LR probe"), ("fu", "Follow-up probe")]):
    means = {}
    for role, color in (("Carrot", BLUE), ("Parsnip", ORANGE)):
        g = [grade(float(r[f"{col}_full"]), cuts[col]) for r in rows if r["role"] == role]
        share = [g.count(k) / len(g) for k in grades]
        ax.bar(grades, share, width=1.0, color=color, alpha=0.45, edgecolor=color, linewidth=1.5)
        means[role] = np.mean(g)
    title = f"{title}: mean Carrot {means['Carrot']:.1f}, Parsnip {means['Parsnip']:.1f}"
    ax.set(xticks=grades, xlim=(0.5, 7.5), ylim=(0, 0.5), xlabel="Grade (1-7)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(title, loc="left", fontsize=10.5, color=INK2)
axes[0].set_ylabel("Share of that role’s messages")
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in (BLUE, ORANGE)],
           ["Carrot messages", "Parsnip messages"], loc="lower left", bbox_to_anchor=(0.02, 0.0),
           ncol=2, fontsize=9)
h = fig.get_figheight()
fig.text(0.02, 1 - 0.14 / h, "Grades the auditor saw, Carrot vs Parsnip messages", fontsize=13,
         fontweight="bold", va="top")
fig.text(0.02, 1 - 0.44 / h, "Base games, 50 audited games: 600 Carrot and 150 Parsnip messages.",
         fontsize=9.5, color=INK2, va="top")
fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.25, wspace=0.06)
fig.savefig(OUT / "figures" / "grade_by_role.png", dpi=170)
print("wrote", OUT / "figures" / "grade_by_role.png")
