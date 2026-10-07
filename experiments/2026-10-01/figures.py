"""Figures for the follow-up probe (CPU; after analyze.py and 2026-10-02-handoff/calibrate.py).

  figures/followup_vs_lr.png   AUROC per test: follow-up probe vs instructed-pairs LR probe
  figures/handoff.png          auditor accuracy vs the grades' own "highest mean grade" rule

Run:  .venv/bin/python experiments/2026-10-01/figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = Path(__file__).resolve().parent
EXP = OUT.parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
R = json.loads((OUT / "results.json").read_text())
RULE = json.loads((EXP / "2026-10-02-handoff" / "grade_rule.json").read_text())

SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2a9"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "x",
    "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "ytick.left": False, "xtick.bottom": False, "legend.frameon": False})


def titled(fig, title, subtitle):
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.14 / h, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.44 / h, subtitle, fontsize=9.5, color=INK2, va="top")


def save(fig, name):
    fig.savefig(FIG / name, dpi=170)
    plt.close(fig)
    print("wrote", FIG / name)


# ── 1. follow-up vs LR probe ─────────────────────────────────────────────────
G, C = R["2. Parsnip vs Carrot"], R["3. controls and instructed pairs"]


def lr_auroc(arm: str) -> float:
    """LR probe (full turn) Parsnip-vs-Carrot AUROC over all games of an arm."""
    with open(EXP / "2026-09-29" / f"scores_{arm}.csv") as f:
        rows = list(csv.DictReader(f))
    return round(roc_auc_score([r["role"] == "Parsnip" for r in rows],
                               [float(r["lr_full"]) for r in rows]), 2)


rows = [
    ("Held-out roleplaying replies", R["1. roleplaying training (held-out 20%)"]["thinking off"]["val_auroc"], None),
    ("Parsnip vs Carrot, base games", G["base/followup_full parsnip-vs-carrot AUROC (all games)"],
     lr_auroc("base")),
    ("Parsnip vs Carrot, steered games", G["steered/followup_full parsnip-vs-carrot AUROC (all games)"],
     lr_auroc("steered")),
    ("Parsnip vs Carrot, no-thinking games", G["nothink/followup_full parsnip-vs-carrot AUROC (all games)"],
     lr_auroc("nothink")),
    ("Same Parsnip turn: conceal vs admit", C["conceal vs admit AUROC (clean pairs)"],
     C["old LR probe conceal vs admit AUROC"]),
]
fig, ax = plt.subplots(figsize=(8.6, 3.9))
for i, (label, fu, lr) in enumerate(rows):
    y = len(rows) - 1 - i
    pts = [v for v in (fu, lr) if v is not None]
    if len(pts) == 2:
        ax.plot(pts, [y, y], color=AXIS, lw=2, zorder=1)
    if lr is not None:
        ax.scatter([lr], [y], s=70, color=BLUE, edgecolor=SURFACE, lw=2, zorder=3)
        ax.text(lr - 0.012, y, f"{lr:.2f}", ha="right", va="center", fontsize=9, color=INK2)
    ax.scatter([fu], [y], s=70, color=ORANGE, edgecolor=SURFACE, lw=2, zorder=3)
    ax.text(fu + 0.012, y, f"{fu:.2f}", ha="left", va="center", fontsize=9, color=INK2)
ax.axvline(0.5, color=MUTED, lw=1)
ax.text(0.507, len(rows) - 0.45, "chance", fontsize=8.5, color=MUTED)
ax.set(xlim=(0.35, 1.05), ylim=(-0.6, len(rows) - 0.2), xlabel="AUROC",
       yticks=range(len(rows)), yticklabels=[r[0] for r in rows][::-1])
ax.legend([plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=8),
           plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8)],
          ["Follow-up probe ('Was your last response deceptive?' → 'No')",
           "Instructed-pairs LR probe (previous)"],
          loc="lower left", bbox_to_anchor=(-0.55, -0.42), ncol=2, fontsize=9)
titled(fig, "The follow-up probe separates liars better than the LR probe",
       "AUROC (1.0 = perfect, 0.5 = chance). Layer 18, trained on Qwen's own roleplaying replies.")
fig.subplots_adjust(left=0.36, right=0.97, top=0.80, bottom=0.25)
save(fig, "followup_vs_lr.png")

# ── 2. handoff: grades vs auditor ────────────────────────────────────────────
A = R["4. auditor accuracy (2k tokens, forced final answer)"]
arms = [("base", "Base games"), ("steered", "Steered games")]
series = [("No grades (auditor)", GRAY, lambda a: A[f"{a}/none"]["acc"]),
          ("Follow-up grades (auditor)", BLUE, lambda a: A[f"{a}/followup"]["acc"]),
          ("Same grades, pick highest average (no LLM)", ORANGE,
           lambda a: RULE[f"{a}/fu"]["highest_mean_grade_rule"])]
fig, ax = plt.subplots(figsize=(8.6, 3.6))
h = 0.22
for i, (_, label) in enumerate(arms):
    y = len(arms) - 1 - i
    for j, (_, color, val) in enumerate(series):
        v = val(arms[i][0])
        yy = y + (1 - j) * (h + 0.04)
        ax.barh(yy, v, height=h, color=color)
        ax.text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=9, color=INK2)
ax.axvline(0.2, color=MUTED, lw=1)
ax.text(0.207, len(arms) - 0.38, "chance", fontsize=8.5, color=MUTED)
ax.set(xlim=(0, 1), ylim=(-0.55, len(arms) - 0.3), xlabel="Share of games where the Parsnip is named",
       yticks=range(len(arms)), yticklabels=[l for _, l in arms][::-1])
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in series],
          [s for s, _, _ in series], loc="lower left", bbox_to_anchor=(-0.2, -0.5), ncol=3,
          fontsize=8.5)
titled(fig, "The grades point to the Parsnip; the auditor doesn't use them",
       "50 games; auditor 5 seeds per game. Same follow-up probe grades in the blue and orange bars.")
fig.subplots_adjust(left=0.17, right=0.97, top=0.78, bottom=0.30)
save(fig, "handoff.png")
