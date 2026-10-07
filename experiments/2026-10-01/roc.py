"""ROC curves for the follow-up probe (CPU).

  left:  held-out roleplaying replies, honest vs deceptive (what it was trained for)
  right: Parsnip vs Carrot messages, base games (all 100), follow-up vs the
         instructed-pairs LR probe (2026-09-22/29)

Output: figures/roc.png, figures/roc_probes.png (one panel: follow-up, LR,
        random-direction baseline on base-game messages), figures/roc_base_vs_steered.png,
        figures/violins_base_vs_steered.png

Run:  .venv/bin/python experiments/2026-10-01/roc.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))
from linear_probes.probes import LinearProbe  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "legend.frameon": False})


def table(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def curve(ax, y, s, color, label):
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUROC {roc_auc_score(y, s):.2f})")


# held-out roleplaying replies
graded = [json.loads(l) for l in open(OUT / "judge_roleplaying_off.jsonl")]
val = json.loads((OUT / "train_off.json").read_text())["val_indices"]
X = np.load(OUT / "activations_followup_roleplaying_off.npz")["X"]
probe = LinearProbe.load(str(OUT / "probe_followup_off.npz"))
keep = [i for i in val if graded[i]["label"] in ("honest", "deceptive")]
y_rp = [graded[i]["label"] == "deceptive" for i in keep]
s_rp = probe.predict_proba(X[keep])

# game messages, base games
fu = table(OUT / "scores_base.csv")
lr = table(OUT.parent / "2026-09-29" / "scores_base.csv")
y_fu = [r["role"] == "Parsnip" for r in fu]
y_lr = [r["role"] == "Parsnip" for r in lr]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
curve(axes[0], y_rp, s_rp, ORANGE, "Follow-up probe")
axes[0].set_title(f"Held-out roleplaying ({sum(y_rp)} deceptive, {len(y_rp) - sum(y_rp)} honest)",
                  loc="left", fontsize=10, color=INK2)
curve(axes[1], y_fu, [float(r["followup_full"]) for r in fu], ORANGE, "Follow-up probe")
curve(axes[1], y_lr, [float(r["lr_full"]) for r in lr], BLUE, "LR probe")
axes[1].set_title("Parsnip vs Carrot, base games (300 vs 1,200 msgs)", loc="left",
                  fontsize=10, color=INK2)
for ax in axes:
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1)
    ax.text(0.62, 0.55, "chance", color=MUTED, fontsize=8.5, rotation=38)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal",
           xlabel="False positive rate (honest flagged)",
           ylabel="True positive rate (liar caught)")
    ax.legend(loc="lower right", fontsize=9)
h = fig.get_figheight()
fig.text(0.02, 1 - 0.14 / h, "ROC curves: follow-up (‘did you lie’) probe", fontsize=13,
         fontweight="bold", va="top")
fig.text(0.02, 1 - 0.44 / h, "Each curve traces every possible threshold; the further it bows "
         "toward the top-left, the better.", fontsize=9.5, color=INK2, va="top")
fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.12, wspace=0.25)
(OUT / "figures").mkdir(exist_ok=True)
fig.savefig(OUT / "figures" / "roc.png", dpi=170)
print("wrote", OUT / "figures" / "roc.png")

# ── single panel: follow-up vs LR vs random-direction baseline, base games ────
GRAY = "#8f8d86"
fig, ax = plt.subplots(figsize=(5.6, 5.4))
curve(ax, y_fu, [float(r["followup_full"]) for r in fu], ORANGE, "Follow-up (‘did you lie’) probe")
curve(ax, y_lr, [float(r["lr_full"]) for r in lr], BLUE, "LR probe")
curve(ax, y_lr, [float(r["rand_full"]) for r in lr], GRAY, "Random direction (baseline)")
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (1, 2)))
ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal", xlabel="False positive rate (honest flagged)",
       ylabel="True positive rate (Parsnip caught)")
ax.legend(loc="lower right", fontsize=9)
h = fig.get_figheight()
fig.text(0.03, 1 - 0.14 / h, "ROC: Parsnip vs Carrot messages", fontsize=13, fontweight="bold",
         va="top")
fig.text(0.03, 1 - 0.44 / h, "Base games, 300 Parsnip vs 1,200 Carrot messages.", fontsize=9.5,
         color=INK2, va="top")
fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.11)
fig.savefig(OUT / "figures" / "roc_probes.png", dpi=170)
print("wrote", OUT / "figures" / "roc_probes.png")

# ── base vs steered, same three curves ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
for ax, arm, title in zip(axes, ("base", "steered"), ("Base games", "Steered games")):
    fa = table(OUT / f"scores_{arm}.csv")
    la = table(OUT.parent / "2026-09-29" / f"scores_{arm}.csv")
    yf = [r["role"] == "Parsnip" for r in fa]
    yl = [r["role"] == "Parsnip" for r in la]
    curve(ax, yf, [float(r["followup_full"]) for r in fa], ORANGE, "Follow-up probe")
    curve(ax, yl, [float(r["lr_full"]) for r in la], BLUE, "LR probe")
    curve(ax, yl, [float(r["rand_full"]) for r in la], GRAY, "Random direction")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (1, 2)))
    ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal", xlabel="False positive rate (honest flagged)",
           ylabel="True positive rate (Parsnip caught)")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK2)
    ax.legend(loc="lower right", fontsize=9)
h = fig.get_figheight()
fig.text(0.02, 1 - 0.14 / h, "ROC: Parsnip vs Carrot messages, base vs steered games", fontsize=13,
         fontweight="bold", va="top")
fig.text(0.02, 1 - 0.44 / h, "Same 100 games; steered = deception direction added during play "
         "(Parsnip toward honest, Carrots toward deceptive).", fontsize=9.5, color=INK2, va="top")
fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.1, wspace=0.22)
fig.savefig(OUT / "figures" / "roc_base_vs_steered.png", dpi=170)
print("wrote", OUT / "figures" / "roc_base_vs_steered.png")

# ── violins: score distributions, Carrot vs Parsnip, base vs steered ─────────
def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


probes = [("followup_full", "fu", "Follow-up (‘did you lie’) probe"),
          ("lr_full", "lr", "LR probe"), ("rand_full", "lr", "Random direction")]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
for ax, (col, src, title) in zip(axes, probes):
    pos, xt = 0, []
    for arm, arm_label in (("base", "Base"), ("steered", "Steered")):
        if src == "fu":     # exact log-odds from activations (saved scores are rounded)
            roles = [json.loads(l)["role"] for l in
                     open(OUT.parent / "2026-09-29" / f"transcripts_{arm}.jsonl")]
            Xa = np.load(OUT / f"activations_followup_{arm}.npz")["X"]
            z = ((Xa - probe.mu) / probe.sd) @ probe.w + probe.b
            vals = {role: z[[r == role for r in roles]] for role in ("Carrot", "Parsnip")}
        else:
            rows_ = table(OUT.parent / "2026-09-29" / f"scores_{arm}.csv")
            vals = {role: logit([float(r[col]) for r in rows_ if r["role"] == role])
                    for role in ("Carrot", "Parsnip")}
        for k, (role, color) in enumerate((("Carrot", BLUE), ("Parsnip", ORANGE))):
            v = ax.violinplot(vals[role], positions=[pos + k * 0.9], widths=0.8,
                              showextrema=False, showmedians=True)
            for b in v["bodies"]:
                b.set_facecolor(color)
                b.set_edgecolor(color)
                b.set_alpha(0.5)
            v["cmedians"].set_color(INK)
        auc = roc_auc_score([0] * len(vals["Carrot"]) + [1] * len(vals["Parsnip"]),
                            [*vals["Carrot"], *vals["Parsnip"]])
        xt.append((pos + 0.45, f"{arm_label}\nAUROC {auc:.2f}"))
        pos += 2.4
    ax.set_xticks([x for x, _ in xt], [t for _, t in xt])
    lo, hi = ax.get_ylim()
    cands = ((1e-6, 1e-4, 1e-2, 0.5, 0.99) if hi - lo > 10 else
             (1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99))
    probs = [p for p in cands if lo <= np.log(p / (1 - p)) <= hi]
    ax.set_yticks([np.log(p / (1 - p)) for p in probs],
                  [f"{100 * p:.{max(0, -int(np.floor(np.log10(100 * p))))}f}%" for p in probs])
    ax.set_title(title, loc="left", fontsize=10.5, color=INK2)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Probe score (probability, log-odds spacing)")
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in (BLUE, ORANGE)],
           ["Carrot messages", "Parsnip messages"], loc="lower left", bbox_to_anchor=(0.02, 0.0),
           ncol=2, fontsize=9)
h = fig.get_figheight()
fig.text(0.02, 1 - 0.14 / h, "Probe scores for Carrot vs Parsnip messages, base vs steered games",
         fontsize=13, fontweight="bold", va="top")
fig.text(0.02, 1 - 0.44 / h, "All 100 games per arm (1,200 Carrot, 300 Parsnip messages). "
         "Black line = median.", fontsize=9.5, color=INK2, va="top")
fig.subplots_adjust(left=0.09, right=0.98, top=0.8, bottom=0.2, wspace=0.28)
fig.savefig(OUT / "figures" / "violins_base_vs_steered.png", dpi=170)
print("wrote", OUT / "figures" / "violins_base_vs_steered.png")
