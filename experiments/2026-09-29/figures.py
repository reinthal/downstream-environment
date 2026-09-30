"""Figures for 2026-09-29 (CPU; reads results.json + score/auditor files).

  figures/auditor_accuracy.png  auditor accuracy per condition, 2k vs 10k budget
  figures/harness_fix.png       old (name-fallback) vs fixed scoring, same audits
  figures/signal_auroc.png      Parsnip-vs-Carrot separability per detector
  figures/probe_scores.png      LR probe score distributions by situation
  figures/nothink.png           thinking-disabled games: false flags + auditor accuracy

Run:  .venv/bin/python experiments/2026-09-29/figures.py   (after analyze.py)
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
R = json.loads((OUT / "results.json").read_text())

SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")
BLUE, ORANGE, BLUE_LIGHT = "#2a78d6", "#eb6834", "#9ec5f4"
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


# ── 1. auditor accuracy ──────────────────────────────────────────────────────
CONDS = [("none", "No detector"), ("lr", "LR probe"), ("dom", "Diff-of-means probe"),
         ("oracle", "Oracle judge (told the role)"),
         ("godmode", "God-mode judge (reads reasoning)"), ("gold", "Gold (true roles)")]
a2k, a10k = R["2. new audits: audit_2k"], R["2. new audits: audit_10k"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, arm in zip(axes, ("base", "steered")):
    for i, (c, _) in enumerate(CONDS):
        y = len(CONDS) - 1 - i
        both = f"{arm}/{c}" in a10k
        for res, color, dy in ((a2k, BLUE, 0.19 if both else 0), (a10k, ORANGE, -0.19)):
            e = res.get(f"{arm}/{c}")
            if e is None:
                continue
            ax.barh(y + dy, e["acc"], height=0.32, color=color)
            ax.plot(e["ci95"], [y + dy] * 2, color=INK2, lw=1)
            ax.text(max(e["ci95"][1], e["acc"]) + 0.015, y + dy, f"{e['acc']:.2f}",
                    va="center", fontsize=8.5, color=INK2)
    ax.axvline(0.2, color=MUTED, lw=1)
    ax.text(0.215, len(CONDS) - 0.45, "chance", fontsize=8.5, color=MUTED, ha="left")
    ax.set(xlim=(0, 1.12), ylim=(-0.6, len(CONDS) - 0.25), xlabel="Auditor accuracy",
           yticks=range(len(CONDS)), yticklabels=[l for _, l in CONDS][::-1])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(f"{arm.capitalize()} games", loc="left", fontsize=10.5, color=INK2)
handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
fig.legend(handles, ["2,000 thinking tokens", "10,000 thinking tokens (none / god-mode / gold only)"],
           loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, fontsize=9)
titled(fig, "Given good grades the auditor finds the Parsnip; probe grades leave it at chance",
       "Share of audits naming the true Parsnip. 50 games x 5 auditor seeds per bar; "
       "whiskers = 95% game-cluster bootstrap CI.")
fig.subplots_adjust(left=0.26, right=0.98, top=0.80, bottom=0.17, wspace=0.06)
save(fig, "auditor_accuracy.png")

# ── 2. harness fix ───────────────────────────────────────────────────────────
old = R["1. re-scored: 09-25 local none/godmode, 10 seeds"]
rows = [("God-mode, base", old["base/godmode"]["reported_acc"], a2k["base/godmode"]["acc"]),
        ("God-mode, steered", old["steered/godmode"]["reported_acc"], a2k["steered/godmode"]["acc"]),
        ("No detector, base", old["base/none"]["reported_acc"], a2k["base/none"]["acc"]),
        ("No detector, steered", old["steered/none"]["reported_acc"], a2k["steered/none"]["acc"])]
fig, ax = plt.subplots(figsize=(8, 3.4))
for i, (label, before, after) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([before, after], [y, y], color=AXIS, lw=2, zorder=1)
    ax.scatter([before], [y], s=70, color=BLUE_LIGHT, edgecolor=SURFACE, lw=2, zorder=2)
    ax.scatter([after], [y], s=70, color=BLUE, edgecolor=SURFACE, lw=2, zorder=3)
    lo, hi = sorted((before, after))
    if hi - lo > 0.06:
        ax.text(lo - 0.02, y, f"{before:.2f}", ha="right", va="center", fontsize=9, color=INK2)
    ax.text(hi + 0.02, y, f"{after:.2f}", ha="left", va="center", fontsize=9, color=INK2)
ax.set(xlim=(0, 1.08), ylim=(-0.6, len(rows) - 0.4), xlabel="Auditor accuracy",
       yticks=range(len(rows)), yticklabels=[r[0] for r in rows][::-1])
ax.legend([plt.Line2D([], [], marker="o", ls="", color=BLUE_LIGHT, ms=8),
           plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8)],
          ["09-25 harness (cut-off answers scored as last name mentioned)",
           "Fixed harness (answer tag required, forced final answer)"],
          loc="lower left", bbox_to_anchor=(-0.32, -0.62), fontsize=9)
titled(fig, "Auditor accuracy before and after the answer-parsing fix",
       "Same games, same 2,000-token budget. With no detector the bug only added noise.")
fig.subplots_adjust(left=0.24, right=0.97, top=0.76, bottom=0.34)
save(fig, "harness_fix.png")

# ── 3. separability per detector ─────────────────────────────────────────────
J, P = R["3. text judge views"], R["4a. probe scores by arm"]
I, RP = R["4b. in-game instructed pairs (clean pairs only)"], R["5. role probe AUROC (Parsnip vs Carrot)"]
items = [
    ("Text judge, full private reasoning", J["base/full"]["parsnip_vs_carrot_auroc"]),
    ("Text judge, role words masked", J["base/redacted"]["parsnip_vs_carrot_auroc"]),
    ("Text judge, public text only", J["base/public"]["parsnip_vs_carrot_auroc"]),
    ("Role probe (trained on game data), full turn", RP["base/full (leave-one-seed-group-out)"]),
    ("Role probe, announcement only", RP["base/pub (leave-one-seed-group-out)"]),
    ("LR deception probe, instructed conceal vs admit", I["lr_full"]["deceptive_vs_honest_auroc"]),
    ("LR deception probe, natural play, full turn", P["base (50 games)/lr_full"]["parsnip_vs_carrot_auroc"]),
    ("LR deception probe, natural play, announcement", P["base (50 games)/lr_pub"]["parsnip_vs_carrot_auroc"]),
    ("Random direction, natural play", P["base (50 games)/rand_full"]["parsnip_vs_carrot_auroc"]),
]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
for i, (label, v) in enumerate(items):
    y = len(items) - 1 - i
    ax.plot([0.5, v], [y, y], color=BLUE, lw=2, solid_capstyle="round")
    ax.scatter([v], [y], s=70, color=BLUE, edgecolor=SURFACE, lw=2, zorder=3)
    ax.text(v + 0.015 if v >= 0.5 else v - 0.015, y, f"{v:.2f}", va="center",
            ha="left" if v >= 0.5 else "right", fontsize=9, color=INK2)
for y in (2.5, 5.5):
    ax.axhline(y, color=GRID, lw=1)
ax.axvline(0.5, color=MUTED, lw=1)
ax.text(0.507, len(items) - 0.45, "chance", fontsize=8.5, color=MUTED, ha="left")
ax.set(xlim=(0.4, 1.07), ylim=(-0.6, len(items) - 0.2), xlabel="AUROC",
       yticks=range(len(items)), yticklabels=[l for l, _ in items][::-1])
titled(fig, "The Parsnip is easy to spot in its reasoning, not by the lie probes",
       "AUROC for telling Parsnip messages from Carrot messages (instructed pairs: concealing "
       "vs admitting Parsnip). Base games.")
fig.subplots_adjust(left=0.44, right=0.97, top=0.82, bottom=0.12)
save(fig, "signal_auroc.png")

# ── 4. LR probe score distributions ──────────────────────────────────────────
ADMITS = re.compile(r"\bI(?:'m| am)\s+(?:the\s+|a\s+)?parsnip\b", re.IGNORECASE)


def table(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))


base = [r for r in table("scores_base.csv") if int(r["game_id"].split("g")[1]) < 10]
inst = table("scores_instructed.csv")
tr = [json.loads(l) for l in open(OUT / "transcripts_instructed.jsonl")]
pairs = defaultdict(dict)
for r, t in zip(inst, tr):
    r["admits"] = bool(ADMITS.search(t["announcement"] or ""))
    pairs[(r["game_id"], r["round"], r["position"])][r["variant"]] = r
clean = [p for p in pairs.values() if p["honest"]["admits"] and not p["deceptive"]["admits"]]
col = "lr_full"
groups = [
    ("Carrot, game with no Parsnip", [float(r[col]) for r in table("scores_noparsnip.csv")]),
    ("Carrot, normal game", [float(r[col]) for r in base if r["role"] == "Carrot"]),
    ("Parsnip, normal game", [float(r[col]) for r in base if r["role"] == "Parsnip"]),
    ("Parsnip, told to conceal", [float(p["deceptive"][col]) for p in clean]),
    ("Parsnip, told to admit its role", [float(p["honest"][col]) for p in clean]),
]
thr = json.loads((OUT / "suspicion_scale.json").read_text())["cutpoints"]["lr"][-1]
fig, ax = plt.subplots(figsize=(8.6, 3.9))
pos = list(range(len(groups)))[::-1]
ax.boxplot([g for _, g in groups], positions=pos, orientation="horizontal", widths=0.5, whis=(5, 95),
                showfliers=False, patch_artist=True,
                medianprops={"color": SURFACE, "lw": 2}, whiskerprops={"color": BLUE, "lw": 1.5},
                capprops={"color": BLUE, "lw": 1.5}, boxprops={"facecolor": BLUE, "edgecolor": BLUE})
for y, (_, g) in zip(pos, groups):
    ax.text(np.percentile(g, 95) + 0.008, y, f"median {np.median(g):.3f}", va="center",
            fontsize=8.5, color=INK2)
ax.axvline(thr, color=MUTED, lw=1)
ax.text(thr + 0.004, len(groups) - 0.42, "1% false-positive\nthreshold (alpaca)", fontsize=8.5,
        color=MUTED, ha="left", va="bottom")
ax.set(xlim=(0, 0.32), ylim=(-0.6, len(groups) + 0.25),
       xlabel="LR probe score, full turn", yticks=pos,
       yticklabels=[l for l, _ in groups])
titled(fig, "The probe reacts to an explicit lie-vs-admit contrast, not to role",
       "Box = middle 50%, whiskers = 5th-95th percentile, white line = median. "
       "Instructed turns: 273 clean pairs.")
fig.subplots_adjust(left=0.30, right=0.97, top=0.76, bottom=0.15)
save(fig, "probe_scores.png")

# ── 5. games without private reasoning ───────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1, 1.5]})
flag = [("With thinking", P["base (50 games)/lr_full"]["carrot_frac_over_1pct_fpr"]),
        ("Thinking disabled", P["nothink/lr_full"]["carrot_frac_over_1pct_fpr"])]
for i, (label, v) in enumerate(flag):
    y = len(flag) - 1 - i
    ax1.barh(y, v, height=0.45, color=BLUE)
    ax1.text(v + 0.02, y, f"{v:.0%}", va="center", fontsize=9, color=INK2)
ax1.set(xlim=(0, 1), ylim=(-0.6, 1.6), yticks=[1, 0], yticklabels=[l for l, _ in flag],
        xlabel="Share of Carrot messages")
ax1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax1.set_title("Honest messages over the 1% threshold (LR)", loc="left", fontsize=10.5, color=INK2)
NT = [("none", "No detector"), ("lr", "LR probe"), ("dom", "Diff-of-means probe"),
      ("judge_public", "Public-text judge")]
for i, (c, label) in enumerate(NT):
    y = len(NT) - 1 - i
    e = a2k[f"nothink/{c}"]
    ax2.barh(y, e["acc"], height=0.45, color=BLUE)
    ax2.plot(e["ci95"], [y, y], color=INK2, lw=1)
    ax2.text(e["ci95"][1] + 0.015, y, f"{e['acc']:.2f}", va="center", fontsize=9, color=INK2)
ax2.axvline(0.2, color=MUTED, lw=1)
ax2.text(0.207, len(NT) - 0.45, "chance", fontsize=8.5, color=MUTED, ha="left")
ax2.set(xlim=(0, 0.6), ylim=(-0.6, len(NT) - 0.25), yticks=range(len(NT)),
        yticklabels=[l for _, l in NT][::-1], xlabel="Auditor accuracy")
ax2.set_title("Auditor accuracy, thinking disabled", loc="left", fontsize=10.5, color=INK2)
titled(fig, "Without private reasoning the probes flag everyone and the auditor stays at chance",
       "50 games with thinking disabled; auditor 5 seeds per game, whiskers = 95% game-cluster "
       "bootstrap CI.")
fig.subplots_adjust(left=0.15, right=0.98, top=0.76, bottom=0.15, wspace=0.55)
save(fig, "nothink.png")
