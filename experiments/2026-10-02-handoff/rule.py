""""Highest average score" rule for every detector (CPU).

Per game: average each player's detector scores over their messages and name
the player with the highest average (ties split evenly). No auditor involved —
this is what each detector is worth at the game level, compared with what
the LLM auditor achieved when shown the same detector as 1-7 grades.

Probe scores are averaged as log-odds (averaging probabilities lets a few
near-1 outliers dominate); judge scores are their 1-7 ratings.
Scored on the 50 audited games (game_idx < 10 per seed group), like the audits.
The role probe's scores are out-of-fold (leave-one-seed-group-out) predictions.

Output: rule.json, figures/rule_vs_auditor.png, figures/probes_auditor_vs_average.png,
        figures/handoff_simple.png

Run:  .venv/bin/python experiments/2026-10-02-handoff/rule.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
EXP = OUT.parent
REPO = EXP.parent
E29, E01 = EXP / "2026-09-29", EXP / "2026-10-01"
KEY = ("game_id", "round", "position")


# god-mode judge outputs with no parsable 1-7 score (8/1500 base, 5/1500
# steered in 2026-09-29) are scored as the scale midpoint
UNPARSED_JUDGE = 4

def jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def table(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def key(r):
    return (r["game_id"], int(r["round"]), int(r["position"]))


def audited(gid: str) -> bool:
    return int(gid.split("g")[1]) < 10


def rule(meta: dict, scores: dict) -> float | None:
    """meta: key -> (speaker, role); scores: key -> float."""
    games = defaultdict(lambda: defaultdict(list))
    parsnip = {}
    for k, (speaker, role) in meta.items():
        if not audited(k[0]) or k not in scores:
            continue
        games[k[0]][speaker].append(scores[k])
        if role == "Parsnip":
            parsnip[k[0]] = speaker
    if not games:
        return None
    acc = []
    for g, players in games.items():
        mean = {p: np.mean(v) for p, v in players.items()}
        top = [p for p, v in mean.items() if v == max(mean.values())]
        acc.append((parsnip[g] in top) / len(top))
    return round(float(np.mean(acc)), 3)


def judge(path):
    return {key(r): float(UNPARSED_JUDGE if r["judge_score"] is None else r["judge_score"]) for r in jsonl(path)} if path.exists() else {}


def column(path, col):
    """Probe scores as log-odds."""
    if not path.exists():
        return {}
    p = {key(r): min(max(float(r[col]), 1e-6), 1 - 1e-6) for r in table(path)}
    return {k: float(np.log(v / (1 - v))) for k, v in p.items()}


def role_probe_oof(arm: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    acts = E29 / f"message_activations_{arm}.npz"
    if not acts.exists():
        return {}
    rows = table(E29 / f"scores_{arm}.csv")
    X = np.load(acts)["X_full"]
    y = np.array([r["role"] == "Parsnip" for r in rows], dtype=int)
    groups = np.array([int(r["seed_group"]) for r in rows])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    p = cross_val_predict(clf, X, y, groups=groups, cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {key(r): float(np.log(v / (1 - v))) for r, v in zip(rows, p)}


def auditor_acc(path, cond):
    if not path.exists():
        return None
    rs = [r for r in jsonl(path) if r["condition"] == cond]
    return round(float(np.mean([r["correct"] for r in rs])), 3) if rs else None


def main():
    A29, A01 = E29 / "audit_2k", E01 / "audit"
    oracle = REPO / "data" / "carrot-parsnip_qwen3.5-27b_judge_oracle_{arm}.jsonl"
    results = {}
    for arm in ("base", "steered", "nothink"):
        meta = {key(r): (r["speaker"], r["role"]) for r in jsonl(E29 / f"transcripts_{arm}.jsonl")}
        sc = E29 / f"scores_{arm}.csv"
        detectors = {
            # name: (scores, auditor accuracy with these grades or None)
            "God-mode judge (reads reasoning)": (judge(E29 / f"judge_full_{arm}.jsonl"),
                                                 auditor_acc(A29 / f"auditor_seeds_{arm}.jsonl", "godmode")),
            "Oracle judge (told the role)": (judge(Path(str(oracle).format(arm=arm))),
                                             auditor_acc(A29 / f"auditor_seeds_{arm}.jsonl", "oracle")),
            "Judge, role words masked": (judge(E29 / f"judge_redacted_{arm}.jsonl"), None),
            "Judge, public text only": (judge(E29 / f"judge_public_{arm}.jsonl"),
                                        auditor_acc(A29 / f"auditor_seeds_{arm}.jsonl", "judge_public")),
            "Role probe (trained on game)": (role_probe_oof(arm), None),
            "Follow-up probe": (column(E01 / f"scores_{arm}.csv", "followup_full"),
                                auditor_acc(A01 / f"auditor_seeds_{arm}.jsonl", "followup")),
            "LR probe": (column(sc, "lr_full"), auditor_acc(A29 / f"auditor_seeds_{arm}.jsonl", "lr")),
            "Diff-of-means probe": (column(sc, "dom_full"),
                                    auditor_acc(A29 / f"auditor_seeds_{arm}.jsonl", "dom")),
            "Random direction": (column(sc, "rand_full"), None),
        }
        res = {}
        for name, (scores, aud) in detectors.items():
            r = rule(meta, scores)
            if r is not None:
                res[name] = {"highest_average_rule": r, "auditor_with_grades": aud}
        res["No detector (auditor)"] = {
            "highest_average_rule": None,
            "auditor_with_grades": auditor_acc(A29 / f"auditor_seeds_{arm}.jsonl", "none")}
        results[arm] = res
    (OUT / "rule.json").write_text(json.dumps(results, indent=1) + "\n")
    for arm, res in results.items():
        print(f"\n== {arm}")
        for name, v in res.items():
            print(f"  {name:34s} rule {v['highest_average_rule']}   auditor {v['auditor_with_grades']}")
    plot(results)
    plot_probes(results)
    plot_simple(results)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                             "#e1e0d9", "#c3c2b7")
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "x",
        "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
        "ytick.left": False, "xtick.bottom": False, "legend.frameon": False})
    res = results["base"]
    names = [n for n in res if res[n]["highest_average_rule"] is not None]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, n in enumerate(names):
        y = len(names) - 1 - i
        r, a = res[n]["highest_average_rule"], res[n]["auditor_with_grades"]
        if a is not None:
            ax.plot([a, r], [y, y], color=AXIS, lw=2, zorder=1)
            ax.scatter([a], [y], s=70, color=BLUE, edgecolor=SURFACE, lw=2, zorder=3)
            ax.text(a - 0.015, y, f"{a:.2f}", ha="right", va="center", fontsize=9, color=INK2)
        ax.scatter([r], [y], s=70, color=ORANGE, edgecolor=SURFACE, lw=2, zorder=3)
        ax.text(r + 0.015, y, f"{r:.2f}", ha="left", va="center", fontsize=9, color=INK2)
    ax.axvline(0.2, color=MUTED, lw=1)
    ax.text(0.207, len(names) - 0.45, "chance", fontsize=8.5, color=MUTED)
    ax.set(xlim=(0, 1.08), ylim=(-0.6, len(names) - 0.2),
           xlabel="Share of games where the Parsnip is named",
           yticks=range(len(names)), yticklabels=names[::-1])
    ax.legend([plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=8),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8)],
              ["Pick the player with the highest average score (no LLM)",
               "LLM auditor shown the same detector as 1-7 grades"],
              loc="lower left", bbox_to_anchor=(-0.45, -0.36), ncol=2, fontsize=9)
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.14 / h, "Just picking the highest-scoring player beats the auditor",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.44 / h, "Base games, 50 games (auditor: 5 seeds each). Chance = 0.20.",
             fontsize=9.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.33, right=0.97, top=0.82, bottom=0.21)
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "rule_vs_auditor.png", dpi=170)
    print("wrote", OUT / "figures" / "rule_vs_auditor.png")



def plot_probes(results):
    """LR vs follow-up probe: auditor with grades vs the highest-average rule, per arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    arms = [("base", "Base"), ("steered", "Steered"), ("nothink", "No thinking")]
    probes = [("LR probe", "LR probe (instructed pairs)"), ("Follow-up probe", "Follow-up ‘did you lie’ probe")]
    series = [("Auditor, no detector", GRAY,
               lambda arm, p: results[arm]["No detector (auditor)"]["auditor_with_grades"]),
              ("Auditor shown the probe’s grades", BLUE,
               lambda arm, p: results[arm][p]["auditor_with_grades"]),
              ("Pick the highest average score (no LLM)", ORANGE,
               lambda arm, p: results[arm][p]["highest_average_rule"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    h = 0.24
    for ax, (pkey, ptitle) in zip(axes, probes):
        for i, (arm, _) in enumerate(arms):
            y = len(arms) - 1 - i
            for j, (_, color, val) in enumerate(series):
                v = val(arm, pkey)
                yy = y + (1 - j) * (h + 0.03)
                ax.barh(yy, v, height=h, color=color)
                ax.text(v + 0.015, yy, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
        ax.axvline(0.2, color=MUTED, lw=1)
        ax.text(0.207, len(arms) - 0.42, "chance", fontsize=8.5, color=MUTED)
        ax.set(xlim=(0, 1), ylim=(-0.55, len(arms) - 0.3), xlabel="Share of games where the Parsnip is named",
               yticks=range(len(arms)), yticklabels=[l for _, l in arms][::-1])
        ax.set_title(ptitle, loc="left", fontsize=10.5, color=INK2)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in series],
               [s for s, _, _ in series], loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=3,
               fontsize=9)
    hgt = fig.get_figheight()
    fig.text(0.02, 1 - 0.14 / hgt, "Averaging the probe’s scores finds the Parsnip far more often than the auditor",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.44 / hgt, "50 games per arm; auditor 5 seeds per game (2,000 tokens). "
             "Probe scores averaged per player as log-odds. ±0.12 noise at 50 games.",
             fontsize=9.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.78, bottom=0.22, wspace=0.08)
    fig.savefig(OUT / "figures" / "probes_auditor_vs_average.png", dpi=170)
    print("wrote", OUT / "figures" / "probes_auditor_vs_average.png")


def plot_simple(results):
    """Base games: auditor vs highest-average rule for random / LR / follow-up."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                             "#e1e0d9", "#c3c2b7")
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "x",
        "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
        "ytick.left": False, "xtick.bottom": False, "legend.frameon": False})
    res = results["base"]
    groups = [("Random direction", "Random direction"), ("LR probe", "LR probe"),
              ("Follow-up probe", "Follow-up ‘did you lie’ probe")]
    none = res["No detector (auditor)"]["auditor_with_grades"]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    h = 0.32
    for i, (k, label) in enumerate(groups):
        y = len(groups) - 1 - i
        aud, rule_v = res[k]["auditor_with_grades"], res[k]["highest_average_rule"]
        if aud is not None:
            ax.barh(y + h / 2 + 0.02, aud, height=h, color=BLUE)
            ax.text(aud + 0.015, y + h / 2 + 0.02, f"{aud:.2f}", va="center", fontsize=9, color=INK2)
        else:
            ax.text(0.01, y + h / 2 + 0.02, "auditor not run", va="center", fontsize=8.5,
                    color=MUTED, style="italic")
        ax.barh(y - h / 2 - 0.02, rule_v, height=h, color=ORANGE)
        ax.text(rule_v + 0.015, y - h / 2 - 0.02, f"{rule_v:.2f}", va="center", fontsize=9, color=INK2)
    ax.axvline(none, color=MUTED, lw=1)
    ax.text(none + 0.007, len(groups) - 0.42, f"auditor, no detector ({none:.2f})", fontsize=8.5,
            color=MUTED)
    ax.set(xlim=(0, 1), ylim=(-0.6, len(groups) - 0.3), xlabel="Share of games where the Parsnip is named",
           yticks=range(len(groups)), yticklabels=[l for _, l in groups][::-1])
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)],
              ["LLM auditor shown the probe’s grades", "Pick the player with the highest average score"],
              loc="lower left", bbox_to_anchor=(-0.3, -0.42), ncol=2, fontsize=9)
    hgt = fig.get_figheight()
    fig.text(0.02, 1 - 0.14 / hgt, "Averaging the probe finds the Parsnip; the auditor doesn’t",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.44 / hgt, "Base games, 50 games (auditor: 5 seeds each). Chance = 0.20; "
             "each value ±0.12.", fontsize=9.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.78, bottom=0.27)
    fig.savefig(OUT / "figures" / "handoff_simple.png", dpi=170)
    print("wrote", OUT / "figures" / "handoff_simple.png")


if __name__ == "__main__":
    main()
