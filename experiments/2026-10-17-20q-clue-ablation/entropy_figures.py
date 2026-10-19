"""EXPLORATORY figures for the entropy follow-ups (no model calls; reads the
score files and games).

figures/entropy_remaining_by_question.svg
    No-clue games: bits of uncertainty left in the game's own 512-secret pool
    after k questions (hard count: log2 of the pool concepts consistent with
    every answer), mean over the games still running at k (>= 3), one line
    per model; panels Jev | Qwen3.8-27B with the same axes. Dashed: the
    pooled median pace extended to question 14 (turn 15 = the guess).
figures/bits_question_vs_clue.svg
    Clue games: mean hard bits per yes/no question vs per clue under the
    three scorers (dots: medians).

    uv run --project environments/twenty_questions python experiments/2026-10-17-20q-clue-ablation/entropy_figures.py
"""
from __future__ import annotations

import json
import math
import statistics

from common import ENV, OUT, AblationConfig, cell, slug

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # validated (dataviz), fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
POOL = [c["word"] for c in json.loads((ENV / "concepts/things_n512_s0.json").read_text())["concepts"]]
LABELS = {"qwen3_5-9b": "Qwen3.5-9B", "qwen3_5-27b": "Qwen3.5-27B", "qwen3_5-122b-a10b": "Qwen3.5-122B-A10B",
          "qwen3_5-397b-a17b": "Qwen3.5-397B-A17B", "glm-5_3-flash": "GLM-5.3 Flash", "glm-5_3": "GLM-5.3"}


def question_scores(path, text_key):
    d = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind", "question") == "question":
            d[(r[text_key], r["concept"])] = r["p_yes"]
    return d


def remaining_bits(P, game):
    alive, out = set(POOL), [math.log2(len(POOL))]
    for e in game["exchanges"]:
        if e["kind"] != "question":
            continue
        yes = e["answer"] == " yes"
        alive = {c for c in alive if (P[(e["text"], c)] > 0.5) == yes}
        out.append(math.log2(max(len(alive), 1)))
    return out


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_remaining(cfg, plt):
    scorers = [("Jev", question_scores(OUT / "jev_scores.jsonl", "question")),
               ("Qwen3.8-27B", question_scores(OUT / "qwen38_scores.jsonl", "text"))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    for ax, (name, P) in zip(axes, scorers):
        rates, ends = [], []
        for i, m in enumerate(cfg.models):
            curves = [remaining_bits(P, g) for g in json.loads((OUT / f"games_{cell(m, 'no_clues', False)}.json").read_text())["games"]]
            rates += [(c[0] - c[-1]) / (len(c) - 1) for c in curves if len(c) > 1]
            ks = [k for k in range(15) if sum(len(c) > k for c in curves) >= 3]
            ys = [statistics.mean(c[k] for c in curves if len(c) > k) for k in ks]
            ax.plot(ks, ys, color=SERIES[i], lw=2, label=LABELS[slug(m)], solid_capstyle="round")
            ends.append([ys[-1], ks[-1], ys[-1], LABELS[slug(m)]])
        ends.sort()                                   # dodge end labels: min 0.42 bits apart
        for j in range(1, len(ends)):
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.42)
        for y_label, k, y, text in ends:
            ax.annotate(text, (k, y), xytext=(k + 0.35, y_label), textcoords="data", fontsize=7, color=INK,
                        va="center", arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
        rate = statistics.median(rates)
        ax.plot([0, 14], [9, 9 - 14 * rate], color=MUTED, lw=1.2, ls="--")
        ax.annotate(f"median pace {rate:.2f} bits/question", (14, max(0, 9 - 14 * rate)), xytext=(-4, 8),
                    textcoords="offset points", fontsize=7, color=MUTED, ha="right")
        ax.axhline(0, color=INK, lw=0.8)
        ax.axvline(14, color=GRID, lw=1)
        ax.text(14, 8.7, "q14 (turn 15 = guess)", fontsize=7, color=MUTED, ha="right")
        ax.set_title(f"scored by {name}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("questions asked", fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("bits left in the 512-secret pool (hard count)", fontsize=9, color=INK)
    axes[0].set_xlim(0, 16.5)
    axes[0].set_ylim(-0.3, 9.4)
    axes[0].set_yticks(range(0, 10))
    axes[1].legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    fig.suptitle("No-clue games: uncertainty left after k questions (mean over games still running, n >= 3)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "entropy_remaining_by_question.svg")


def fig_bits(plt):
    rows = []
    for name, f, tag in (("Jev", "compare_oracles_qwen38.json", "jev"), ("Qwen3.8-27B", "compare_oracles_qwen38.json", "qwen"),
                         ("Qwen3.5-9B", "compare_oracles.json", "qwen")):
        p = json.loads((OUT / f).read_text())["pooled"]
        rows.append((name, p[f"{tag}_clues_question_hard"], p[f"{tag}_clues_clue_hard"]))
    fig, ax = plt.subplots(figsize=(7.2, 4))
    w = 0.34
    for i, (name, q, c) in enumerate(rows):
        for j, (val, color, lab) in enumerate(((q, SERIES[0], "yes/no question"), (c, SERIES[1], "clue"))):
            x = i + (j - 0.5) * (w + 0.04)
            ax.bar(x, val["mean"], width=w, color=color, label=lab if i == 0 else None, edgecolor="#fcfcfb", lw=2)
            ax.plot(x, val["median"], "o", ms=6, color=INK, mec="#fcfcfb", mew=1.5,
                    label="median" if (i == 0 and j == 1) else None)
            ax.annotate(f"{val['mean']:.2f}", (x, val["mean"]), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=9, color=INK)
    ax.set_ylabel("hard bits per observation (clue games)", fontsize=9, color=INK)
    ax.set_ylim(0, 3)
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("A clue vs a yes/no question: bits removed, by scorer (bars: mean, dots: median)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "bits_question_vs_clue.svg")


def fig_no_clue_hard_soft(cfg, plt):
    """No-clue games, Jev, all 1,796 THINGS: hard (log2 consistent count) and
    soft (posterior entropy) per question; per-model mean where >= 3 games
    run, pooled mean (bold) with its n per point."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True, sharey=True)
    data = {m: json.loads((OUT / f"entropy_{cell(m, 'no_clues', False)}.json").read_text())["games"] for m in cfg.models}
    h0 = math.log2(1796)
    for ax, (title, key) in zip(axes, (("hard entropy: log2 |consistent concepts|", "hard"), ("soft entropy: H(posterior)", "soft"))):
        val = (lambda st: math.log2(max(st["n_consistent"], 1))) if key == "hard" else (lambda st: st["H"])
        pooled = {}
        for i, m in enumerate(cfg.models):
            curves = [[val(st) for st in g["steps"]] for g in data[m]]
            for c in curves:
                for k, v in enumerate(c):
                    pooled.setdefault(k, []).append(v)
            ks = [k for k in range(13) if sum(len(c) > k for c in curves) >= 3]
            ax.plot(ks, [statistics.mean(c[k] for c in curves if len(c) > k) for k in ks], color=SERIES[i], lw=1.4,
                    alpha=0.85, label=LABELS[slug(m)])
        ks = [k for k in sorted(pooled) if len(pooled[k]) >= 5]
        ys = [statistics.mean(pooled[k]) for k in ks]
        ax.plot(ks, ys, color=INK, lw=2.6, label="pooled mean", zorder=5)
        for k, y in zip(ks, ys):
            ax.annotate(f"{len(pooled[k])}", (k, h0 + 0.35), ha="center", fontsize=7, color=MUTED)
        ax.text(0, h0 + 0.75, "n games per point (pooled):", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("questions asked", fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("bits over all 1,796 THINGS (scored by Jev)", fontsize=9, color=INK)
    axes[0].set_ylim(0, h0 + 1.1)
    axes[0].set_xlim(-0.3, 12)
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("No-clue games: entropy after k questions (per-model lines where >= 3 games, pooled where >= 5)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "entropy_no_clues_hard_soft.svg")


def fig_bits_vs_length(cfg, plt):
    """No-clue games: per-game mean soft bits per question vs the turn the
    game ended (unsolved = hollow, at the 12-turn cap), one panel per scorer."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, (name, prefix) in zip(axes, (("Jev", "entropy_"), ("Qwen3.8-27B", "entropy_qwen38_"))):
        for i, m in enumerate(cfg.models):
            games = {g["game_id"]: g for g in json.loads((OUT / f"games_{cell(m, 'no_clues', False)}.json").read_text())["games"]}
            for g in json.loads((OUT / f"{prefix}{cell(m, 'no_clues', False)}.json").read_text())["games"]:
                bits = [s["bits"] for s in g["steps"][1:] if s.get("kind") in (None, "question")]
                if not bits:
                    continue
                gm = games[g["game_id"]]
                solved = gm["winner"] == "auditor"
                ax.plot(statistics.mean(bits), gm["end_turn"] if solved else cfg.max_questions, "o", ms=7,
                        mfc=SERIES[i] if solved else "none", mec=SERIES[i], mew=1.6,
                        )
        ax.set_title(f"scored by {name}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("mean soft bits per question (game)", fontsize=9, color=INK)
        style(ax)
    for i, m in enumerate(cfg.models):
        axes[1].plot([], [], "o", ms=7, color=SERIES[i], label=LABELS[slug(m)])
    axes[1].plot([], [], "o", ms=7, mfc="none", mec=MUTED, label="unsolved (shown at 12)")
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("turn the game ended", fontsize=9, color=INK)
    axes[0].set_ylim(0, 13)
    axes[0].set_xlim(0, 2.2)
    fig.suptitle("No-clue games: does information per question predict game length?", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "bits_vs_game_length.svg")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = AblationConfig.load()
    fig_remaining(cfg, plt)
    fig_bits(plt)
    fig_no_clue_hard_soft(cfg, plt)
    fig_bits_vs_length(cfg, plt)
    print("wrote figures/entropy_remaining_by_question.svg, bits_question_vs_clue.svg, "
          "entropy_no_clues_hard_soft.svg, bits_vs_game_length.svg")


if __name__ == "__main__":
    main()
