"""Exploratory figure: total probe score of the player Opus named, for true
positives (named the Parsnip) against false positives (named a Carrot). (CPU)

Total score = sum of the named player's raw follow-up log-odds over the
discussion rounds. Two panels, per audit condition: the raw total, and the
total minus the mean total of the game's players. The panels show different
quantities, so each has its own y range, but both span the same number of
log-odds. Violins are Gaussian KDEs on one common density scale; the tick is
the median.

Output: tp_fp_total_score.svg

Run:  .venv-api/bin/python experiments/2026-10-09-3p-game/plot_tp_fp_total_score.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from common import OUT, read_jsonl
from plot_scores_by_round import ARM, GRID, INK, MUTED, ROLE_COLOR, SURFACE, kde

# a TP names the Parsnip, an FP names a Carrot: same colours as the role figure
OUTCOME = {"FP": ("named a Carrot", ROLE_COLOR["Carrot"], -1), "TP": ("named the Parsnip", ROLE_COLOR["Parsnip"], 1)}
COND_LABEL = {"none": "Opus, no scores", "fu_notool": "Opus, with scores"}

W, H = 1040, 540
TOP, BOTTOM = 96, 84
PANELS = [(70, 480), (590, 1010)]  # x extents
SPAN = 28  # log-odds shown in each panel


def main():
    games = {g["game_id"]: g for g in json.loads((OUT / f"games_{ARM}.json").read_text())["games"]}
    raw = json.loads((OUT / f"raw_fu_{ARM}.json").read_text())
    total = {}
    for gid in games:
        per = defaultdict(float)
        for r in raw[gid]:
            per[r["player"]] += r["score"]
        total[gid] = dict(per)
    centred = {gid: {p: s - np.mean(list(t.values())) for p, s in t.items()} for gid, t in total.items()}

    quantities = [("Total score of the named player", total),
                  ("Same, minus the mean total of the game's three players", centred)]
    groups = {}
    for a in read_jsonl(OUT / "audit" / f"auditor_{ARM}.jsonl"):
        if a["game_id"] not in games:
            continue
        outcome = "TP" if a["guess"] == games[a["game_id"]]["parsnip"] else "FP"
        for q, (_, table) in enumerate(quantities):
            groups.setdefault((q, a["condition"], outcome), []).append(table[a["game_id"]][a["guess"]])
    groups = {k: np.array(v) for k, v in groups.items()}

    ranges, grids, dens = [], [], {}
    for q in range(len(quantities)):
        allv = np.concatenate([v for k, v in groups.items() if k[0] == q])
        mid = round((allv.min() + allv.max()) / 2)
        ranges.append((mid - SPAN / 2, mid + SPAN / 2))
        grids.append(np.linspace(*ranges[q], 300))
    for k, v in groups.items():
        dens[k] = kde(v, grids[k[0]])
    dmax = max(d.max() for d in dens.values())

    def text(x, yy, s, size=13, fill=MUTED, anchor="middle", weight="normal"):
        return (f'<text x="{x:.1f}" y="{yy:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                f'font-weight="{weight}">{s}</text>')

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="system-ui, sans-serif">', f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           text(PANELS[0][0], 28, "(3-player setting) Probe total of Opus's pick: TP vs FP", 17, INK, "start", "600"),
           text(PANELS[0][0], 48, f"{len(games)} games, one audit per condition; sum of raw log-odds over the "
                "three rounds, higher = more likely deceptive; dark tick = median", 13, MUTED, "start")]
    for n, (k, (desc, color, _)) in enumerate(OUTCOME.items()):
        out.append(f'<rect x="{PANELS[1][1] - 330 + n * 180}" y="18" width="12" height="12" rx="2" fill="{color}"/>')
        out.append(text(PANELS[1][1] - 312 + n * 180, 28, f"{k}: {desc}", 13, MUTED, "start"))

    for q, ((x0, x1), (title, _)) in enumerate(zip(PANELS, quantities)):
        lo, hi = ranges[q]

        def y(v):
            return TOP + (hi - v) / (hi - lo) * (H - TOP - BOTTOM)

        out.append(text(x0, 78, title, 13, INK, "start", "600"))
        for t in np.arange(np.ceil(lo / 4) * 4, hi + 0.01, 4):
            out.append(f'<line x1="{x0}" x2="{x1}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="{GRID}"/>')
            out.append(text(x0 - 8, y(t) + 4, f"{t:.0f}", 12, MUTED, "end"))
        step = (x1 - x0) / len(COND_LABEL)
        half = step * 0.4
        for j, (cond, label) in enumerate(COND_LABEL.items()):
            cx = x0 + step * (j + 0.5)
            out.append(text(cx, H - BOTTOM + 22, label, 13, INK))
            for k, (_, color, side) in OUTCOME.items():
                v, d = groups[(q, cond, k)], dens[(q, cond, k)]
                w = d / dmax * half
                ys = [y(g) for g in grids[q]]
                edge = [(cx + side * x, yy) for x, yy in zip(w, ys)]
                pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in [(cx, ys[0]), *edge, (cx, ys[-1])])
                tip = (f"{label}, {k}: n={len(v)}, median {np.median(v):.2f}, mean {v.mean():.2f}, "
                       f"sd {v.std(ddof=1):.2f}")
                out.append(f'<polygon points="{pts}" fill="{color}" fill-opacity="0.8" stroke="{SURFACE}" '
                           f'stroke-width="2"><title>{tip}</title></polygon>')
                m = float(np.median(v))
                wm = float(np.interp(m, grids[q], d)) / dmax * half
                out.append(f'<line x1="{cx:.1f}" x2="{cx + side * wm:.1f}" y1="{y(m):.1f}" y2="{y(m):.1f}" '
                           f'stroke="{INK}" stroke-width="1.5"/>')
                out.append(text(cx + side * step * 0.2, H - BOTTOM + 42, f"{k} n={len(v)}", 12))
                out.append(text(cx + side * step * 0.2, H - BOTTOM + 60, f"{m:.1f}", 12))
    out.append(text(PANELS[0][0] - 8, H - BOTTOM + 60, "median", 12, MUTED, "end"))
    out.append("</svg>")

    (OUT / "tp_fp_total_score.svg").write_text("\n".join(out) + "\n")
    print("wrote tp_fp_total_score.svg")


if __name__ == "__main__":
    main()
