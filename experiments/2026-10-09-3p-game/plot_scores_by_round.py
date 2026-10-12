"""Exploratory figure: follow-up-probe score distribution per discussion round. (CPU)

Two panels with one shared y range (raw log-odds): each round split by role,
and each round split by speaking position. Violins are Gaussian KDEs
(Silverman bandwidth) on one common density scale; the tick is the median.
Written as plain SVG because this host has no plotting library.

Output: scores_by_round.svg

Run:  .venv-api/bin/python experiments/2026-10-09-3p-game/plot_scores_by_round.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

from common import OUT

ARM = "base"
ROLE_COLOR = {"Carrot": "#eb6834", "Parsnip": "#f2d55c"}
POS_COLOR = ["#52514e", "#8a8983", "#b9b8b1"]
POS_LABEL = ["1st", "2nd", "3rd"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"

W, H = 1040, 520
TOP, BOTTOM = 96, 64
PANELS = [(70, 480), (590, 1010)]  # x extents


def kde(v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    bw = 0.9 * min(v.std(ddof=1), np.subtract(*np.percentile(v, [75, 25])) / 1.34) * len(v) ** -0.2
    return np.exp(-0.5 * ((grid[:, None] - v[None, :]) / bw) ** 2).sum(1) / (len(v) * bw * np.sqrt(2 * np.pi))


def main():
    games = {g["game_id"] for g in json.loads((OUT / f"games_{ARM}.json").read_text())["games"]}
    with open(OUT / f"scores_{ARM}.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["game_id"] in games]
    score = np.array([float(r["followup_logodds"]) for r in rows])
    rnd = np.array([int(r["round"]) for r in rows])
    pos = np.array([int(r["position"]) for r in rows])
    role = np.array([r["role"] for r in rows])
    rounds = sorted(set(rnd))

    lo, hi = np.floor(score.min()) - 1, np.ceil(score.max()) + 1
    grid = np.linspace(lo, hi, 300)

    def y(v):
        return TOP + (hi - v) / (hi - lo) * (H - TOP - BOTTOM)

    role_groups = {(r, k): score[(rnd == r) & (role == k)] for r in rounds for k in ROLE_COLOR}
    pos_groups = {(r, p): score[(rnd == r) & (pos == p)] for r in rounds for p in range(len(POS_LABEL))}
    dens = {k: kde(v, grid) for k, v in {**role_groups, **pos_groups}.items()}
    dmax = max(d.max() for d in dens.values())

    def violin(cx: float, d: np.ndarray, half: float, side: int, color: str, tip: str) -> str:
        """side: -1 left half, +1 right half, 0 both."""
        w = d / dmax * half
        ys = [y(g) for g in grid]
        left = [(cx - (0 if side > 0 else x), yy) for x, yy in zip(w, ys)]
        right = [(cx + (0 if side < 0 else x), yy) for x, yy in zip(w, ys)]
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in left + right[::-1])
        return (f'<polygon points="{pts}" fill="{color}" fill-opacity="0.8" stroke="{SURFACE}" '
                f'stroke-width="2"><title>{tip}</title></polygon>')

    def median_tick(cx: float, v: np.ndarray, d: np.ndarray, half: float, side: int) -> str:
        m = float(np.median(v))
        w = float(np.interp(m, grid, d)) / dmax * half
        x0, x1 = cx - (0 if side > 0 else w), cx + (0 if side < 0 else w)
        return f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{y(m):.1f}" y2="{y(m):.1f}" stroke="{INK}" stroke-width="1.5"/>'

    def tip(name: str, v: np.ndarray) -> str:
        return f"{name}: n={len(v)}, median {np.median(v):.2f}, mean {v.mean():.2f}, sd {v.std(ddof=1):.2f}"

    def text(x, yy, s, size=13, fill=MUTED, anchor="middle", weight="normal"):
        return (f'<text x="{x:.1f}" y="{yy:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                f'font-weight="{weight}">{s}</text>')

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="system-ui, sans-serif">', f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           text(PANELS[0][0], 28, "(3-player setting) Follow-up probe score by discussion round", 17, INK, "start", "600"),
           text(PANELS[0][0], 48, f"{len(score)} announcements, {len(games)} games; raw log-odds, higher = "
                "more likely deceptive; dark tick = median", 13, MUTED, "start")]

    for i, (x0, x1) in enumerate(PANELS):
        for t in np.arange(np.ceil(lo / 4) * 4, hi + 0.01, 4):
            out.append(f'<line x1="{x0}" x2="{x1}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="{GRID}"/>')
            out.append(text(x0 - 8, y(t) + 4, f"{t:.0f}", 12, MUTED, "end"))
        step = (x1 - x0) / len(rounds)
        for j, r in enumerate(rounds):
            cx = x0 + step * (j + 0.5)
            label = f"Round {r + 1}"
            out.append(text(cx, H - BOTTOM + 22, label, 13, INK))
            if i == 0:
                for k, side in (("Carrot", -1), ("Parsnip", 1)):
                    v = role_groups[(r, k)]
                    out.append(violin(cx, dens[(r, k)], step * 0.4, side, ROLE_COLOR[k], tip(f"{label}, {k}", v)))
                    out.append(median_tick(cx, v, dens[(r, k)], step * 0.4, side))
                    out.append(text(cx + side * step * 0.22, H - BOTTOM + 40, f"{np.median(v):.1f}", 12))
            else:
                for p, name in enumerate(POS_LABEL):
                    v = pos_groups[(r, p)]
                    px = cx + (p - 1) * step * 0.3
                    out.append(violin(px, dens[(r, p)], step * 0.13, 0, POS_COLOR[p], tip(f"{label}, {name} speaker", v)))
                    out.append(median_tick(px, v, dens[(r, p)], step * 0.13, 0))
                    out.append(text(px, H - BOTTOM + 40, f"{np.median(v):.1f}", 12))

    # panel titles and legends
    x0 = PANELS[0][0]
    out.append(text(x0, 78, "By role", 13, INK, "start", "600"))
    for n, (k, c) in enumerate(ROLE_COLOR.items()):
        out.append(f'<rect x="{x0 + 70 + n * 90}" y="68" width="12" height="12" rx="2" fill="{c}"/>')
        out.append(text(x0 + 88 + n * 90, 78, k, 13, MUTED, "start"))
    x0 = PANELS[1][0]
    out.append(text(x0, 78, "By speaking position in the round", 13, INK, "start", "600"))
    for n, (k, c) in enumerate(zip(POS_LABEL, POS_COLOR)):
        out.append(f'<rect x="{x0 + 230 + n * 60}" y="68" width="12" height="12" rx="2" fill="{c}"/>')
        out.append(text(x0 + 248 + n * 60, 78, k, 13, MUTED, "start"))
    out.append(text(PANELS[0][0] - 8, H - BOTTOM + 40, "median", 12, MUTED, "end"))
    out.append("</svg>")

    (OUT / "scores_by_round.svg").write_text("\n".join(out) + "\n")
    print("wrote scores_by_round.svg")


if __name__ == "__main__":
    main()
