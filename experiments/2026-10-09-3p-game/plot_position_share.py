"""Exploratory figure: is a round-3 speaking position over-picked? (CPU)

For the first and the last speaker of round 3: over all valid games, the
share in which that speaker was named (by Opus without scores, by Opus with
probe scores, or by the players' own elimination vote), whatever their role.
A uniform pick gives 1/num_players; the last row is how often the Parsnip
really spoke in that position, which is what a perfect auditor would show.
Intervals are 95% game bootstraps (cfg.n_boot resamples, seed 0), as in
analyze.py. Both panels share one x range.

Output: position_share.svg

Run:  .venv-api/bin/python experiments/2026-10-09-3p-game/plot_position_share.py
"""
from __future__ import annotations

import json

import numpy as np

from common import OUT, GameRunConfig, read_jsonl
from plot_scores_by_round import ARM, GRID, INK, MUTED, SURFACE

DOT = "#2a78d6"
REFERENCE = "Parsnip really there"
W, H = 1050, 386
TOP, ROW = 118, 56
PANELS = [(150, 440), (675, 965)]  # x extents of the 0..1 axis
POSITIONS = {"First speaker of round 3": 0, "Last speaker of round 3": -1}


def main():
    cfg = GameRunConfig.load(OUT / "config.json")
    games = {g["game_id"]: g for g in json.loads((OUT / f"games_{ARM}.json").read_text())["games"]}
    names = list(next(iter(games.values()))["roles"])
    audits = [a for a in read_jsonl(OUT / "audit" / f"auditor_{ARM}.jsonl") if a["game_id"] in games]
    deciders = {
        "Opus, no scores": {a["game_id"]: a["guess"] for a in audits if a["condition"] == "none"},
        "Opus, with scores": {a["game_id"]: a["guess"] for a in audits if a["condition"] == "fu_notool"},
        # games where the players' vote eliminated nobody name no one
        "Players' own vote": {gid: g["eliminated"][0] for gid, g in games.items() if g["eliminated"]},
        REFERENCE: {gid: g["parsnip"] for gid, g in games.items()},
    }
    chance = 1 / len(names)

    def share(named: dict, pos: int) -> tuple[int, int, float, float]:
        hits = []
        for gid, g in games.items():
            speaker = names[g["orders"][-1][pos]]
            if gid in named:
                hits.append(named[gid] == speaker)
        v = np.asarray(hits, dtype=float)
        idx = np.random.default_rng(0).integers(0, len(v), (cfg.n_boot, len(v)))
        lo, hi = np.percentile(v[idx].mean(1), [2.5, 97.5])
        return int(v.sum()), len(v), float(lo), float(hi)

    def text(x, yy, s, size=13, fill=MUTED, anchor="middle", weight="normal"):
        return (f'<text x="{x:.1f}" y="{yy:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                f'font-weight="{weight}">{s}</text>')

    bottom = TOP + ROW * len(deciders)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="system-ui, sans-serif">', f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           text(24, 28, "Is a round-3 speaking position over-picked? "
                "(Close to dashed-line = better)", 17, INK, "start", "600"),
           text(24, 48, f"3-player setting, {len(games)} games. Bars = 95% game-bootstrap interval.", 13, MUTED, "start")]
    for (x0, x1), (title, pos) in zip(PANELS, POSITIONS.items()):
        def x(v):
            return x0 + v * (x1 - x0)

        out.append(text(x0, 98, title, 13, INK, "start", "600"))
        for t in (0, 0.25, 0.5, 0.75, 1):
            out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{TOP}" y2="{bottom}" '
                       f'stroke="{GRID}"/>')
            out.append(text(x(t), bottom + 18, f"{t:g}", 12))
        out.append(f'<line x1="{x(chance):.1f}" x2="{x(chance):.1f}" y1="{TOP}" y2="{bottom}" '
                   f'stroke="{MUTED}" stroke-dasharray="4 3"/>')
        out.append(text(x(chance), bottom + 36, "uniform", 12))
        for j, (label, named) in enumerate(deciders.items()):
            k, n, lo, hi = share(named, pos)
            cy = TOP + ROW * (j + 0.5)
            color = MUTED if label == REFERENCE else DOT
            out.append(text(x0 - 14, cy + 4, label, 13, INK, "end"))
            out.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{cy}" y2="{cy}" stroke="{color}" '
                       f'stroke-width="2" stroke-linecap="round"/>')
            out.append(f'<circle cx="{x(k / n):.1f}" cy="{cy}" r="5" fill="{color}" stroke="{SURFACE}" '
                       f'stroke-width="2"><title>{label}: {k}/{n} = {k / n:.2f}, 95% CI {lo:.2f}-{hi:.2f}'
                       f'</title></circle>')
            out.append(text(x1 + 12, cy + 4, f"{k}/{n} = {k / n:.2f}", 12, MUTED, "start"))
            print(f"{title} | {label}: {k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]")
    out.append("</svg>")

    (OUT / "position_share.svg").write_text("\n".join(out) + "\n")
    print("wrote position_share.svg")


if __name__ == "__main__":
    main()
