"""Exploratory figure: is a round-3 speaking position over-picked, before
(the older 5-player games) and after (the 3-player games, with the fix in
the reasoning prompt)? (CPU)

For the first and the last speaker of round 3: the share of games in which
Opus, shown no probe scores, named that speaker (whatever their role), minus
the uniform rate 1/num_players. Zero is a random pick. Intervals are 95% game
bootstraps (cfg.n_boot resamples), as in analyze.py; the change (after minus
before) resamples the two sets of games independently (seeds 0 and 1). One
figure per position, both on the same x range.

The 5-player games are the 100 discussion-only games of 2026-09-22 (old
engine and harness), audited by audit_5p/ — the 2026-10-04 Opus config
rerun; only its `none` condition is used. Audits without a tagged answer are
dropped.

Output: position_share_first.svg, position_share_last.svg

Run:  .venv-api/bin/python experiments/2026-10-09-3p-game/plot_position_share.py
"""
from __future__ import annotations

import json

import numpy as np

from common import OUT, REPO, GameRunConfig, read_jsonl
from plot_scores_by_round import ARM, GRID, INK, MUTED, SURFACE

DOT = "#2a78d6"
W, H = 680, 300
TOP, ROW = 100, 50
AXIS = (220, 470)  # x extent of the axis
LIM = 0.3  # axis runs from -LIM to +LIM around the uniform rate
POSITIONS = {"first": ("first speaker", 0), "last": ("last speaker", -1)}  # file suffix -> (label, index)


def picks(games_path, audits_path, pos: int) -> tuple[np.ndarray, float]:
    """(1 per audited game where Opus named the speaker at `pos` of the last round, uniform rate)."""
    games = {g["game_id"]: g for g in json.loads(games_path.read_text())["games"]}
    names = list(next(iter(games.values()))["roles"])
    guess = {a["game_id"]: a["guess"] for a in read_jsonl(audits_path)
             if a["game_id"] in games and a["condition"] == "none" and a["guess"] is not None}
    return (np.asarray([guess[gid] == names[g["orders"][-1][pos]] for gid, g in games.items() if gid in guess],
                       dtype=float), 1 / len(names))


def main():
    cfg = GameRunConfig.load(OUT / "config.json")
    data_5p = REPO / json.loads((OUT / "audit_5p" / "config.json").read_text())["data_dir"]
    settings = {
        "Before: 5 players": (data_5p / f"games_{ARM}.json", OUT / "audit_5p" / f"auditor_{ARM}.jsonl"),
        "After: 3 players + prompt fix": (OUT / f"games_{ARM}.json", OUT / "audit" / f"auditor_{ARM}.jsonl"),
    }

    def text(x, yy, s, size=13, fill=MUTED, anchor="middle", weight="normal"):
        return (f'<text x="{x:.1f}" y="{yy:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                f'font-weight="{weight}">{s}</text>')

    bottom = TOP + ROW * len(settings)
    x0, x1 = AXIS

    def x(v):
        return x0 + (v + LIM) / (2 * LIM) * (x1 - x0)

    for name, (who, pos) in POSITIONS.items():
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
               f'font-family="system-ui, sans-serif">', f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
               text(24, 28, f"Is the {who} of round 3 over-picked? (Close to dashed-line = better)", 17, INK,
                    "start", "600"),
               text(24, 48, f"Opus auditor, no probe scores: share of games naming the {who}, minus the", 13,
                    MUTED, "start"),
               text(24, 66, "random rate (1/5 before, 1/3 after). Bars = 95% game-bootstrap interval.", 13, MUTED,
                    "start")]
        for t in (-0.2, -0.1, 0, 0.1, 0.2):
            out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{TOP}" y2="{bottom}" stroke="{GRID}"/>')
            out.append(text(x(t), bottom + 18, f"{t:+.1f}" if t else "0", 12))
        out.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{TOP}" y2="{bottom}" stroke="{MUTED}" '
                   f'stroke-dasharray="4 3"/>')
        out.append(text(x(0), bottom + 36, "random", 12))
        boots = []
        for j, (label, paths) in enumerate(settings.items()):
            v, chance = picks(*paths, pos)
            idx = np.random.default_rng(j).integers(0, len(v), (cfg.n_boot, len(v)))  # seed 0 before, 1 after
            boots.append(v[idx].mean(1) - chance)
            rate, (lo, hi) = v.mean(), np.percentile(boots[-1], [2.5, 97.5])
            cy = TOP + ROW * (j + 0.5)
            out.append(text(x0 - 14, cy + 4, label, 13, INK, "end"))
            out.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{cy}" y2="{cy}" stroke="{DOT}" '
                       f'stroke-width="2" stroke-linecap="round"/>')
            out.append(f'<circle cx="{x(rate - chance):.1f}" cy="{cy}" r="5" fill="{DOT}" stroke="{SURFACE}" '
                       f'stroke-width="2"><title>{label}: {int(v.sum())}/{len(v)} = {rate:.2f} against '
                       f'{chance:.2f} random</title></circle>')
            out.append(text(x1 + 12, cy + 4, f"{rate - chance:+.2f} ({lo:+.2f} to {hi:+.2f})", 12, MUTED, "start"))
            print(f"{who} | {label}: {int(v.sum())}/{len(v)} = {rate:.3f}, random {chance:.3f}, "
                  f"excess {rate - chance:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        change = boots[1] - boots[0]
        lo, hi = np.percentile(change, [2.5, 97.5])
        out.append(text(x0, bottom + 62, f"Change, after minus before: {change.mean():+.2f} ({lo:+.2f} to {hi:+.2f})",
                        13, INK, "start"))
        print(f"{who} | change after - before: {change.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        out.append("</svg>")
        (OUT / f"position_share_{name}.svg").write_text("\n".join(out) + "\n")
        print(f"wrote position_share_{name}.svg")


if __name__ == "__main__":
    main()
