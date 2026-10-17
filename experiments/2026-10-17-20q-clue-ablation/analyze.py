"""Per model x arm: games, excluded, mean / median end turn (unsolved = the
turn cap), solve rate; the clue-arm mean against the too-easy threshold
(point estimate — the preregistered analysis is qualitative); a figure with
one shared 0..cap y-axis; and clues_{cell}.md, every game's exchanges for
reading.

    uv run --project environments/twenty_questions python experiments/2026-10-17-20q-clue-ablation/analyze.py
"""
from __future__ import annotations

import json
import statistics

from common import OUT, AblationConfig, cell, slug


def load_cell(name: str, cfg: AblationConfig) -> tuple[list[dict], list[dict]]:
    """Games of a cell with re-runs folded in (a re-run replaces its errored seed)."""
    main = json.loads((OUT / f"games_{name}.json").read_text())
    games, excluded = list(main["games"]), []
    for e in main["errors"]:
        retry = OUT / f"games_{name}_retry_s{e['seed']}.json"
        r = json.loads(retry.read_text()) if retry.exists() else {"games": [], "errors": [e]}
        games += r["games"]
        excluded += r["errors"]
    return sorted(games, key=lambda g: g["seed"]), excluded


def end_turn(g: dict, cap: int) -> int:
    return g["end_turn"] if g["winner"] == "auditor" else cap


def main():
    cfg = AblationConfig.load()
    cap = cfg.max_questions
    rows = []
    for model in cfg.models:
        for arm in cfg.arms:
            name = cell(model, arm, pilot=False)
            if not (OUT / f"games_{name}.json").exists():
                rows.append({"model": model, "arm": arm, "missing": True})
                continue
            games, excluded = load_cell(name, cfg)
            turns = [end_turn(g, cap) for g in games]
            n_total = len(games) + len(excluded)
            row = {"model": model, "arm": arm, "n": len(games), "excluded": len(excluded),
                   "void": n_total == 0 or len(excluded) / n_total > cfg.void_excluded_fraction,
                   "mean_end_turn": statistics.mean(turns) if turns else None,
                   "median_end_turn": statistics.median(turns) if turns else None,
                   "solve_rate": sum(g["winner"] == "auditor" for g in games) / len(games) if games else None,
                   "end_turns": turns, "secrets": [g["secret"] for g in games]}
            if arm == "clues" and turns:
                row["too_easy_point_estimate"] = row["mean_end_turn"] < cfg.too_easy_threshold
            rows.append(row)
            with open(OUT / f"clues_{name}.md", "w") as f:
                f.write(f"# {slug(model)} — {arm}\n\n")
                for g in games:
                    f.write(f"## seed {g['seed']}: {g['secret']} — {g['winner']} (turn {g['end_turn']}, "
                            f"{g['end_reason']})\n\n")
                    for e in g["exchanges"]:
                        ans = (e["answer"] or "").strip()
                        f.write(f"- {e['turn']} {e['kind']}: {e['text']}"
                                + (f" -> {ans}" if ans else "") + (f" — clue: {e['clue']}" if e["clue"] else "")
                                + (" [identified]" if e["judged_correct"] else "") + "\n")
                    f.write("\n")
    (OUT / "results.json").write_text(json.dumps({"config": cfg.__dict__, "cells": rows}, indent=1) + "\n")
    print(f"{'model':28} {'arm':9} {'n':>3} {'excl':>4} {'mean':>5} {'median':>6} {'solved':>6}")
    for r in rows:
        if r.get("missing"):
            print(f"{slug(r['model']):28} {r['arm']:9} missing")
            continue
        print(f"{slug(r['model']):28} {r['arm']:9} {r['n']:>3} {r['excluded']:>4} "
              f"{r['mean_end_turn'] or float('nan'):>5.2f} {r['median_end_turn'] or float('nan'):>6} "
              f"{r['solve_rate'] or 0:>6.0%}" + ("  VOID" if r["void"] else "")
              + ("  too easy (<%.1f)" % cfg.too_easy_threshold if r.get("too_easy_point_estimate") else ""))
    figure(cfg, rows)


def figure(cfg: AblationConfig, rows: list[dict]):
    """Mean end turn per model, both arms, individual games as dots; one
    y-axis 0..cap; threshold line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap = cfg.max_questions
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"clues": "#2a6fdb", "no_clues": "#d9822b"}
    for k, arm in enumerate(cfg.arms):
        for i, model in enumerate(cfg.models):
            r = next(x for x in rows if x["model"] == model and x["arm"] == arm)
            if r.get("missing") or not r["n"]:
                continue
            x = i + (k - 0.5) * 0.3
            ax.scatter([x] * r["n"], r["end_turns"], s=12, color=colors[arm], alpha=0.35)
            ax.scatter([x], [r["mean_end_turn"]], s=60, color=colors[arm], marker="D",
                       label=arm if i == 0 else None, edgecolor="black", linewidth=0.5)
    ax.axhline(cfg.too_easy_threshold, color="grey", ls="--", lw=1, label=f"too easy < {cfg.too_easy_threshold}")
    ax.set_xticks(range(len(cfg.models)), [slug(m) for m in cfg.models], rotation=20, ha="right")
    ax.set_ylim(0, cap + 0.5)
    ax.set_ylabel(f"end turn (unsolved = {cap})")
    ax.set_title("Twenty Questions: game length with and without clues")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "end_turn.svg")


if __name__ == "__main__":
    main()
