"""EXPLORATORY (not in PREREGISTRATION.md; asked for after the run).

Posterior entropy over turns for every game of the ablation, scored with
TypeSafe's Jev (twenty_questions.entropy): each question a guesser asked is
evaluated against all 1,796 THINGS concepts, and the posterior after the
actual answers gives H_t (bits) per turn. Clues and clue requests are not
part of the yes/no channel and are ignored; a clue request turn carries no
question, so H is unchanged over it.

    uv run --project environments/twenty_questions python experiments/2026-10-17-20q-clue-ablation/entropy_analysis.py

Writes: jev_scores.jsonl (cache), entropy_<cell>.json, entropy_summary.json,
figures/entropy_by_turn.svg. Needs TYPESAFE_API_KEY (repo .env).
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics

from common import ENV, OUT, REPO, AblationConfig, cell, slug

from twenty_questions.entropy import EntropyConfig, analyse

CACHE = OUT / "jev_scores.jsonl"


def load_key():
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("TYPESAFE_API_KEY="):
            os.environ["TYPESAFE_API_KEY"] = line.split("=", 1)[1].strip()
            return
    raise SystemExit("TYPESAFE_API_KEY not in .env")


def cell_games(cfg, model, arm):
    return json.loads((OUT / f"games_{cell(model, arm, False)}.json").read_text())["games"]


def main():
    load_key()
    cfg = AblationConfig.load()
    ecfg = EntropyConfig.from_yaml(ENV / "configs" / "entropy.yaml")
    every = [g for m in cfg.models for a in cfg.arms for g in cell_games(cfg, m, a)]
    first = asyncio.run(analyse(ecfg, every, CACHE))      # scores every unique question once
    print(f"scored: {first['input_tokens']} input tokens")
    h0 = first["H0"]
    summary = {"entropy_config": first["config"], "H0": h0, "n_concepts": first["n_concepts"],
               "input_tokens": first["input_tokens"], "cells": []}
    for m in cfg.models:
        for a in cfg.arms:
            name = cell(m, a, False)
            res = asyncio.run(analyse(ecfg, cell_games(cfg, m, a), CACHE))   # all cached
            (OUT / f"entropy_{name}.json").write_text(json.dumps(res, indent=1) + "\n")
            # H by question number (step k = after k questions), games drop out when they end
            by_step: dict[int, list[float]] = {}
            for g in res["games"]:
                for s in g["steps"]:
                    by_step.setdefault(s["step"], []).append(s["H"])
            bits = [s["bits"] for g in res["games"] for s in g["steps"][1:]]
            final = [g["steps"][-1] for g in res["games"]]
            summary["cells"].append({
                "model": m, "arm": a,
                "mean_bits_per_question": statistics.mean(bits) if bits else None,
                "median_bits_per_question": statistics.median(bits) if bits else None,
                "n_questions": len(bits),
                "mean_H_by_step": {k: statistics.mean(v) for k, v in sorted(by_step.items())},
                "n_by_step": {k: len(v) for k, v in sorted(by_step.items())},
                "final_H": [round(s["H"], 3) for s in final],
                "final_rank_secret": [s.get("rank_secret") for s in final],
                "final_secret_consistent": [s.get("secret_consistent") for s in final],
            })
    (OUT / "entropy_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"H0 = {h0:.2f} bits over {summary['n_concepts']} concepts")
    print(f"{'model':20} {'arm':9} {'q':>4} {'bits/q':>6} {'med':>5} {'H@3':>5} {'H@6':>5} {'H@11':>5} "
          f"{'final H':>7} {'median rank':>11} {'secret consistent':>17}")
    for c in summary["cells"]:
        h = c["mean_H_by_step"]
        get = lambda k: f"{h[k]:5.2f}" if k in h else "    -"
        ranks = [r for r in c["final_rank_secret"] if r is not None]
        cons = [x for x in c["final_secret_consistent"] if x is not None]
        print(f"{slug(c['model']):20} {c['arm']:9} {c['n_questions']:>4} {c['mean_bits_per_question']:6.2f} "
              f"{c['median_bits_per_question']:5.2f} {get(3)} {get(6)} {get(11)} "
              f"{statistics.mean(c['final_H']):7.2f} {statistics.median(ranks):11.0f} "
              f"{sum(cons)}/{len(cons):<14}")
    figure(cfg, summary)


def figure(cfg, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for ax, arm in zip(axes, cfg.arms):
        for c in summary["cells"]:
            if c["arm"] != arm:
                continue
            steps = [int(k) for k in c["mean_H_by_step"]]
            ax.plot(steps, list(c["mean_H_by_step"].values()), marker="o", ms=3, label=slug(c["model"]))
        ax.set_title(f"{arm}: mean posterior entropy after k questions")
        ax.set_xlabel("questions asked (k)")
        ax.axhline(0, color="grey", lw=0.5)
    axes[0].set_ylabel("H (bits, over all THINGS)")
    axes[0].set_xlim(0, cfg.max_questions)
    axes[0].set_ylim(0, summary["H0"] + 0.5)
    axes[1].legend(fontsize=7, loc="lower left")
    fig.text(0.5, -0.02, "games drop out of the mean when they end; clue turns ask no question", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "entropy_by_turn.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
