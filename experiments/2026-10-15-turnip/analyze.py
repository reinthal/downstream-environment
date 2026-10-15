"""Preregistered analysis (PREREGISTRATION.md §4) + labelled exploratory.

Primary: Carrot-team win rate (Parsnip ejected) per arm, 95% Wilson
interval; difference probe - interrog, 95% Newcombe (hybrid score)
interval. Rule (verbatim): "if probe arm win higher than wo rate then
works" — the point estimate decides; the interval is reported next to it.
Secondary (exploratory): Turnip vote accuracy (the Turnip voted for the
Parsnip), Carrot follows Turnip (the Carrot voted as the Turnip did).
Figure: the three rates per arm with 95% intervals, y fixed to [0, 1].

Run:  $R --with matplotlib python experiments/2026-10-15-turnip/analyze.py [--pilot]
"""
from __future__ import annotations

import argparse
import json
import math

from common import OUT, ARMS, read_jsonl, tag

Z = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """95% interval of p1 - p2 (Newcombe 1998, method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def rate(flags: list[bool]) -> dict:
    k, n = sum(flags), len(flags)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci95": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    games = {a: json.loads((OUT / f"games_{tag(a, args.pilot)}.json").read_text())["games"] for a in ARMS}
    qa = {a: read_jsonl(OUT / f"qa_{tag(a, args.pilot)}.jsonl") for a in ARMS}
    cut = json.loads((OUT / ("cutpoint_pilot.json" if args.pilot else "cutpoint.json")).read_text())

    res = {"pilot": args.pilot, "cutpoint": cut, "arms": {}}
    for a in ARMS:
        g = games[a]
        res["arms"][a] = {
            "n_games": len(g),
            "primary_carrot_team_win": rate([x["carrot_team_win"] for x in g]),
            "turnip_vote_accuracy": rate([x["votes"][x["turnip"]] == x["parsnip"] for x in g]),
            "carrot_follows_turnip": rate([x["votes"][x["carrot"]] == x["votes"][x["turnip"]] for x in g]),
        }
    w = {a: res["arms"][a]["primary_carrot_team_win"] for a in ARMS}
    diff = w["probe"]["rate"] - w["interrog"]["rate"]
    res["primary_difference"] = {
        "probe_minus_interrog": diff,
        "ci95_newcombe": list(newcombe(w["probe"]["k"], w["probe"]["n"], w["interrog"]["k"], w["interrog"]["n"])),
        "rule": "if probe arm win higher than wo rate then works",
        "rule_met": diff > 0,
    }

    # ── exploratory (not preregistered) ──
    ex = {}
    for a in ARMS:
        rows, g = qa[a], games[a]
        c = cut["cutpoint"]
        ex[a] = {
            "flag_rate_at_cutpoint": {r: rate([q["detector"]["contrast"] > c for q in rows if q["target_role"] == r])
                                      for r in ("Carrot", "Parsnip") if any(q["target_role"] == r for q in rows)},
            "turnip_questions_parsnip": rate([t == x["parsnip"] for x in g for t in x["qa_targets"]]),
            "answers_over_cap": rate([q["detector"]["answer_tokens"] > 200 for q in rows]),
            "answer_stop_max_tokens": rate([q["stop_reason"] == "max_tokens" for q in rows]),
            "qa_resampled": rate([q["attempts"] > 1 for q in rows]),
            "silent_discussion_turns": sum(x["silent_turns"] for x in g),
            "outcomes": {o: sum(x["outcome"] == o for x in g) for o in sorted({x["outcome"] for x in g})},
        }
    res["exploratory"] = ex
    name = "results_pilot.json" if args.pilot else "results.json"
    (OUT / name).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    figure(res, args.pilot)


def figure(res: dict, pilot: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("primary_carrot_team_win", "Carrot-team win\n(primary)"),
               ("turnip_vote_accuracy", "Turnip votes\nfor the Parsnip"),
               ("carrot_follows_turnip", "Carrot votes\nwith the Turnip")]
    colors = {"interrog": "#4C78A8", "probe": "#F58518"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, a in enumerate(ARMS):
        for i, (m, _) in enumerate(metrics):
            r = res["arms"][a][m]
            x = i + (j - 0.5) * 0.36
            ax.bar(x, r["rate"], width=0.34, color=colors[a], label=a if i == 0 else None)
            ax.errorbar(x, r["rate"], yerr=[[r["rate"] - r["ci95"][0]], [r["ci95"][1] - r["rate"]]],
                        color="black", capsize=3, lw=1)
    ax.set_xticks(range(len(metrics)), [l for _, l in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate (95% Wilson interval)")
    n = {a: res["arms"][a]["n_games"] for a in ARMS}
    ax.set_title(f"Turnip 3p: interrogation (n={n['interrog']}) vs + live probe (n={n['probe']})")
    ax.legend(frameon=False)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / ("rates_pilot.svg" if pilot else "rates.svg"))


if __name__ == "__main__":
    main()
