"""Auditor accuracy per scenario x detector (CPU, .venv). Exploratory.

Per (scenario, detector):
  accuracy        share of games where the auditor named the Parsnip, over
                  completed audits, 95% game bootstrap (cfg.n_boot, seed 0);
                  accuracy_all counts failed / unanswered audits as wrong
  failed          audits with no verdict (auditor or player could not finish)
  questions       mean questions the auditor asked per game
  carrot_win      share of games the Carrots won by the players' own vote
  probe_top1      (probe sets) the player with the highest mean probe score
                  over their messages is the Parsnip
  probe_auroc     (probe sets) message AUROC Parsnip vs Carrot

Output: results{_pilot}.json, figure_accuracy{_pilot}.png

Run:  .venv/bin/python experiments/2026-10-14-auditor-in-game/analyze.py [--pilot]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from common import OUT, load_config, read_jsonl
from game import SETS

ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
LABEL = {"S0": "S0 passive", "S1": "S1 final questions", "S2": "S2 per-round questions",
         "S3": "S3 question budget", "S4": "S4 private interviews", "S5": "S5 role claims",
         "S6": "S6 public recommendation"}


def boot(x: np.ndarray, n: int) -> list[float]:
    rng = np.random.default_rng(0)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(v), 3) for v in np.percentile(means, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    sfx = "_pilot" if args.pilot else ""

    audits, players, scored = defaultdict(list), defaultdict(list), defaultdict(list)
    for s in SETS:
        path = OUT / f"games_{s}{sfx}.json"
        if not path.exists():
            print("missing", path.name)
            continue
        for g in json.loads(path.read_text())["games"]:
            if "scenario" not in g:                              # the whole game errored
                for sc in ({"base": ["S0", "S1"], "claim": ["S5"]}.get(s) or [s.split("_")[0]]):
                    for d in (["none", "probe"] if s in ("base", "claim") else [s.split("_")[1]]):
                        audits[(sc, d)].append({"correct": None, "failed": g["failed"], "questions_asked": 0})
                continue
            if g["detector"] == "players_vote":
                players[(g["scenario"], "shared")].append(g)
            else:
                audits[(g["scenario"], g["detector"])].append(g)
                if "winner" in g:
                    players[(g["scenario"], g["detector"])].append(g)
        tpath = OUT / f"transcripts_{s}{sfx}.jsonl"
        if tpath.exists():
            for t in read_jsonl(tpath):
                if "score" in t:
                    scored[(t["scenario"], t["detector"], t["game_id"])].append(t)

    results = {}
    for (sc, d), rows in sorted(audits.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        ok = np.array([r["correct"] for r in rows if r["correct"] is not None], dtype=float)
        allx = np.array([bool(r["correct"]) for r in rows], dtype=float)
        res = {"n_games": len(rows), "failed": sum(r["correct"] is None for r in rows),
               "fail_reasons": sorted({(r["failed"] or "no verdict")[:120] for r in rows if r["correct"] is None}),
               "accuracy": round(float(ok.mean()), 3) if len(ok) else None,
               "ci95": boot(ok, cfg.n_boot) if len(ok) > 1 else None,
               "accuracy_all": round(float(allx.mean()), 3),
               "questions": round(float(np.mean([r["questions_asked"] for r in rows])), 2)}
        pv = players.get((sc, d)) or (players.get((sc, "shared")) if sc in ("S0", "S1", "S5") else None)
        if pv:
            res["carrot_win"] = round(float(np.mean([p.get("winner") == "Carrot" for p in pv])), 3)
        msgs = [(gid, t) for (s2, d2, gid), ts in scored.items() if s2 == sc and d2 == d for t in ts]
        if d == "probe" and msgs:
            y = np.array([t["role"] == "Parsnip" for _, t in msgs])
            s = np.array([t["score"] for _, t in msgs])
            res["probe_auroc"] = round(float(roc_auc_score(y, s)), 3) if 0 < y.sum() < len(y) else None
            per = defaultdict(lambda: defaultdict(list))
            for gid, t in msgs:
                per[gid][t["speaker"]].append(t["score"])
            parsnip = {gid: t["speaker"] for gid, t in msgs if t["role"] == "Parsnip"}
            res["probe_top1"] = round(float(np.mean([max(p, key=lambda k: np.mean(p[k])) == parsnip.get(gid)
                                                     for gid, p in per.items()])), 3)
        results[f"{sc}_{d}"] = res
        print(f"{LABEL[sc]:<26} {d:<5}", json.dumps(res), flush=True)

    (OUT / f"results{sfx}.json").write_text(json.dumps(results, indent=1) + "\n")
    figure(results, sfx)


def figure(results: dict, sfx: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, d in enumerate(("none", "probe")):
        xs, ys, lo, hi = [], [], [], []
        for i, sc in enumerate(ORDER):
            r = results.get(f"{sc}_{d}")
            if not r or r["accuracy"] is None:
                continue
            ci = r["ci95"] or [r["accuracy"], r["accuracy"]]
            xs.append(i + (k - 0.5) * 0.3)
            ys.append(r["accuracy"])
            lo.append(r["accuracy"] - ci[0])
            hi.append(ci[1] - r["accuracy"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", capsize=3,
                    label="text only" if d == "none" else "with layer-48 probe scores")
    ax.axhline(0.2, color="0.5", lw=0.8, ls="--", label="chance (1/5)")
    ax.set_xticks(range(len(ORDER)), [LABEL[s] for s in ORDER], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("auditor accuracy (names the Parsnip)")
    ax.set_title("Auditor in the 5-player game, Qwen3.8-27B" + (" — PILOT" if sfx else ""))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"figure_accuracy{sfx}.png", dpi=150)


if __name__ == "__main__":
    main()
