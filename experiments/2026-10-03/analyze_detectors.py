"""Step 1: what information do the detectors' scores carry? (CPU)

  a. message-level separation: Parsnip-vs-Carrot AUROC, 95% CI resampled by
     game, and its percentile among many random directions on the same activations
  b. player-level combination: for mean / max / min / last-round / change
     (round 3 - round 1) of a player's scores: how often the top player is the
     Parsnip, how often the Parsnip is in the top 2, its mean rank
  c. confounds: does the score track message length, reasoning length, round
     or speaking position; does the Parsnip differ on those; does the
     separation survive after regressing them out of the score

Output: step1.json

Run:  .venv/bin/python experiments/2026-10-03/analyze_detectors.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from data import ARMS, OUT, messages, scores

N_RANDOM = 200
N_BOOT = 1000
AGGS = {"mean": np.mean, "max": np.max, "min": np.min,
        "last round": lambda v: v[-1], "change (r3 - r1)": lambda v: v[-1] - v[0]}


def player_table(msgs, s):
    """{game: {speaker: [scores in round order]}}, plus game -> parsnip."""
    g = defaultdict(lambda: defaultdict(list))
    parsnip = {}
    for m, v in sorted(zip(msgs, s), key=lambda t: (t[0]["game_id"], t[0]["round"])):
        g[m["game_id"]][m["speaker"]].append(v)
        if m["role"] == "Parsnip":
            parsnip[m["game_id"]] = m["speaker"]
    return g, parsnip


def game_metrics(msgs, s, agg=np.mean):
    g, parsnip = player_table(msgs, s)
    top1, top2, rank = [], [], []
    for gid, players in g.items():
        val = {p: agg(np.array(v)) for p, v in players.items()}
        order = sorted(val, key=val.get, reverse=True)
        r = order.index(parsnip[gid]) + 1
        top1.append(r == 1)
        top2.append(r <= 2)
        rank.append(r)
    return {"top1": float(np.mean(top1)), "top2": float(np.mean(top2)),
            "mean_rank": float(np.mean(rank)), "n_games": len(top1)}


def boot_auroc(msgs, s, y):
    games = sorted({m["game_id"] for m in msgs})
    idx = defaultdict(list)
    for i, m in enumerate(msgs):
        idx[m["game_id"]].append(i)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[games[j]] for j in rng.integers(0, len(games), len(games))])
        if y[pick].min() != y[pick].max():
            vals.append(roc_auc_score(y[pick], s[pick]))
    return [round(float(x), 3) for x in np.percentile(vals, [2.5, 97.5])]


def residualize(msgs, s):
    """Remove the linear effect of length, reasoning length, round and position."""
    F = np.column_stack([
        np.ones(len(msgs)),
        np.log1p([m["ann_len"] for m in msgs]),
        np.log1p([m["reas_len"] for m in msgs]),
        *[[m["round"] == r for m in msgs] for r in (1, 2)],
        *[[m["position"] == p for m in msgs] for p in (1, 2, 3, 4)],
    ]).astype(float)
    beta, *_ = np.linalg.lstsq(F, s, rcond=None)
    return s - F @ beta


def main():
    out = {}
    for arm in ARMS:
        msgs = messages(arm)
        y = np.array([m["role"] == "Parsnip" for m in msgs])
        res = {}
        for det in ("fu", "lr", "godmode"):
            s = scores(det, arm)
            if s is None:
                continue
            d = {"message_auroc": round(float(roc_auc_score(y, s)), 3),
                 "message_auroc_ci95": boot_auroc(msgs, s, y)}
            if det in ("fu", "lr"):
                null_auc, null_top1 = [], []
                for k in range(N_RANDOM):
                    r = scores(f"rand_{det}", arm, seed=k)
                    null_auc.append(roc_auc_score(y, r))
                    null_top1.append(game_metrics(msgs, r)["top1"])
                d["random_directions"] = {
                    "auroc_middle95": [round(float(x), 3) for x in np.percentile(null_auc, [2.5, 97.5])],
                    "share_auroc_ge_detector": round(float(np.mean(np.array(null_auc) >= d["message_auroc"])), 3),
                    "top1_mean": round(float(np.mean(null_top1)), 3),
                    "top1_middle95": [round(float(x), 3) for x in np.percentile(null_top1, [2.5, 97.5])],
                }
            d["player_level"] = {name: game_metrics(msgs, s, f) for name, f in AGGS.items()}
            if det in ("fu", "lr"):
                d["random_directions"]["share_top1_ge_detector"] = round(float(np.mean(
                    np.array(null_top1) >= d["player_level"]["mean"]["top1"])), 3)
            # confounds
            feats = {"announcement length": [m["ann_len"] for m in msgs],
                     "reasoning length": [m["reas_len"] for m in msgs],
                     "round": [m["round"] for m in msgs], "position": [m["position"] for m in msgs]}
            d["confounds"] = {
                name: {"spearman_with_score": round(float(spearmanr(f, s)[0]), 3),
                       "feature_alone_parsnip_auroc": round(float(roc_auc_score(y, f)), 3)}
                for name, f in feats.items()}
            rs = residualize(msgs, s)
            d["after_removing_confounds"] = {
                "message_auroc": round(float(roc_auc_score(y, rs)), 3),
                "top1 (player mean)": game_metrics(msgs, rs)["top1"]}
            res[det] = d
        out[arm] = res
    (OUT / "step1.json").write_text(json.dumps(out, indent=1) + "\n")

    for arm, res in out.items():
        print(f"\n===== {arm}")
        for det, d in res.items():
            pl = d["player_level"]
            print(f"  {det:8s} msg AUROC {d['message_auroc']:.2f} {d['message_auroc_ci95']}"
                  + (f" | random dirs 95% {d['random_directions']['auroc_middle95']}, share >= {d['random_directions']['share_auroc_ge_detector']}"
                     if "random_directions" in d else ""))
            print("           top1 by aggregation: " + ", ".join(f"{k} {v['top1']:.2f}" for k, v in pl.items())
                  + f" | mean: top2 {pl['mean']['top2']:.2f}, rank {pl['mean']['mean_rank']:.2f}")
            if "random_directions" in d:
                rd = d["random_directions"]
                print(f"           random-direction top1: mean {rd['top1_mean']:.2f}, 95% {rd['top1_middle95']}, share >= detector {rd['share_top1_ge_detector']}")
            print("           confounds (corr with score / feature-alone AUROC): " + ", ".join(
                f"{k} {v['spearman_with_score']:+.2f}/{v['feature_alone_parsnip_auroc']:.2f}"
                for k, v in d["confounds"].items()))
            print(f"           after removing confounds: msg AUROC {d['after_removing_confounds']['message_auroc']:.2f}, "
                  f"top1 {d['after_removing_confounds']['top1 (player mean)']:.2f}")


if __name__ == "__main__":
    main()
