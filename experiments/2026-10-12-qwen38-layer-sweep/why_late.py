"""Why does the follow-up probe work late and not at layers 16-24? (CPU,
.venv; exploratory, after the run.) From the saved activations, per layer:

  in_domain_games   LR trained on the games themselves (Parsnip vs Carrot),
                    5-fold cross-validated with whole games held out — is the
                    information linearly present at all?
  transfer_games    the roleplay-trained probe on the games (= analyze.py)
  cos_meandiff      cosine between the roleplay mean difference (deceptive -
                    honest) and the game mean difference (Parsnip - Carrot),
                    raw space — does "lying" in roleplay point the same way
                    as "Parsnip" in the game?
  dprime_games      separation along the game mean-difference direction:
                    |mean difference| / pooled within-class s.d.
  shift_share       share of the games' total variance that the class shift
                    accounts for (large -> many random directions separate)

Output: why_late.json

Run:  .venv/bin/python experiments/2026-10-12-qwen38-layer-sweep/why_late.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import OUT, load_config, read_jsonl


def standardized_lr(cfg, X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=1.0 / cfg.reg_coeff, fit_intercept=False, max_iter=5000, random_state=42)
    clf.fit((X - mu) / sd, y)
    return lambda Z: ((Z - mu) / sd) @ clf.coef_[0]


def main():
    cfg = load_config()
    zr, zg = np.load(OUT / "activations_roleplaying.npz"), np.load(OUT / "activations_games.npz")
    layers = [int(L) for L in zr["layers"]]
    graded = read_jsonl(OUT / "judge_roleplaying.jsonl")
    y_r = np.array([{"deceptive": 1, "honest": 0, "ambiguous": -1}[graded[i]["label"]] for i in zr["rows"]])
    turns = read_jsonl(OUT / "transcripts_base.jsonl")
    rows_g = [turns[i] for i in zg["rows"]]
    y_g = np.array([r["role"] == "Parsnip" for r in rows_g], dtype=int)
    games = np.array([r["game_id"] for r in rows_g])
    results = json.loads((OUT / "results.json").read_text())["per_layer"]

    out = {}
    for j, L in enumerate(layers):
        Xr, Xg = zr["X"][y_r >= 0, j], zg["X"][:, j]
        yr = y_r[y_r >= 0]

        cv = np.zeros(len(y_g))
        for tr, te in GroupKFold(5).split(Xg, y_g, games):
            cv[te] = standardized_lr(cfg, Xg[tr], y_g[tr])(Xg[te])

        d_r = Xr[yr == 1].mean(0) - Xr[yr == 0].mean(0)
        d_g = Xg[y_g == 1].mean(0) - Xg[y_g == 0].mean(0)
        u = d_g / np.linalg.norm(d_g)
        proj = Xg @ u
        pooled = np.sqrt((proj[y_g == 1].var() + proj[y_g == 0].var()) / 2)
        total_var = Xg.var(0).sum()
        p = y_g.mean()
        out[L] = {
            "in_domain_games": round(float(roc_auc_score(y_g, cv)), 4),
            "transfer_games": results[str(L)]["games"]["auroc"],
            "cos_meandiff": round(float(d_r @ d_g / np.linalg.norm(d_r) / np.linalg.norm(d_g)), 3),
            "dprime_games": round(float(np.linalg.norm(d_g) / pooled), 2),
            "shift_share": round(float(p * (1 - p) * (d_g @ d_g) / total_var), 4),
        }
        print(L, out[L], flush=True)
    (OUT / "why_late.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
