"""Layer-48 follow-up probe vs the model's own answer, P(Yes), in normal and
deny games (CPU, .venv). Exploratory throughout (no decision rule was
preregistered); the designated primary metric is computed first.

Per arm (normal, deny), message AUROC Parsnip vs Carrot with a 95% game
bootstrap (cfg.n_boot resamples, seed 0) for: the probes at cfg.layers
(loaded via probes/, scored as log-odds) and P(Yes) (log P(Yes) - log P(No)).

  primary          deny arm: AUROC(layer cfg.primary_layer) - AUROC(P(Yes)),
                   paired (the same game resamples for both detectors)
  diff_normal      the same difference in the normal arm
  top1             share of games where the highest mean score is the Parsnip
  says_yes         share of turns, by role, where log P(Yes) > log P(No)

Output: results.json, figure_detectors.png

Run:  .venv/bin/python experiments/2026-10-13-l48-vs-selfreport/analyze.py
"""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from common import OUT, REPO, load_config, scored_turns

from linear_probes.probes import LinearProbe  # noqa: E402

ARMS_ANALYSED = ("normal", "deny")


def boot_idx(groups: np.ndarray, n_boot: int) -> list[np.ndarray]:
    ids = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(0)
    return [np.concatenate([rows[g] for g in rng.choice(ids, len(ids))]) for _ in range(n_boot)]


def ci(values: list[float]) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def main():
    cfg = load_config()
    probes = {L: LinearProbe.load(str(REPO / cfg.probe_paths[str(L)])) for L in cfg.layers}
    results = {"arms": {}}
    for arm in ARMS_ANALYSED:
        turns = scored_turns(arm)
        y = np.array([r["role"] == "Parsnip" for r in turns], dtype=int)
        games = np.array([r["game_id"] for r in turns])
        z = np.load(OUT / f"activations_{arm}.npz")
        if [int(L) for L in z["layers"]] != cfg.layers or len(z["X"]) != len(turns):
            raise ValueError(f"activations_{arm}.npz does not match the config / transcripts")
        with open(OUT / f"pyes_{arm}.csv") as f:
            prow = list(csv.DictReader(f))
        if [(p["game_id"], int(p["round"]), p["speaker"]) for p in prow] != \
                [(r["game_id"], r["round"], r["speaker"]) for r in turns]:
            raise ValueError(f"pyes_{arm}.csv rows do not match the transcripts")

        scores = {f"L{L}": (z["X"][:, j] - probes[L].mu) / probes[L].sd @ probes[L].w + probes[L].b
                  for j, L in enumerate(cfg.layers)}
        scores["P(Yes)"] = np.array([float(p["score"]) for p in prow])
        boots = boot_idx(games, cfg.n_boot)
        auroc = {}
        for name, s in scores.items():
            means: dict = {}
            for g, r, v in zip(games, turns, s):
                means.setdefault(g, {}).setdefault(r["speaker"], []).append(v)
            parsnip = {r["game_id"]: r["speaker"] for r in turns if r["role"] == "Parsnip"}
            top1 = np.mean([max(d, key=lambda sp: np.mean(d[sp])) == parsnip[g] for g, d in means.items()])
            auroc[name] = {"auroc": round(float(roc_auc_score(y, s)), 4),
                           "ci95": ci([roc_auc_score(y[b], s[b]) for b in boots]),
                           "top1": round(float(top1), 3)}
        primary = f"L{cfg.primary_layer}"
        diffs = [roc_auc_score(y[b], scores[primary][b]) - roc_auc_score(y[b], scores["P(Yes)"][b]) for b in boots]
        says_yes = {role: round(float(np.mean([float(p["score"]) > 0 for p in prow if p["role"] == role])), 3)
                    for role in ("Parsnip", "Carrot")}
        results["arms"][arm] = {
            "n_games": len(set(games)), "n_turns": len(y), "n_parsnip_turns": int(y.sum()),
            "detectors": auroc,
            "diff_primary_layer_minus_pyes": {
                "value": round(auroc[primary]["auroc"] - auroc["P(Yes)"]["auroc"], 4), "ci95": ci(diffs)},
            "says_yes": says_yes,
        }
        print(arm, json.dumps(results["arms"][arm]), flush=True)

    results["primary"] = {"metric": f"deny arm: AUROC(L{cfg.primary_layer}) - AUROC(P(Yes))",
                          **results["arms"]["deny"]["diff_primary_layer_minus_pyes"]}
    (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    figure(results, cfg)
    print("primary:", results["primary"])


def figure(results: dict, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [f"L{L}" for L in cfg.layers] + ["P(Yes)"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, arm in enumerate(ARMS_ANALYSED):
        d = results["arms"][arm]["detectors"]
        x = np.arange(len(names)) + (k - 0.5) * 0.3
        a = np.array([[d[n]["auroc"], *d[n]["ci95"]] for n in names])
        ax.errorbar(x, a[:, 0], yerr=[a[:, 0] - a[:, 1], a[:, 2] - a[:, 0]], fmt="o", capsize=3,
                    label=f"{arm} games")
    ax.axhline(0.5, color="0.5", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("message AUROC, Parsnip vs Carrot")
    ax.set_title("Follow-up probe by layer vs the model's own answer")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "figure_detectors.png", dpi=150)


if __name__ == "__main__":
    main()
