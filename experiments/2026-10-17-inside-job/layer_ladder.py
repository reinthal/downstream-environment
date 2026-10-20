"""Detector-quality ladder across layers on the crew-of-5 pilot (pilot v3).

collect (GPU, .venv): for the pilot's sampled turns (score_offline.sampled:
every entry, answer and mole turn, a share of loyal talk) read the speaker's
dialogue + the follow-up question + "No" (as the live scorer does) and save
the residual activations on the "No" at layers cfg.ladder_layers.

analyze (CPU): per layer three detectors —
  transfer   the roleplay-trained follow-up probe (probes/, 2026-10-12)
  god        the best linear probe for this game: logistic regression trained
             on this game's own turn labels (lie = the mole's false claim,
             wrong entry or lying answer), standardized, C = 1 / cfg.reg_coeff,
             5-fold cross-fitted with whole games held out
  random     cfg.ladder_random random unit directions (standardized space)
and per detector:
  lie AUROC  per turn kind (talk / entry / answer) and all turns
  top-1      in liar / silent mole games, the agent with the highest mean
             score over its scored turns is the mole
Random directions: mean and 2.5 / 97.5 percentiles over directions.

Output: activations_ladder.npz + ladder_rows.jsonl (collect; activations are not
committed), ladder_report.json, figure_ladder.png (analyze)

Run:  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python experiments/2026-10-17-inside-job/layer_ladder.py collect
      .venv/bin/python experiments/2026-10-17-inside-job/layer_ladder.py analyze
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

from common import OUT, REPO, load_config, read_jsonl
from score_offline import sampled

KINDS = ("talk", "entry", "answer")
KEYS = ("game_id", "turn", "mission", "kind", "speaker", "role", "mole_type", "lie", "loyal_error", "honest_wrong")


def collect(cfg):
    from linear_probes.activations import collect_activations, load_truncated_decoder
    from linear_probes.dyl_probe import encode_no_span, with_followup
    turns = [t for t in read_jsonl(OUT / "transcripts_trunk_pilot.jsonl") if sampled(t, cfg.pilot_loyal_talk_sample)]
    model, tok = load_truncated_decoder(cfg.model_id, max(cfg.ladder_layers), dtype=cfg.dtype, device=cfg.device,
                                        truncate=cfg.truncate_layers)
    enc = [encode_no_span(tok, with_followup([*t["input_messages"], {"role": "assistant", "content": t["text"]}]),
                          max_len=cfg.max_len, enable_thinking=False) for t in turns]
    order = np.argsort([len(e[0]) for e in enc])
    X = collect_activations(model, tok, [enc[i] for i in order], cfg.ladder_layers, batch_size=cfg.batch_size)
    inv = np.argsort(order)
    np.savez(OUT / "activations_ladder.npz", **{f"l{L}": X[L][inv].astype(np.float16) for L in cfg.ladder_layers})
    with open(OUT / "ladder_rows.jsonl", "w") as f:
        for t in turns:
            f.write(json.dumps({k: t.get(k) for k in KEYS}) + "\n")
    print(f"collected {len(turns)} turns at layers {cfg.ladder_layers}", flush=True)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def top1(rows, s):
    per, mole = defaultdict(lambda: defaultdict(list)), {}
    for r, v in zip(rows, s):
        if r["mole_type"] in ("liar", "silent", "incentive"):
            per[r["game_id"]][r["speaker"]].append(v)
            if r["role"] == "mole":
                mole[r["game_id"]] = r["speaker"]
    return float(np.mean([max(p, key=lambda k: np.mean(p[k])) == mole[g] for g, p in per.items()]))


def metrics(rows, s):
    y = np.array([bool(r["lie"]) for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    out = {"lie_all": auroc(y, s), "top1": top1(rows, s)}
    for k in KINDS:
        m = kinds == k
        out[f"lie_{k}"] = auroc(y[m], s[m])
    return out


def analyze(cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    from linear_probes.probes import LinearProbe
    rows = read_jsonl(OUT / "ladder_rows.jsonl")
    acts = np.load(OUT / "activations_ladder.npz")
    keep = np.array([r["mole_type"] != "open" for r in rows])        # the open saboteur is a separate control
    rows = [r for r, k in zip(rows, keep) if k]
    y = np.array([bool(r["lie"]) for r in rows])
    groups = np.array([r["game_id"] for r in rows])
    rng = np.random.default_rng(cfg.seed)
    rep = {"n_turns": len(rows), "n_lies": int(y.sum()), "layers": {}}
    for L in cfg.ladder_layers:
        X = acts[f"l{L}"].astype(np.float32)[keep]
        probe = LinearProbe.load(str(REPO / f"probes/qwen3.8-27b_layer{L}_followup_logistic_regression.npz"))
        s_tr = ((X - probe.mu) / probe.sd) @ probe.w + probe.b
        s_god = np.zeros(len(rows))
        for tr, te in GroupKFold(5).split(X, y, groups):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            clf = LogisticRegression(C=1.0 / cfg.reg_coeff, fit_intercept=False, max_iter=5000)
            clf.fit((X[tr] - mu) / sd, y[tr])
            s_god[te] = ((X[te] - mu) / sd) @ clf.coef_[0]
        Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
        U = rng.standard_normal((cfg.ladder_random, X.shape[1]))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        rnd = [metrics(rows, Z @ u) for u in U]
        rep["layers"][L] = {
            "transfer": metrics(rows, s_tr),
            "god": metrics(rows, s_god),
            "random": {k: [round(float(np.mean([r[k] for r in rnd if r[k] is not None])), 3),
                           *[round(float(v), 3) for v in np.percentile([r[k] for r in rnd if r[k] is not None],
                                                                        [2.5, 97.5])]]
                       for k in rnd[0]},
        }
        for d in ("transfer", "god"):
            rep["layers"][L][d] = {k: None if v is None else round(v, 3) for k, v in rep["layers"][L][d].items()}
        print(L, json.dumps(rep["layers"][L]), flush=True)
    (OUT / "ladder_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    figure(rep, cfg)


def figure(rep, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ls = cfg.ladder_layers
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in ((axes[0], "lie_all", "lie AUROC (all scored turns)"),
                           (axes[1], "top1", "mole identification (top mean score)")):
        for d, col in (("god", "#2F7D4F"), ("transfer", "#2563A8")):
            ax.plot(Ls, [rep["layers"][L][d][key] for L in Ls], "-o", color=col,
                    label={"god": "god probe (trained on this game)", "transfer": "roleplay-trained probe"}[d])
        r = [rep["layers"][L]["random"][key] for L in Ls]
        ax.fill_between(Ls, [x[1] for x in r], [x[2] for x in r], color="0.85", label="random directions (95%)")
        ax.plot(Ls, [x[0] for x in r], "--", color="0.5")
        ax.set_xticks(Ls), ax.set_ylim(0, 1.02), ax.set_xlabel("layer"), ax.set_title(title, fontsize=11)
        ax.axhline(0.5 if key == "lie_all" else 0.2, color="0.6", lw=0.8, ls=":")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Inside Job pilot v3: detector quality by layer", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "figure_ladder.png", dpi=150)


if __name__ == "__main__":
    cfg = load_config()
    {"collect": collect, "analyze": analyze}[sys.argv[1]](cfg)
