"""Why the layer-48 probe beats P(Yes) (CPU, .venv; exploratory, after the run).

From the saved activations, P(Yes) and the 2026-10-12 probes, per arm:

  hidden_lies     AUROC on Parsnip turns where the model answers "No" to the
                  follow-up (log P(Yes) < log P(No)) vs all Carrot turns —
                  lies the model will not admit; P(Yes) on the same subset
  incremental     game-grouped 5-fold CV logistic regression on [P(Yes)] vs
                  [P(Yes), probe score]: does the probe add to self-report?
  spearman        rank correlation of probe score and P(Yes), within Parsnip
                  and within Carrot turns
  arm_shift       AUROC(deny-arm scores vs normal-arm scores) within a role:
                  Carrot prompts are identical across arms (should be ~0.5);
                  a Parsnip shift reflects the deny line and/or its effect
  unembed_cos     cosine of the raw-space probe direction (w / sd) with the
                  direction that raises the Yes-minus-No logit,
                  (1 + final-norm weight) * (W_U[Yes] - W_U[No]); random
                  directions give |cos| ~ 1/sqrt(d) ~ 0.014

Output: why_l48.json

Run:  HF_HUB_OFFLINE=1 .venv/bin/python experiments/2026-10-13-l48-vs-selfreport/why_l48.py
"""
from __future__ import annotations

import csv
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import OUT, REPO, load_config, scored_turns

from linear_probes.probes import LinearProbe  # noqa: E402


def unembed_direction(cfg) -> np.ndarray:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)
    (yes,), (no,) = (tok(a, add_special_tokens=False)["input_ids"] for a in ("Yes", "No"))
    index = json.load(open(hf_hub_download(cfg.model_id, "model.safetensors.index.json",
                                           revision=cfg.model_revision)))["weight_map"]

    def tensor(name: str):
        path = hf_hub_download(cfg.model_id, index[name], revision=cfg.model_revision)
        with safe_open(path, "pt") as f:
            return f.get_slice(name)

    W = tensor("lm_head.weight")
    g = tensor("model.language_model.norm.weight")[:].float().numpy()
    d = (W[yes:yes + 1].float().numpy() - W[no:no + 1].float().numpy())[0]
    return (1.0 + g) * d


def cv_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(max_iter=5000).fit((X[tr] - mu) / sd, y[tr])
        s[te] = clf.decision_function((X[te] - mu) / sd)
    return round(float(roc_auc_score(y, s)), 4)


def main():
    cfg = load_config()
    probes = {L: LinearProbe.load(str(REPO / cfg.probe_paths[str(L)])) for L in cfg.layers}
    u = unembed_direction(cfg)
    out = {"unembed_cos": {}, "arms": {}}
    rng = np.random.default_rng(0)
    rand_cos = [abs(v @ u) / np.linalg.norm(v) / np.linalg.norm(u) for v in rng.standard_normal((200, len(u)))]
    out["unembed_cos"]["random_95pct_abs"] = round(float(np.percentile(rand_cos, 95)), 4)
    for L, p in probes.items():
        w_raw = p.w / p.sd
        out["unembed_cos"][f"L{L}"] = round(float(w_raw @ u / np.linalg.norm(w_raw) / np.linalg.norm(u)), 4)

    scores = {}
    for arm in ("normal", "deny"):
        turns = scored_turns(arm)
        y = np.array([r["role"] == "Parsnip" for r in turns])
        games = np.array([r["game_id"] for r in turns])
        X = np.load(OUT / f"activations_{arm}.npz")["X"]
        with open(OUT / f"pyes_{arm}.csv") as f:
            pyes = np.array([float(r["score"]) for r in csv.DictReader(f)])
        s = {f"L{L}": (X[:, j] - p.mu) / p.sd @ p.w + p.b for j, (L, p) in enumerate(probes.items())}
        scores[arm] = (y, s)
        hidden = y & (pyes < 0)
        sub = hidden | ~y
        res = {"n_parsnip": int(y.sum()), "n_parsnip_denies": int(hidden.sum()),
               "hidden_lies": {"P(Yes)": round(float(roc_auc_score(y[sub], pyes[sub])), 4)},
               "incremental": {"P(Yes)": cv_auroc(pyes[:, None], y.astype(int), games)},
               "spearman": {}}
        for name, v in s.items():
            res["hidden_lies"][name] = round(float(roc_auc_score(y[sub], v[sub])), 4)
            res["incremental"][f"P(Yes)+{name}"] = cv_auroc(np.c_[pyes, v], y.astype(int), games)
            res["spearman"][name] = {role: round(float(spearmanr(v[m], pyes[m])[0]), 3)
                                     for role, m in (("Parsnip", y), ("Carrot", ~y))}
        out["arms"][arm] = res
        print(arm, json.dumps(res), flush=True)

    out["arm_shift"] = {}
    for name in scores["normal"][1]:
        out["arm_shift"][name] = {}
        for role, want in (("Parsnip", True), ("Carrot", False)):
            a = scores["normal"][1][name][scores["normal"][0] == want]
            b = scores["deny"][1][name][scores["deny"][0] == want]
            out["arm_shift"][name][role] = round(float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))],
                                                                     np.r_[a, b])), 4)
    print("arm_shift (AUROC deny vs normal, within role):", out["arm_shift"])
    print("unembed_cos:", out["unembed_cos"])
    (OUT / "why_l48.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
