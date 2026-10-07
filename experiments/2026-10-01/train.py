"""Fit the follow-up probe on graded roleplaying replies (CPU, .venv).

As the paper's Experiment: the dataset is shuffled (seeded), the last 20% is
held out for validation, ambiguous replies (grade 3-5) are not trained on,
features are standardized, and LogisticRegression(C=1/10, fit_intercept=False).

Output: probe_followup_{off,on}.npz (LinearProbe format), train_{off,on}.json
(label counts, validation AUROC). The thinking-off probe is the main one and
gets a probes/ symlink.

Run:  .venv/bin/python experiments/2026-10-01/train.py --thinking off
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, read_jsonl  # noqa: E402

from auditor.config import AuditorConfig  # noqa: E402
from linear_probes import ProbeConfig  # noqa: E402
from linear_probes.probes import make_probe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    args = ap.parse_args()
    cfg = AuditorConfig.load(OUT / "config.json")

    graded = read_jsonl(OUT / f"judge_roleplaying_{args.thinking}.jsonl")
    X = np.load(OUT / f"activations_followup_roleplaying_{args.thinking}.npz")["X"]
    assert len(graded) == len(X)
    labels = np.array([{"deceptive": 1, "honest": 0}.get(r["label"], -1) for r in graded])

    order = np.random.default_rng(cfg.seed).permutation(len(graded))
    n_val = int(len(order) * 0.2)
    train, val = order[:-n_val], order[-n_val:]
    train = train[labels[train] >= 0]
    val_lab = val[labels[val] >= 0]

    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    clf = LogisticRegression(C=1.0 / cfg.reg_coeff, random_state=42, fit_intercept=False,
                             max_iter=5000)
    clf.fit((X[train] - mu) / sd, labels[train])

    probe = make_probe(ProbeConfig(model_id=cfg.model_id, layer=cfg.layer,
                                   probe_type="logistic_regression"))
    probe.mu, probe.sd, probe.w, probe.b = mu, sd, clf.coef_[0], 0.0
    path = OUT / f"probe_followup_{args.thinking}.npz"
    probe.save(str(path))

    s = probe.predict_proba(X[val_lab])
    summary = {
        "labels_all": dict(Counter(r["label"] for r in graded)),
        "n_train": {"deceptive": int((labels[train] == 1).sum()),
                    "honest": int((labels[train] == 0).sum())},
        "n_val": {"deceptive": int((labels[val_lab] == 1).sum()),
                  "honest": int((labels[val_lab] == 0).sum())},
        "val_auroc": round(float(roc_auc_score(labels[val_lab], s)), 4),
        "val_indices": val.tolist(),
    }
    (OUT / f"train_{args.thinking}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print({k: v for k, v in summary.items() if k != "val_indices"}, "->", path)


if __name__ == "__main__":
    main()
