"""Fit the 1-7 suspicion scale on the alpaca control (CPU, cached activations).

Cutpoint fitting / grading / serialization live in the `auditor` package;
this script wires in the experiment's detectors: the trained LR and DoM
probes (via probes/) and the five per-seed-group random-direction controls
(reusing the 2026-09-22 RandomProbe with its training activations).

Reads the cached 2026-09-24 alpaca activations; no model, no GPU.

Run:  uv run --no-sync python experiments/2026-09-25/calibrate_scale.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
PREV = OUT.parents[0] / "2026-09-22"
sys.path.insert(0, str(PREV))          # score_messages (RandomProbe), common
sys.path.insert(0, str(REPO))

from score_messages import RandomProbe  # noqa: E402

from auditor import AuditorConfig, fit_cutpoints, grade, save_scale  # noqa: E402
from linear_probes.probes import LinearProbe  # noqa: E402

ALPACA_ACTS = OUT.parents[0] / "2026-09-24" / "activations_alpaca.npz"


def main():
    cfg = AuditorConfig.load(OUT / "config.json")
    cfg.save(OUT)

    X = np.load(ALPACA_ACTS)[f"X_{cfg.span}"]
    lr = LinearProbe.load(str(cfg.path(cfg.probe_paths["lr"])))
    dom = LinearProbe.load(str(cfg.path(cfg.probe_paths["dom"])))
    X_train = np.load(cfg.path(cfg.probe_paths["train_activations"]))[f"layer_{cfg.layer}"]
    rand = [RandomProbe(g, lr.mu, lr.sd, X_train) for g in range(cfg.seed_groups)]

    q = cfg.scale_quantiles
    cutpoints = {"lr": fit_cutpoints(lr.predict_proba(X), q),
                 "dom": fit_cutpoints(dom.predict_proba(X), q),
                 "random": [fit_cutpoints(r.predict_proba(X), q) for r in rand]}

    path = save_scale(OUT / "suspicion_scale.json", span=cfg.span, quantiles=q,
                      cutpoints=cutpoints, grade_legend=cfg.grade_legend,
                      n_control=int(X.shape[0]))
    print("wrote", path)

    for name in ("lr", "dom"):
        print(f"{name:6s} cutpoints:", np.round(cutpoints[name], 3))
    for g, c in enumerate(cutpoints["random"]):
        print(f"rand_g{g} cutpoints:", np.round(c, 3))

    # sanity: grade distribution of the game messages under this scale
    import csv
    game = list(csv.DictReader(open(cfg.path(cfg.data_dir) / "scores_base.csv")))
    for name, col in (("lr", f"lr_{cfg.span}"), ("dom", f"dom_{cfg.span}"),
                      ("random", f"rand_{cfg.span}")):
        for role in ("Carrot", "Parsnip"):
            gr = [grade(float(r[col]),
                        cutpoints[name] if name != "random"
                        else cutpoints["random"][int(r["seed_group"])])
                  for r in game if r["role"] == role]
            c = Counter(gr)
            dist = " ".join(f"{c.get(k, 0):4d}" for k in range(1, 8))
            print(f"{name:6s} {role:8s} grades 1..7: {dist}  mean={np.mean(gr):.2f}")


if __name__ == "__main__":
    main()
