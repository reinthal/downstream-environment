"""Alpaca 1%-FPR thresholds for the LR probe, both spans (CPU, cached acts).

Writes alpaca_thresholds.json next to this script; audit_task.py reads it.
The full-span value reproduces the 2026-09-25 scale's top cutpoint (p99 on
the same 2000 control rollouts).

Run:  uv run --no-sync python experiments/2026-10-02/compute_thresholds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from linear_probes.probes import LinearProbe  # noqa: E402

ALPACA_ACTS = REPO / "experiments/2026-09-24/activations_alpaca.npz"
FPR = 0.01


def main():
    X = np.load(ALPACA_ACTS)
    lr = LinearProbe.load(str(REPO / "probes/qwen3.5-27b_layer16_logistic_regression.npz"))
    out = {"fpr": FPR, "n_control": int(X["X_full"].shape[0]),
           "lr_full": float(np.quantile(lr.predict_proba(X["X_full"]), 1 - FPR)),
           "lr_pub": float(np.quantile(lr.predict_proba(X["X_pub"]), 1 - FPR))}
    scale_top = json.load(open(HERE / "suspicion_scale.json"))["cutpoints"]["lr"][-1]
    assert abs(out["lr_full"] - scale_top) < 1e-9, (out["lr_full"], scale_top)
    path = HERE / "alpaca_thresholds.json"
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path, out)


if __name__ == "__main__":
    main()
