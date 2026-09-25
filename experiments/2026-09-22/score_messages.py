"""Score every game message with the trained probes + a random-direction control.

For each recorded message, re-encodes the EXACT generation context (system
prompt with role, user prompt, private reasoning, public announcement) and
mean-pools layer-L residual activations over:
  * pub  — the public announcement tokens only (primary; what the detector
           reports to the auditor)
  * full — the whole assistant turn incl. the private think block (diagnostic)

Probes: logistic regression, diff-of-means (trained on instructed pairs), and
one random direction per seed group (calibrated on the training activations to
have centred, unit-spread projections — same treatment the DoM probe gets, so
scores land in a comparable range).

Run:  CUDA_VISIBLE_DEVICES=5,6 uv run --no-sync python experiments/2026-09-22/score_messages.py --arm base
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, collect_activations, encode_with_spans, load_model, read_jsonl

from linear_probes.probes import LinearProbe

PROBE_DIR = OUT / "probes"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class RandomProbe:
    """Random unit direction in standardized space, one per seed group.
    Calibrated on the instructed-pairs training activations: projections are
    shifted/scaled to zero mean, unit std (mirrors DoM's calibration, minus
    the class information)."""

    def __init__(self, seed_group: int, mu, sd, X_train):
        rng = np.random.default_rng(10_000 + seed_group)
        d = rng.standard_normal(mu.shape[0])
        d /= np.linalg.norm(d)
        p = ((X_train - mu) / sd) @ d
        scale = p.std() + 1e-9
        self.mu, self.sd = mu, sd
        self.w, self.b = d / scale, -p.mean() / scale

    def predict_proba(self, X):
        return _sigmoid(((X - self.mu) / self.sd) @ self.w + self.b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["base", "steered"], default="base")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    recs = read_jsonl(OUT / f"transcripts_{args.arm}.jsonl")
    print(f"{len(recs)} messages ({args.arm})", flush=True)

    lr = LinearProbe.load(str(PROBE_DIR / "probe_logistic_regression.npz"))
    dom = LinearProbe.load(str(PROBE_DIR / "probe_diff_of_means.npz"))
    layer = lr.config.layer
    tr = np.load(PROBE_DIR / "train_activations.npz")
    X_train = tr[f"layer_{layer}"]
    rand = {g: RandomProbe(g, lr.mu, lr.sd, X_train) for g in range(5)}

    model, tok = load_model()
    encoded = []
    for r in recs:
        msgs = list(r["input_messages"]) + [{
            "role": "assistant",
            "content": r["announcement"],
            "reasoning_content": r["reasoning"],
        }]
        ids, full, pub = encode_with_spans(tok, msgs, max_len=4096)
        encoded.append((ids, [full, pub]))
    acts = collect_activations(model, tok, encoded, [layer], batch_size=args.batch_size)[layer]
    X_full, X_pub = acts[:, 0], acts[:, 1]
    np.savez(OUT / f"message_activations_{args.arm}.npz", X_full=X_full, X_pub=X_pub,
             layer=layer)
    print("activations:", X_pub.shape, flush=True)

    out_path = OUT / f"scores_{args.arm}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game_id", "seed_group", "round", "position", "speaker", "role",
                    "silent", "lr_pub", "lr_full", "dom_pub", "dom_full",
                    "rand_pub", "rand_full"])
        lp, lf = lr.predict_proba(X_pub), lr.predict_proba(X_full)
        dp, df = dom.predict_proba(X_pub), dom.predict_proba(X_full)
        for i, r in enumerate(recs):
            g = r["seed_group"]
            rp = rand[g].predict_proba(X_pub[i:i + 1])[0]
            rf = rand[g].predict_proba(X_full[i:i + 1])[0]
            w.writerow([r["game_id"], g, r["round"], r["position"], r["speaker"],
                        r["role"], int(not r["announcement"]),
                        f"{lp[i]:.6f}", f"{lf[i]:.6f}", f"{dp[i]:.6f}",
                        f"{df[i]:.6f}", f"{rp:.6f}", f"{rf:.6f}"])
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
