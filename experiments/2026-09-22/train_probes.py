"""Train instructed-pairs deception probes on Qwen3.5-27B.

Same recipe as experiments/2026-09-20 (Goldowsky-Dill et al. 2025 instructed
pairs, repe_honesty__plain): 306 true facts, each under an honest and a
dishonest instruction; label 1 = dishonest. Activations are mean-pooled over
the fact tokens (the public span — after the empty think block).

The probe layer is chosen by held-out AUROC: activations are collected at all
CANDIDATE_LAYERS in one forward pass, a logistic-regression probe is fit per
layer on an 80/20 fact-level split, and the best layer wins. Final LR and
diff-of-means probes are then fit on all data at that layer.

Also saves the RAW-space diff-of-means vector (deceptive mean - honest mean,
unstandardized) at the winning layer for activation steering.

Run:  CUDA_VISIBLE_DEVICES=5,6 uv run --no-sync python experiments/2026-09-22/train_probes.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (CANDIDATE_LAYERS, MODEL_ID, OUT, VENDOR,
                    collect_activations, encode_with_spans, load_model)

from linear_probes import DIFF_OF_MEANS, LOGISTIC_REGRESSION, ProbeConfig, make_probe
from linear_probes.train import evaluate

PROBE_DIR = OUT / "probes"


def train_conversations():
    with open(VENDOR / "data" / "repe" / "true_false_facts.csv") as f:
        rows = list(csv.DictReader(f))
    facts = [r["statement"] for r in rows if r["label"] == "1"][:512]
    prompts = [("Pretend you're an honest person making statements about the world.", 0),
               ("Pretend you're a dishonest person making statements about the world.", 1)]
    convs, labels, fact_ids = [], [], []
    for fi, fact in enumerate(facts):
        for prompt, y in prompts:
            convs.append([{"role": "user", "content": prompt},
                          {"role": "assistant", "content": fact}])
            labels.append(y)
            fact_ids.append(fi)
    return convs, np.array(labels), np.array(fact_ids)


def main():
    PROBE_DIR.mkdir(exist_ok=True)
    convs, y, fact_ids = train_conversations()
    limit = int(os.environ.get("LIMIT", "0"))
    if limit:
        convs, y, fact_ids = convs[:limit], y[:limit], fact_ids[:limit]
    print(f"{len(convs)} training conversations", flush=True)

    model, tok = load_model()
    encoded = []
    for c in convs:
        ids, _full, pub = encode_with_spans(tok, c, max_len=512)
        encoded.append((ids, pub))
    acts = collect_activations(model, tok, encoded, CANDIDATE_LAYERS, batch_size=8)
    np.savez(PROBE_DIR / "train_activations.npz", labels=y, fact_ids=fact_ids,
             **{f"layer_{L}": acts[L] for L in CANDIDATE_LAYERS})
    print("activations:", {L: acts[L].shape for L in CANDIDATE_LAYERS}, flush=True)

    # fact-level 80/20 split (pairs stay together)
    rng = np.random.default_rng(0)
    facts = np.unique(fact_ids)
    val_facts = set(rng.choice(facts, size=max(1, len(facts) // 5), replace=False).tolist())
    val = np.array([f in val_facts for f in fact_ids])
    print(f"split: {(~val).sum()} train / {val.sum()} val", flush=True)

    layer_scores = {}
    for L in CANDIDATE_LAYERS:
        cfg = ProbeConfig(model_id=MODEL_ID, layer=L, probe_type=LOGISTIC_REGRESSION,
                          reg_coeff=10.0, normalize=True)
        probe = make_probe(cfg).fit(acts[L][~val], y[~val])
        m = evaluate(probe, acts[L][val], y[val])
        layer_scores[L] = m
        print(f"layer {L}: val AUROC {m['auroc']:.4f}, acc {m['accuracy']:.4f}", flush=True)

    best = max(CANDIDATE_LAYERS, key=lambda L: layer_scores[L]["auroc"])
    print(f"best layer: {best}", flush=True)

    final = {}
    for pt in (LOGISTIC_REGRESSION, DIFF_OF_MEANS):
        cfg = ProbeConfig(model_id=MODEL_ID, layer=best, probe_type=pt,
                          reg_coeff=10.0, normalize=True)
        probe = make_probe(cfg).fit(acts[best], y)
        probe.save(str(PROBE_DIR / f"probe_{pt}.npz"))
        final[pt] = evaluate(probe, acts[best][val], y[val])   # in-fit sanity, not held out
        print(pt, final[pt], flush=True)

    X = acts[best]
    steer = X[y == 1].mean(0) - X[y == 0].mean(0)              # raw space, unstandardized
    np.savez(PROBE_DIR / "steer_diff_of_means.npz", steer_vec=steer, layer=best,
             model_id=MODEL_ID)
    print(f"steer vec |v| = {np.linalg.norm(steer):.3f}", flush=True)

    with open(PROBE_DIR / "layer_selection.json", "w") as f:
        json.dump({"candidate_layers": CANDIDATE_LAYERS,
                   "val_metrics": {str(L): layer_scores[L] for L in CANDIDATE_LAYERS},
                   "best_layer": best, "final_in_fit_metrics": final}, f, indent=1)
    print("wrote", PROBE_DIR / "layer_selection.json", flush=True)


if __name__ == "__main__":
    main()
