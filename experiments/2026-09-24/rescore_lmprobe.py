"""Proof that LMProbe reproduces the 2026-09-24 scoring path.

Scores the first N sandbagging rollouts twice on the same slice:

  old — the committed score_rollouts.py path: full (untruncated) model via
        common.load_model, encode_with_spans + collect_activations, probes'
        predict_proba;
  new — LMProbe.from_config(config.json): model truncated at the probe layer,
        same helpers under the hood.

Blocks <= the probe layer are identical in both, so activations and scores
must match. Prints max |delta| and PASS/FAIL; also shows the drift against the
committed scores_sandbagging.csv (expected nonzero in late decimals — the full
run batched all 1000 rollouts together, which changes bf16 padding).

Run:  CUDA_VISIBLE_DEVICES=5,6,7 uv run --no-sync python \
          experiments/2026-09-24/rescore_lmprobe.py [--n 12]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
PREV = REPO / "experiments" / "2026-09-22"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PREV))

from common import collect_activations, encode_with_spans, load_model, read_jsonl  # noqa: E402

from linear_probes import ExperimentConfig, LinearProbe, LMProbe  # noqa: E402


def to_messages(r: dict) -> list[dict]:
    return list(r["input_messages"]) + [{
        "role": "assistant", "content": r["public"],
        "reasoning_content": r["reasoning"],
    }]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    cfg = ExperimentConfig.load(OUT / "config.json")
    recs = read_jsonl(OUT / "rollouts_sandbagging.jsonl")[:args.n]
    convs = [to_messages(r) for r in recs]
    print(f"{len(recs)} rollouts, layer {cfg.layer}", flush=True)

    # ── new path: LMProbe (truncated decoder) ────────────────────────────────
    lr = LMProbe.from_config(cfg, probe="lr")
    dom = lr.with_probe(REPO / cfg.probe_paths["dom"])
    X_new = lr.collect(convs, spans=("full", "public"))
    new = {("lr", s): lr.probe.predict_proba(X_new[:, i]) for i, s in enumerate(("full", "public"))}
    new |= {("dom", s): dom.probe.predict_proba(X_new[:, i]) for i, s in enumerate(("full", "public"))}
    del lr, dom
    import torch
    torch.cuda.empty_cache()

    # ── old path: score_rollouts.py verbatim (full model) ────────────────────
    lrp = LinearProbe.load(str(REPO / cfg.probe_paths["lr"]))
    domp = LinearProbe.load(str(REPO / cfg.probe_paths["dom"]))
    model, tok = load_model()
    encoded = []
    for msgs in convs:
        ids, full, pub = encode_with_spans(tok, msgs, max_len=cfg.max_len)
        encoded.append((ids, [full, pub]))
    acts = collect_activations(model, tok, encoded, [cfg.layer],
                               batch_size=cfg.batch_size)[cfg.layer]
    X_old_full, X_old_pub = acts[:, 0], acts[:, 1]
    old = {("lr", "full"): lrp.predict_proba(X_old_full),
           ("lr", "public"): lrp.predict_proba(X_old_pub),
           ("dom", "full"): domp.predict_proba(X_old_full),
           ("dom", "public"): domp.predict_proba(X_old_pub)}

    # ── compare ──────────────────────────────────────────────────────────────
    act_diff = float(np.abs(X_new - acts).max())
    score_diff = max(float(np.abs(new[k] - old[k]).max()) for k in old)
    print(f"max |activation delta| old vs new: {act_diff:.3e}")
    print(f"max |score delta|      old vs new: {score_diff:.3e}")

    csv_path = OUT / "scores_sandbagging.csv"
    if csv_path.exists():                      # informational; absent until score_rollouts runs
        with open(csv_path) as f:
            rows = {int(r["idx"]): r for r in csv.DictReader(f)}
        csv_lr = np.array([float(rows[r["idx"]]["lr_full"]) for r in recs])
        print(f"vs committed csv (lr_full, batching differs): "
              f"max |delta| {np.abs(new[('lr', 'full')] - csv_lr).max():.3e}")

    ok = score_diff < 1e-6
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
