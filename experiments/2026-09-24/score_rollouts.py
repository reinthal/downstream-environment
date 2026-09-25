"""Score the Qwen on-policy rollouts with the 2026-09-22 instructed-pairs probes.

Reuses experiments/2026-09-22/common.py (encode_with_spans, collect_activations
— bare decoder, see AGENTS.md) and the probes trained there (layer 16). Two
poolings per rollout, as in 2026-09-22: the full assistant turn incl. the
private think block (`*_full`) and the post-</think> public text (`*_pub`).

Run:  CUDA_VISIBLE_DEVICES=5,6,7 uv run --no-sync python \
          experiments/2026-09-24/score_rollouts.py --set both
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

from common import MODEL_ID, collect_activations, encode_with_spans, read_jsonl  # noqa: E402

from linear_probes.probes import LinearProbe  # noqa: E402

PROBE_DIR = PREV / "probes"


def load_model_truncated(layer: int):
    """Like common.load_model, but only loads decoder blocks 0..layer (the
    LocalBackend truncation trick) — ~15 GB instead of ~54 GB for layer 16."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    mcfg = AutoConfig.from_pretrained(MODEL_ID)
    mcfg.num_hidden_layers = layer + 1   # blocks above the probe layer are never read
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=mcfg,
                                                 dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["sandbagging", "alpaca", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=6)
    args = ap.parse_args()
    sets = ["sandbagging", "alpaca"] if args.set == "both" else [args.set]

    lr = LinearProbe.load(str(PROBE_DIR / "probe_logistic_regression.npz"))
    dom = LinearProbe.load(str(PROBE_DIR / "probe_diff_of_means.npz"))
    layer = lr.config.layer
    model, tok = load_model_truncated(layer)

    for name in sets:
        recs = read_jsonl(OUT / f"rollouts_{name}.jsonl")
        print(f"[{name}] {len(recs)} rollouts, probe layer {layer}", flush=True)
        encoded = []
        for r in recs:
            msgs = list(r["input_messages"]) + [{
                "role": "assistant",
                "content": r["public"],
                "reasoning_content": r["reasoning"],
            }]
            ids, full, pub = encode_with_spans(tok, msgs, max_len=6144)
            encoded.append((ids, [full, pub]))
        acts = collect_activations(model, tok, encoded, [layer],
                                   batch_size=args.batch_size)[layer]
        X_full, X_pub = acts[:, 0], acts[:, 1]
        np.savez(OUT / f"activations_{name}.npz", X_full=X_full, X_pub=X_pub, layer=layer)

        out_path = OUT / f"scores_{name}.csv"
        lf, lp = lr.predict_proba(X_full), lr.predict_proba(X_pub)
        df, dp = dom.predict_proba(X_full), dom.predict_proba(X_pub)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "grade", "grade_raw", "expected_label",
                        "lr_full", "lr_pub", "dom_full", "dom_pub"])
            for i, r in enumerate(recs):
                w.writerow([r["idx"], r["grade"], r["grade_raw"],
                            r["expected_label"],
                            f"{lf[i]:.6f}", f"{lp[i]:.6f}",
                            f"{df[i]:.6f}", f"{dp[i]:.6f}"])
        print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
