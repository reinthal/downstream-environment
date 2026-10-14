"""Follow-up-probe activations for one arm's game turns (HF env, bare
decoder), as experiments/2026-10-12-qwen38-layer-sweep/collect.py: the
residual stream on the follow-up "No" at each layer in cfg.layers, one
forward pass, decoder loaded through block cfg.layer (final norm removed),
dialogues rendered with thinking off and left-trimmed to cfg.max_len.

Output: activations_{arm}.npz — X (N, len(layers), d) float32 in the order of
common.scored_turns(arm), layers. Untracked.

Run:  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4,5 .venv/bin/python \
          experiments/2026-10-13-l48-vs-selfreport/collect.py --arm normal
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import ARMS, OUT, load_config, scored_turns

from linear_probes.activations import collect_activations, load_truncated_decoder  # noqa: E402
from linear_probes.dyl_probe import encode_no_span, with_followup  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    args = ap.parse_args()
    cfg = load_config()
    if cfg.layer != max(cfg.layers):
        raise SystemExit(f"cfg.layer {cfg.layer} must be the deepest read-out layer {max(cfg.layers)}")
    turns = scored_turns(args.arm)

    model, tok = load_truncated_decoder(cfg.model_id, cfg.layer, dtype=cfg.dtype,
                                        device=cfg.device, truncate=cfg.truncate_layers)
    encoded = [encode_no_span(tok, with_followup([*r["input_messages"],
                                                  {"role": "assistant", "content": r["announcement"]}]),
                              max_len=cfg.max_len, enable_thinking=False) for r in turns]
    print(f"{args.arm}: {len(turns)} turns, longest {max(len(e[0]) for e in encoded)} tokens, "
          f"{sum(len(e[0]) == cfg.max_len for e in encoded)} at the {cfg.max_len} cap", flush=True)
    t0 = time.time()
    acts = collect_activations(model, tok, encoded, cfg.layers, batch_size=cfg.batch_size)
    X = np.stack([acts[L] for L in cfg.layers], axis=1).astype(np.float32)
    path = OUT / f"activations_{args.arm}.npz"
    np.savez(path, X=X, layers=np.array(cfg.layers))
    print(f"{X.shape} in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
