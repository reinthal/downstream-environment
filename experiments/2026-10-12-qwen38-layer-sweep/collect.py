"""Follow-up-probe activations at every sweep layer (HF env, bare decoder).

Each dialogue = the messages the speaker saw + its public reply + "Was your
last response deceptive? (Yes/No)" + "No" (linear_probes.dyl_probe); the
vector is the residual stream on the "No" token(s), read at each layer in
cfg.layers from ONE forward pass (decoder loaded through block cfg.layer =
the deepest sweep layer; its final norm is the identity, so every layer is
the raw block output). Dialogues longer than cfg.max_len are left-trimmed.

  --set roleplaying   judge_roleplaying.jsonl (training data)
  --set games         transcripts_base.jsonl, turns with an announcement
  --set instructed    transcripts_instructed.jsonl, turns with exactly one announcement

Output: activations_{set}.npz — X (N, len(layers), d) float32 in input-file
order (after the filters above), layers, rows (input-file row indices). Untracked.

Run:  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4,5 .venv/bin/python \
          experiments/2026-10-12-qwen38-layer-sweep/collect.py --set roleplaying
Pilot: --pilot (reads the *_pilot inputs, writes activations_{set}_pilot.npz)
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import OUT, load_config, read_jsonl

from linear_probes.activations import collect_activations, load_truncated_decoder  # noqa: E402
from linear_probes.dyl_probe import encode_no_span, with_followup  # noqa: E402

def _game_turn(r: dict) -> list[dict]:
    return [*r["input_messages"], {"role": "assistant", "content": r["announcement"]}]


# set -> (input file, pilot input file, row filter, dialogue ending with the public reply)
SOURCES = {
    "roleplaying": ("judge_roleplaying", "judge_roleplaying_pilot", lambda r: True,
                    lambda r: [*r["messages"], {"role": "assistant", "content": r["reply"]}]),
    "games": ("transcripts_base", "transcripts_pilot", lambda r: bool(r["announcement"]), _game_turn),
    "instructed": ("transcripts_instructed", "transcripts_instructed_pilot",
                   lambda r: r["n_announcements"] == 1, _game_turn),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(SOURCES), required=True)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if cfg.layer != max(cfg.layers):
        raise SystemExit(f"cfg.layer {cfg.layer} must be the deepest sweep layer {max(cfg.layers)}")
    stem, pilot_stem, keep, dialogue = SOURCES[args.set]
    suffix = "_pilot" if args.pilot else ""
    recs = read_jsonl(OUT / f"{pilot_stem if args.pilot else stem}.jsonl")
    rows = [i for i, r in enumerate(recs) if keep(r)]

    model, tok = load_truncated_decoder(cfg.model_id, cfg.layer, dtype=cfg.dtype,
                                        device=cfg.device, truncate=cfg.truncate_layers)
    # every dialogue here was generated with thinking off; Qwen3.8's template
    # adds a "Reasoning effort" system preamble unless told so
    encoded = [encode_no_span(tok, with_followup(dialogue(recs[i])), max_len=cfg.max_len,
                              enable_thinking=False) for i in rows]
    n_trim = sum(len(ids) == cfg.max_len for ids, _ in encoded)
    print(f"{args.set}: {len(rows)}/{len(recs)} dialogues, longest {max(len(e[0]) for e in encoded)} "
          f"tokens, {n_trim} at the {cfg.max_len} cap", flush=True)
    t0 = time.time()
    acts = collect_activations(model, tok, encoded, cfg.layers, batch_size=cfg.batch_size)
    X = np.stack([acts[L] for L in cfg.layers], axis=1).astype(np.float32)
    path = OUT / f"activations_{args.set}{suffix}.npz"
    np.savez(path, X=X, layers=np.array(cfg.layers), rows=np.array(rows))
    print(f"{X.shape} in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
