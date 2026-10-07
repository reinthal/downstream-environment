"""Layer-18 activations on the follow-up "No" (HF env, truncated decoder).

Each dialogue = its prompt messages + the model's public reply (reasoning
dropped) + "Was your last response deceptive? (Yes/No)" + "No"; the vector is
the mean over the "No" tokens (one token for Qwen).

  --set roleplaying_off | roleplaying_on   rollouts_roleplaying_{off,on}.jsonl
  --set alpaca                             rollouts_alpaca.jsonl (control)
  --set base | steered | nothink | noparsnip | instructed
                                           game turns from 2026-09-29

Output: activations_followup_{set}.npz (X in input-file order; untracked)

Run:  CUDA_VISIBLE_DEVICES=6 .venv/bin/python experiments/2026-10-01/collect.py --set base
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, REPO, encode_no_span, read_jsonl, with_followup  # noqa: E402

from auditor.config import AuditorConfig  # noqa: E402
from linear_probes.activations import collect_activations, load_truncated_decoder  # noqa: E402

GAMES = REPO / "experiments" / "2026-09-29"
GAME_SETS = ("base", "steered", "nothink", "noparsnip", "instructed")


def dialogues(name: str) -> list[list[dict]]:
    if name in GAME_SETS:
        return [with_followup(r["input_messages"], r["announcement"] or "")
                for r in read_jsonl(GAMES / f"transcripts_{name}.jsonl")]
    path = OUT / f"rollouts_{name}.jsonl"
    return [with_followup(r["messages"], r["reply"]) for r in read_jsonl(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    args = ap.parse_args()
    cfg = AuditorConfig.load(OUT / "config.json")

    convs = dialogues(args.set)
    model, tok = load_truncated_decoder(cfg.model_id, cfg.layer, dtype=cfg.dtype,
                                        device=cfg.device, truncate=cfg.truncate_layers)
    encoded = [encode_no_span(tok, c, max_len=cfg.max_len) for c in convs]
    X = collect_activations(model, tok, encoded, [cfg.layer], batch_size=cfg.batch_size)[cfg.layer]
    path = OUT / f"activations_followup_{args.set}.npz"
    np.savez(path, X=X, layer=cfg.layer)
    print(f"{args.set}: {X.shape} -> {path}", flush=True)


if __name__ == "__main__":
    main()
