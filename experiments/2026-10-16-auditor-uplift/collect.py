"""Layer-32 / layer-48 follow-up activations for existing 2026-10-15 messages
(GPU, .venv), sharded.

Rows:
  calib    every player message (announcements and answers) in the first
           cfg.calib_games games of each cfg.calib_sets (text-only games):
           calibration of the detectors, never evaluated
  passive  every S0 announcement of the 2026-10-15 base set (the shared
           discussion the S0 auditor read): the passive replay's games

Each message is read as the live scorer reads it: the speaker's dialogue
ending with its reply + the follow-up question and "No", thinking off,
left-trimmed to cfg.max_len; activations on the "No".

Output: activations_shard{k}.npz (l32, l48 probe log-odds; act48 float16) and
rows_shard{k}.jsonl.

Run:  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python \
          experiments/2026-10-16-auditor-uplift/collect.py --shard 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import OUT, REPO, load_config, read_jsonl

from linear_probes.activations import collect_activations, load_truncated_decoder  # noqa: E402
from linear_probes.dyl_probe import encode_no_span, with_followup  # noqa: E402
from linear_probes.probes import LinearProbe  # noqa: E402


def rows(cfg) -> list[dict]:
    src = REPO / cfg.source_dir
    out = []
    for s in cfg.calib_sets:
        keep = {f"g{k:03d}" for k in range(cfg.calib_games)}
        out += [{"part": "calib", "set": s, **t} for t in read_jsonl(src / f"transcripts_{s}.jsonl")
                if t["game_id"] in keep and t["text"]]
    out += [{"part": "passive", "set": "base", **t} for t in read_jsonl(src / "transcripts_base.jsonl")
            if t["scenario"] == "S0" and t["detector"] == "probe" and t["kind"] == "announcement" and t["text"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config()
    mine = rows(cfg)[args.shard::cfg.collect_shards]
    probes = {L: LinearProbe.load(str(REPO / cfg.probe_paths[f"l{L}"])) for L in cfg.layers}
    for L, p in probes.items():
        if p.config.layer != L:
            raise SystemExit(f"probe l{L} is layer {p.config.layer}")
    model, tok = load_truncated_decoder(cfg.model_id, max(cfg.layers), dtype=cfg.dtype,
                                        device=cfg.device, truncate=cfg.truncate_layers)
    enc = [encode_no_span(tok, with_followup([*r["input_messages"], {"role": "assistant", "content": r["text"]}]),
                          max_len=cfg.max_len, enable_thinking=False) for r in mine]
    order = np.argsort([len(e[0]) for e in enc])           # similar lengths per batch
    X = collect_activations(model, tok, [enc[i] for i in order], cfg.layers, batch_size=cfg.batch_size)
    inv = np.argsort(order)
    X = {L: X[L][inv] for L in cfg.layers}
    sc = {L: ((X[L] - probes[L].mu) / probes[L].sd) @ probes[L].w + probes[L].b for L in cfg.layers}
    np.savez(OUT / f"activations_shard{args.shard}.npz", l32=sc[32], l48=sc[48], act48=X[48].astype(np.float16),
             n_tokens=np.array([len(e[0]) for e in enc]))
    keys = ("part", "set", "game_id", "scenario", "detector", "kind", "round", "position", "speaker", "role", "text")
    with open(OUT / f"rows_shard{args.shard}.jsonl", "w") as f:
        for r in mine:
            f.write(json.dumps({k: r.get(k) for k in keys}) + "\n")
    print(f"shard {args.shard}: {len(mine)} messages", flush=True)


if __name__ == "__main__":
    main()
