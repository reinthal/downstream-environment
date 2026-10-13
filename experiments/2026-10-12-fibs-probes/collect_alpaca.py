"""Alpaca control for the paper's probes: the benign Qwen3.5-27B replies of
data/qwen3.5-27b_alpaca_rollouts.jsonl (2026-09-24, thinking on; the public
reply is the assistant turn, the reasoning is dropped by the template)
rendered with the same follow-up and prefills as the games, activations
saved to acts_alpaca/ and scored by every architecture.

  acts_alpaca/part000.pt  list of {prefill: {"acts", "prompt_mask", "completion_mask"}}
  scores_alpaca.csv       idx, {arch}_yes, {arch}_no, {arch}_contrast

Run:  CUDA_VISIBLE_DEVICES=1,2 uv run --no-sync python experiments/2026-10-12-fibs-probes/collect_alpaca.py
"""
from __future__ import annotations

import csv

import torch

from acts import collect_windows, load_model, render_followup, window_masks
from common import OUT, REPO, load_config, model_path, probe_dir, read_jsonl

from probe_inference import load_probe  # noqa: E402
from probe_inference.score import build_read_mask  # noqa: E402

PART = 200


def main():
    cfg = load_config()
    rows = read_jsonl(REPO / cfg.alpaca_path)
    model, tok = load_model(model_path(cfg), max(cfg.probe_layers), dtype=cfg.dtype, device=cfg.device)
    (other,) = [p for p in cfg.prefills if p != cfg.lie_prefill]
    probes = {arch: load_probe(probe_dir(cfg, arch)) for arch in cfg.probe_archs}
    out_dir = OUT / "acts_alpaca"
    out_dir.mkdir(exist_ok=True)

    rendered, keys = [], []
    for r in rows:
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["input_messages"]]
        msgs.append({"role": "assistant", "content": r["public"]})
        for prefill in cfg.prefills:
            rendered.append(render_followup(tok, msgs, cfg.followup_question, prefill))
            keys.append((r["idx"], prefill))
    if max(x.n_tokens for x in rendered) > cfg.acts_max_len:
        raise ValueError("an alpaca reply exceeds acts_max_len")
    print(f"{len(rows)} replies, {len(rendered)} forwards, longest {max(x.n_tokens for x in rendered)}", flush=True)

    acts = collect_windows(model, tok, rendered, cfg.probe_layers, cfg.acts_batch_tokens, cfg.acts_max_batch)
    per_idx: dict = {}
    for (idx, prefill), rd, a in zip(keys, rendered, acts):
        pm, cm = window_masks(rd)
        per_idx.setdefault(idx, {})[prefill] = {"acts": a, "prompt_mask": pm, "completion_mask": cm}
    idxs = sorted(per_idx)
    for p0 in range(0, len(idxs), PART):
        torch.save({i: per_idx[i] for i in idxs[p0:p0 + PART]}, out_dir / f"part{p0 // PART:03d}.pt")

    with open(OUT / "scores_alpaca.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx"] + [f"{a}_{k}" for a in cfg.probe_archs for k in ("yes", "no", "contrast")])
        for idx in idxs:
            out = [idx]
            for arch, p in probes.items():
                s = {}
                for prefill in cfg.prefills:
                    e = per_idx[idx][prefill]
                    mask = build_read_mask(p.read_window, e["prompt_mask"], e["completion_mask"], torch.tensor(0))
                    s[prefill] = p.score({L: e["acts"][i] for i, L in enumerate(p.layers)}, mask)
                out += [f"{s['Yes.']:.6f}", f"{s['No.']:.6f}", f"{s[cfg.lie_prefill] - s[other]:.6f}"]
            w.writerow(out)
    print(f"wrote scores_alpaca.csv ({len(idxs)} replies) and {out_dir}")


if __name__ == "__main__":
    main()
