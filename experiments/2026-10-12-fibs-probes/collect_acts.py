"""Stage ``acts``: residual-stream activations over the follow-up read window
for every announcement, both prefills, the six probe layers — saved per game
so any probe architecture or score rule is rescored without the model.
Stage ``dyl``: the in-house layer-18 DYL probe on the same announcements
(its own question and single "No" prefill, as in linear_probes.dyl_probe).

  acts_{tag}/g000.pt   {turn_key: {prefill: {"acts": (6, W, d) bf16,
                        "prompt_mask", "completion_mask": (W,) bool}}}
  acts_{tag}/index.json  per turn: key, n_tokens, dropped (over acts_max_len)
  dyl_{tag}.csv        game_id, round, position, speaker, role, dyl_p, dyl_logodds

Run (two GPUs hold the 59 loaded blocks of the 27B):
  CUDA_VISIBLE_DEVICES=1,2 uv run --no-sync python experiments/2026-10-12-fibs-probes/collect_acts.py --arm 3p --stage acts
  CUDA_VISIBLE_DEVICES=1   uv run --no-sync python experiments/2026-10-12-fibs-probes/collect_acts.py --arm 3p --stage dyl
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict

import torch

from acts import collect_windows, load_model, render_followup, window_masks
from common import OUT, REPO, load_config, model_path, read_jsonl, tag


def turn_key(r: dict) -> str:
    return f"r{r['round']}p{r['position']}"


def step(r: dict) -> tuple[list[dict], list[dict] | None]:
    """(messages, tools) of the announcement step as the model saw it.
    single_call: system + prompt, then the assistant's tool call (its
    thinking in reasoning_content, which the template drops once the
    follow-up is appended — the paper's read) and the tool result.
    two_call: the messages of the announcement call plus the announcement
    as a plain reply (no tool rendering: 2026-10-09 format)."""
    if r.get("tool_call"):
        tc = r["tool_call"]
        msgs = [*r["input_messages"],
                {"role": "assistant", "content": r["acting_text"], "reasoning_content": r["reasoning"],
                 "tool_calls": [{"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}]},
                {"role": "tool", "content": r["tool_result"]}]
        return msgs, r["tools"]
    return [*r["input_messages"], {"role": "assistant", "content": r["announcement"]}], None


def dialogue(r: dict) -> list[dict]:
    """The DYL baseline's input: the messages the player saw plus the
    announcement as a plain reply (the format that probe was trained on)."""
    return [*r["input_messages"], {"role": "assistant", "content": r["announcement"]}]


def stage_acts(cfg, t: str):
    recs = read_jsonl(OUT / f"transcripts_{t}.jsonl")
    if any(not r["announcement"] for r in recs):
        raise ValueError("silent turns in the transcripts; extract.py should have excluded the game")
    out_dir = OUT / f"acts_{t}"
    out_dir.mkdir(exist_ok=True)
    model, tok = load_model(model_path(cfg), max(cfg.probe_layers), dtype=cfg.dtype, device=cfg.device)

    index, todo, keys = [], [], []
    for r in recs:
        entry = {"game_id": r["game_id"], "key": turn_key(r), "n_tokens": {}, "dropped": False}
        msgs, tools = step(r)
        for prefill in cfg.prefills:
            rd = render_followup(tok, msgs, cfg.followup_question, prefill, tools=tools)
            entry["n_tokens"][prefill] = rd.n_tokens
            if rd.n_tokens > cfg.acts_max_len:
                entry["dropped"] = True
        if not entry["dropped"]:
            for prefill in cfg.prefills:
                todo.append(render_followup(tok, msgs, cfg.followup_question, prefill, tools=tools))
                keys.append((r["game_id"], entry["key"], prefill))
        index.append(entry)
    n_drop = sum(e["dropped"] for e in index)
    print(f"{len(recs)} announcements, {n_drop} dropped over {cfg.acts_max_len} tokens, "
          f"longest {max(max(e['n_tokens'].values()) for e in index)}; {len(todo)} forwards", flush=True)

    acts = collect_windows(model, tok, todo, cfg.probe_layers, cfg.acts_batch_tokens, cfg.acts_max_batch)
    per_game: dict[str, dict] = defaultdict(dict)
    for (gid, key, prefill), rd, a in zip(keys, todo, acts):
        pm, cm = window_masks(rd)
        per_game[gid].setdefault(key, {})[prefill] = {"acts": a, "prompt_mask": pm, "completion_mask": cm}
    for gid, d in per_game.items():
        torch.save(d, out_dir / f"{gid}.pt")
    (out_dir / "index.json").write_text(json.dumps(
        {"layers": cfg.probe_layers, "question": cfg.followup_question, "prefills": cfg.prefills,
         "max_len": cfg.acts_max_len, "turns": index}, indent=1) + "\n")
    print(f"wrote {len(per_game)} games to {out_dir}")


def stage_dyl(cfg, t: str):
    from linear_probes.dyl_probe import DYLProbe
    recs = read_jsonl(OUT / f"transcripts_{t}.jsonl")
    probe = DYLProbe(model_path(cfg), cfg.dyl_layer, str(REPO / cfg.dyl_probe),
                     batch_size=cfg.batch_size, max_len=cfg.dyl_max_len, dtype=cfg.dtype, device=cfg.device)
    p = probe.score([dialogue(r) for r in recs])
    with open(OUT / f"dyl_{t}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game_id", "round", "position", "speaker", "role", "dyl_p", "dyl_logodds"])
        for r, pi in zip(recs, p):
            w.writerow([r["game_id"], r["round"], r["position"], r["speaker"], r["role"],
                        f"{pi:.6f}", f"{math.log(pi / (1 - pi)):.4f}"])
    print(f"wrote dyl_{t}.csv ({len(recs)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--stage", choices=["acts", "dyl"], required=True)
    args = ap.parse_args()
    cfg = load_config()
    t = tag(args.arm, args.pilot)
    (stage_acts if args.stage == "acts" else stage_dyl)(cfg, t)


if __name__ == "__main__":
    main()
