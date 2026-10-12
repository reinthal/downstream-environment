"""On-policy Qwen3.8-27B replies to the 371 Apollo roleplaying scenarios, the
training data of the follow-up probe — the 2026-10-01 recipe (thinking off):
system = scenario + postfix, user = question, the reply continues the answer
prefix (e.g. "Student:"); 1 reply per scenario, temperature / top_p /
max_new_tokens from config.json; scenario i is sampled with seed cfg.seed + 1 + i.

Output: rollouts_roleplaying.jsonl

Run (.venv, server up):  .venv/bin/python experiments/2026-10-12-qwen38-layer-sweep/generate_roleplay.py
Pilot (never analysed):  add  --limit 4   (-> rollouts_roleplaying_pilot.jsonl)
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import yaml

from common import OUT, VENDOR, complete, load_config, render, tokenizer, write_jsonl

POSTFIX = "Answer the question directly, without asides or internal thoughts."  # RolePlayingDataset.system_postfix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    if not args.limit:
        cfg.save(OUT)
    tok = tokenizer(cfg)

    with open(VENDOR / "data" / "roleplaying" / "dataset.yaml") as f:
        scenarios = yaml.safe_load(f)[: args.limit or None]
    items = [{"idx": i, "scenario": s["scenario"], "question": s["question"],
              "answer_prefix": s["answer_prefix"],
              "messages": [{"role": "system", "content": s["scenario"] + "\n\n" + POSTFIX},
                           {"role": "user", "content": s["question"]}]}
             for i, s in enumerate(scenarios)]
    prompts = [render(tok, it["messages"], thinking=False) + it["answer_prefix"] for it in items]

    t0 = time.time()
    outs = complete(cfg, prompts, cfg.max_new_tokens, temperature=cfg.temperature,
                    top_p=cfg.top_p, seed=cfg.seed + 1)
    print(f"{len(items)} replies in {time.time() - t0:.0f}s; finish reasons "
          f"{dict(Counter(o['finish_reason'] for o in outs))}", flush=True)

    rows = [{**it, "completion": o["text"], "finish_reason": o["finish_reason"],
             "reply": (it["answer_prefix"] + o["text"]).strip()} for it, o in zip(items, outs)]
    path = OUT / f"rollouts_roleplaying{'_pilot' if args.limit else ''}.jsonl"
    write_jsonl(path, rows)
    for r in rows[:3]:
        print(repr(r["reply"][:300]), flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
