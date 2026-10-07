"""On-policy Qwen3.5-27B replies for the follow-up probe (vLLM env).

  --task roleplaying --thinking off   training data, as the paper: system =
        scenario + postfix, user = question, the reply continues the answer
        prefix (e.g. "Student:"); 1 reply per scenario, temperature 1, <=3072 tokens
  --task roleplaying --thinking on    variant: Qwen thinks first; the answer
        prefix is placed after </think>; the reasoning is kept in the record
        but never shown to the probe
  --task alpaca                       control: the 9,994 alpaca prompts of the
        paper's control set, thinking off, same sampling

Output: rollouts_roleplaying_{off,on}.jsonl, rollouts_alpaca.jsonl

Run:  CUDA_VISIBLE_DEVICES=4,5 PATH=$PWD/.venv-vllm/bin:$PATH \\
          .venv-vllm/bin/python experiments/2026-10-01/generate.py --task roleplaying --thinking off
Smoke: add  --limit 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, VENDOR, roleplay_prompt, roleplay_scenarios, write_jsonl  # noqa: E402

from auditor.config import AuditorConfig  # noqa: E402
from rollouts.generation import split_think  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["roleplaying", "alpaca"], required=True)
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = AuditorConfig.load(OUT / "config.json")

    if args.task == "roleplaying":
        items = [{"idx": i, "scenario": s["scenario"], "question": s["question"],
                  "answer_prefix": s["answer_prefix"], "messages": roleplay_prompt(s)}
                 for i, s in enumerate(roleplay_scenarios())]
        name = f"rollouts_roleplaying_{args.thinking}"
    else:
        raw = json.load(open(VENDOR / "data" / "rollouts" / "alpaca__plain__llama-70b-3.3.json"))
        items = [{"idx": i, "messages": [{"role": m["role"], "content": m["content"]}
                                         for m in r["input_messages"] if m["role"] != "assistant"],
                  "answer_prefix": ""} for i, r in enumerate(raw["rollouts"])]
        name = "rollouts_alpaca"
    items = items[: args.limit or None]
    path = OUT / f"{name}{'_smoke' if args.limit else ''}.jsonl"
    print(f"{len(items)} prompts -> {path}", flush=True)

    from rollouts.generation import make_backend
    backend = make_backend("vllm", cfg.model_id, tensor_parallel_size=cfg.tensor_parallel_size,
                           max_model_len=cfg.max_model_len)
    backend.gen_kwargs = {"temperature": cfg.temperature, "top_p": cfg.top_p}
    backend.force_close_tokens = cfg.force_close_tokens
    t0 = time.time()

    if args.thinking == "on":
        backend.enable_thinking = True
        raws = backend.generate([it["messages"] for it in items], cfg.max_new_tokens, seed=cfg.seed)
        heads = [raw.split("</think>")[0] + "</think>\n\n" for raw in raws]
        reasoning = [split_think(raw)[0] for raw in raws]
        texts = [backend.render(it["messages"]) + h + it["answer_prefix"]
                 for it, h in zip(items, heads)]
    else:
        backend.enable_thinking = False
        reasoning = [""] * len(items)
        texts = [backend.render(it["messages"]) + it["answer_prefix"] for it in items]
    conts = backend.complete(texts, cfg.max_new_tokens, seed=cfg.seed + 1)
    print(f"generated in {time.time() - t0:.0f}s ({backend.n_force_closed} force-closed)", flush=True)

    records = []
    for it, r, c in zip(items, reasoning, conts):
        rec = {k: v for k, v in it.items()}
        rec.update(thinking=args.thinking, reasoning=r, completion=c,
                   reply=(it["answer_prefix"] + c).strip())
        records.append(rec)
    write_jsonl(path, records)
    for rec in records[:3]:
        print(json.dumps(rec["reply"])[:300], flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":      # vLLM workers re-import this module
    main()
