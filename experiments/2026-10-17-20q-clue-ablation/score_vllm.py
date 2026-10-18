"""EXPLORATORY: score every (question|clue, concept) pair of the ablation with
a local vLLM engine — the comparative analysis to Jev, run on the GPU box
(no OpenRouter). Same prompts (`llm_question_prompt` / `llm_clue_prompt` in
environments/twenty_questions/configs/entropy.yaml) and readout as the
entropy calculator's LLM backends: chat template with thinking off, one
generated token, p(yes) = mass(yes_tokens) / (mass(yes_tokens) + mass(no_tokens))
over its top logprobs; a pair with neither is written with p_yes = null (the
analysis then refuses to run). Output rows are the `precomputed` backend's
format, so compare_oracles.py reads them with no model calls.

Runs under .venv-vllm (twenty_questions is not importable there — it needs
inspect_ai; observations are rebuilt here exactly as
twenty_questions.entropy.game_observations does, checked by
`--check-pairs` from the env venv):

    PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python \
        experiments/2026-10-17-20q-clue-ablation/score_vllm.py --workers N

--workers N: one engine per GPU of CUDA_VISIBLE_DEVICES (held with the
gpu-lock skill), worker i scores todo[i::N] into qwen_vllm_scores.shard<i>.jsonl;
the parent waits, fails if any worker fails, and merges the shards into
qwen_vllm_scores.jsonl. Resumes: pairs already in the main file or any shard
are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import yaml

from common import ENV, OUT, REPO, AblationConfig, cell

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"   # Hub head on 2026-10-18 (not enforced)
SCORES = OUT / "qwen_vllm_scores.jsonl"
CHUNK = 50_000          # prompts per llm.generate call (progress + flush granularity)


def entropy_cfg() -> dict:
    return yaml.safe_load((ENV / "configs" / "entropy.yaml").read_text())


def observations(game: dict) -> list[tuple[str, str]]:
    """(kind, text) as twenty_questions.entropy.game_observations with include_clues."""
    out = []
    for e in game["exchanges"]:
        if e["kind"] == "question":
            out.append(("question", e["text"]))
        if e["kind"] in ("question", "clue_request") and e["clue"]:
            out.append(("clue", e["clue"]))
    return out


def all_pairs(ecfg: dict) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    cfg = AblationConfig.load()
    concepts = json.loads((ENV / ecfg["concept_set"]).read_text())["concepts"]
    defs = {c["word"]: c["definition"] for c in concepts}
    texts = sorted({o for m in cfg.models for a in cfg.arms
                    for g in json.loads((OUT / f"games_{cell(m, a, False)}.json").read_text())["games"]
                    for o in observations(g)})
    return [(k, t, w) for k, t in texts for w in defs], defs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-pairs", action="store_true", help="compare the pair set with twenty_questions (env venv)")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N missing pairs (smoke test)")
    ap.add_argument("--workers", type=int, default=None, help="one engine per visible GPU; parent mode")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    args = ap.parse_args()
    ecfg = entropy_cfg()
    pairs, defs = all_pairs(ecfg)
    print(f"{len(pairs)} pairs ({len({(k, t) for k, t, _ in pairs})} texts x {len(defs)} concepts)", flush=True)

    if args.check_pairs:
        from twenty_questions.entropy import EntropyConfig, game_observations
        c = AblationConfig.load()
        ec = EntropyConfig(**{**ecfg, "include_clues": True})
        theirs = {(k, t) for m in c.models for a in c.arms
                  for g in json.loads((OUT / f"games_{cell(m, a, False)}.json").read_text())["games"]
                  for k, _, t, _ in game_observations(ec, g)}
        ours = {(k, t) for k, t, _ in pairs}
        if ours != theirs:
            raise SystemExit(f"pair sets differ: {len(ours - theirs)} only here, {len(theirs - ours)} only in env")
        print("pair set matches twenty_questions.entropy.game_observations")
        return

    done = scored()
    todo = [p for p in pairs if p not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} already scored, {len(todo)} to score", flush=True)
    if args.workers:
        run_workers(args.workers, args.limit)
        return
    if not todo:
        return
    out = SCORES
    if args.shard is not None:
        todo = todo[args.shard::args.num_shards]
        out = shard_path(args.shard)
        print(f"shard {args.shard}/{args.num_shards} on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}: "
              f"{len(todo)} pairs", flush=True)

    sys.path.insert(0, str(REPO))
    from rollouts.generation import make_backend
    from vllm import SamplingParams

    # make_backend / VLLMBackend take no revision: the weights are the Hub's
    # current Qwen/Qwen3.5-9B (c202236 on 2026-10-18); the row's provider field
    # records it for reference
    backend = make_backend("vllm", MODEL, tensor_parallel_size=1, max_model_len=1024)
    backend.enable_thinking = False                # empty, closed think block: the answer is the first token
    prompts = {"question": ecfg["llm_question_prompt"], "clue": ecfg["llm_clue_prompt"]}
    yes, no = set(ecfg["yes_tokens"]), set(ecfg["no_tokens"])
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=ecfg["top_logprobs"])
    t0 = time.time()
    with open(out, "a") as f:
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            texts = [backend.render([{"role": "user", "content": prompts[k].format(concept=w, definition=defs[w], text=t)}])
                     for k, t, w in chunk]
            outs = backend.llm.generate(texts, sp, use_tqdm=False)
            for (k, t, w), o in zip(chunk, outs):
                top = o.outputs[0].logprobs[0]                       # {token_id: Logprob}
                py = sum(math.exp(lp.logprob) for lp in top.values() if lp.decoded_token in yes)
                pn = sum(math.exp(lp.logprob) for lp in top.values() if lp.decoded_token in no)
                f.write(json.dumps({"model": MODEL, "provider": f"vllm-local@{REVISION[:7]}", "kind": k,
                                    "prompt": prompts[k], "text": t, "concept": w,
                                    "p_yes": py / (py + pn) if py + pn > 0 else None, "mass": py + pn}) + "\n")
            f.flush()
            n = i + len(chunk)
            print(f"  {n}/{len(todo)} pairs, {n / (time.time() - t0):.0f}/s", flush=True)


def shard_path(i: int):
    return OUT / f"qwen_vllm_scores.shard{i}.jsonl"


def scored() -> set:
    done = set()
    for path in [SCORES, *sorted(OUT.glob("qwen_vllm_scores.shard*.jsonl"))]:
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                if r["model"] != MODEL:
                    raise SystemExit(f"{path} holds {r['model']} scores")
                done.add((r["kind"], r["text"], r["concept"]))
    return done


def run_workers(n: int, limit) -> None:
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    if len(visible) < n:
        raise SystemExit(f"--workers {n} but CUDA_VISIBLE_DEVICES={visible}")
    (OUT / "logs").mkdir(exist_ok=True)
    procs = []
    for i in range(n):
        cmd = [sys.executable, __file__, "--shard", str(i), "--num-shards", str(n)]
        if limit:
            cmd += ["--limit", str(limit)]
        env = os.environ | {"CUDA_VISIBLE_DEVICES": visible[i]}
        log = open(OUT / "logs" / f"score_vllm.shard{i}.log", "a")
        procs.append((i, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"worker {i}: GPU {visible[i]}, pid {procs[-1][1].pid}", flush=True)
    failed = []
    for i, p, log in procs:
        p.wait()
        log.close()
        print(f"worker {i} exit {p.returncode}", flush=True)
        if p.returncode != 0:
            failed.append(i)
    if failed:
        raise SystemExit(f"workers {failed} failed; see logs/score_vllm.shard<i>.log (shards kept, rerun resumes)")
    have = set()
    if SCORES.exists():
        have = {(r["kind"], r["text"], r["concept"]) for r in map(json.loads, SCORES.read_text().splitlines())}
    added = 0
    with open(SCORES, "a") as f:
        for i in range(n):
            for line in shard_path(i).read_text().splitlines() if shard_path(i).exists() else []:
                r = json.loads(line)
                k = (r["kind"], r["text"], r["concept"])
                if k not in have:
                    f.write(line + "\n")
                    have.add(k)
                    added += 1
    for i in range(n):
        shard_path(i).unlink(missing_ok=True)
    print(f"merged {added} rows into {SCORES.name}; {len(have)} pairs scored", flush=True)


if __name__ == "__main__":
    main()
