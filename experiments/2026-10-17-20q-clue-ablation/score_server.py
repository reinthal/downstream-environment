"""EXPLORATORY: score every (question|clue, concept) pair of the ablation
against an OpenAI-compatible vLLM *server* on this box — for models that
need the serving env (Qwen3.8-27B: `serving/serve_qwen38.py`, .venv-serve,
vLLM 0.31), with the same prompts, pairs and readout as `score_vllm.py`:
chat template with thinking off (per request), one generated token,
p(yes) = mass(yes_tokens) / (mass(yes_tokens) + mass(no_tokens)) over its top
logprobs; a pair with neither is written with p_yes = null. Rows are the
`precomputed` backend's format; resumes (pairs already in --out are skipped).

    .venv/bin/python experiments/2026-10-17-20q-clue-ablation/score_server.py \
        --base-url http://localhost:8011/v1 --model Qwen/Qwen3.8-27B --out qwen38_scores.jsonl

Usually run inside one gpu-lock run that also starts the server (see
`run_qwen38_scoring.sh`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time

import httpx

from common import OUT
from score_vllm import all_pairs, entropy_cfg


async def score(base_url: str, model: str, out_path, concurrency: int, limit) -> None:
    ecfg = entropy_cfg()
    pairs, defs = all_pairs(ecfg)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            if r["model"] != model:
                raise SystemExit(f"{out_path} holds {r['model']} scores")
            done.add((r["kind"], r["text"], r["concept"]))
    todo = [p for p in pairs if p not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(pairs)} pairs, {len(done)} already scored, {len(todo)} to score", flush=True)
    prompts = {"question": ecfg["llm_question_prompt"], "clue": ecfg["llm_clue_prompt"]}
    yes, no = set(ecfg["yes_tokens"]), set(ecfg["no_tokens"])
    sem = asyncio.Semaphore(concurrency)
    t0, n = time.time(), 0
    retries = {"count": 0}

    async def post(client, body):
        """POST with a bounded retry on connection-level failures only (a
        keep-alive connection closed by the server as it is reused, 502-504);
        every retry is counted and printed, the 5th failure raises."""
        for attempt in range(5):
            try:
                resp = await client.post("/chat/completions", json=body)
            except httpx.TransportError as e:
                err = f"{type(e).__name__}"
            else:
                if resp.status_code not in (502, 503, 504):
                    return resp
                err = f"HTTP {resp.status_code}"
            retries["count"] += 1
            print(f"  retry {attempt + 1}/4 after {err} (total retries {retries['count']})", flush=True)
            if attempt == 4:
                raise RuntimeError(f"request failed 5 times, last: {err}")
            await asyncio.sleep(2 ** attempt)

    async with httpx.AsyncClient(base_url=base_url, timeout=600,
                                 limits=httpx.Limits(max_connections=concurrency)) as client:
        r = await client.get("/models")
        r.raise_for_status()
        served = [m["id"] for m in r.json()["data"]]
        if model not in served:
            raise SystemExit(f"server serves {served}, not {model}")
        with open(out_path, "a") as f:
            async def one(k, t, w):
                nonlocal n
                body = {"model": model, "max_tokens": 1, "temperature": 0, "logprobs": True,
                        "top_logprobs": ecfg["top_logprobs"],
                        "chat_template_kwargs": {"enable_thinking": False},
                        "messages": [{"role": "user", "content": prompts[k].format(
                            concept=w, definition=defs[w], text=t)}]}
                async with sem:
                    resp = await post(client, body)
                if resp.status_code != 200:
                    raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
                top = resp.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                py = sum(math.exp(x["logprob"]) for x in top if x["token"] in yes)
                pn = sum(math.exp(x["logprob"]) for x in top if x["token"] in no)
                f.write(json.dumps({"model": model, "provider": f"vllm-server {base_url}", "kind": k,
                                    "prompt": prompts[k], "text": t, "concept": w,
                                    "p_yes": py / (py + pn) if py + pn > 0 else None,
                                    "mass": py + pn}) + "\n")
                n += 1
                if n % 20000 == 0:
                    f.flush()
                    print(f"  {n}/{len(todo)} pairs, {n / (time.time() - t0):.0f}/s", flush=True)

            for i in range(0, len(todo), 20000):        # bounded task fan-out
                await asyncio.gather(*(one(*p) for p in todo[i:i + 20000]))
    print(f"done: {n} pairs in {time.time() - t0:.0f}s, {retries['count']} connection retries", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="file name inside the experiment folder")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    asyncio.run(score(a.base_url, a.model, OUT / a.out, a.concurrency, a.limit))


if __name__ == "__main__":
    main()
