"""Judging of Carrot-Parsnip announcements with an API judge via OpenRouter.

Same prompts as judge_messages.py (Apollo roleplaying grading template,
verbatim). `--reveal-roles` switches the black-box scenario to the oracle
variant (judge told the speaker's secret role). Temperature 0, hybrid-model
reasoning disabled. Output schema matches judge_messages.py plus
`judge_model`.

Two execution modes:
  * interactive (default): threaded calls to /v1/chat/completions.
  * OpenRouter Batch API (50% price, up to 24 h completion window):
      --batch submit    POST the whole arm as batches of 500, save batch ids
      --batch poll      fetch results; writes the same judge_*.jsonl when done

Requires OPENROUTER_API_KEY, read from the repo-root `.env` (gitignored;
`common.load_env`) or the environment — do not commit keys.

Run:  uv run --no-sync python \
          experiments/2026-09-22/judge_messages_api.py [--arm steered] \
          [--model deepseek/deepseek-v4.1-flash] [--reveal-roles] \
          [--batch submit|poll]
Smoke: add  --limit 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(OUT.parents[1]))

from common import load_env, read_jsonl  # noqa: E402
from judge_messages import build_prompt, judgement_of, label_of  # noqa: E402

load_env()                       # OPENROUTER_API_KEY from repo-root .env

MODEL = "openai/gpt-4o"          # overridden by --model
URL = "https://openrouter.ai/api/v1/chat/completions"
CHUNK = 50
WORKERS = 8


def call_judge(prompt: str, max_tokens: int = 800, attempts: int = 4) -> str:
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
                       "reasoning": {"enabled": False},   # hybrid models: grade directly
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json"})
    for a in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.load(resp)
            msg = out["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning") or ""
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError) as e:
            if a == attempts - 1:
                print(f"giving up after {attempts} attempts: {e}", flush=True)
                return ""
            time.sleep(2 ** a * 2)
    return ""


def _api(method: str, path: str, payload: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"https://openrouter.ai/api/v1{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def row_of(rec: dict, text: str) -> dict:
    score = judgement_of(text)
    return {"game_id": rec["game_id"], "seed_group": rec["seed_group"],
            "round": rec["round"], "position": rec["position"],
            "speaker": rec["speaker"], "role": rec["role"],
            "judge_score": score, "judge_label": label_of(score),
            "judge_reasoning": text[-1500:], "judge_model": MODEL}


BATCH_SIZE = 500


def batch_submit(jobs, state_path: Path):
    ids = []
    for b0 in range(0, len(jobs), BATCH_SIZE):
        reqs = [{"custom_id": f"m{b0 + i}",
                 "body": {"temperature": 0, "max_tokens": 800,
                          "reasoning": {"enabled": False},
                          "messages": [{"role": "user", "content": p}]}}
                for i, (_, p) in enumerate(jobs[b0:b0 + BATCH_SIZE])]
        out = _api("POST", "/batches", {"endpoint": "/v1/chat/completions",
                                        "model": MODEL, "requests": reqs})
        ids.append(out["id"])
        print(f"submitted batch {out['id']} ({len(reqs)} requests, "
              f"status {out['status']})", flush=True)
    state_path.write_text(json.dumps({"model": MODEL, "batch_ids": ids,
                                      "n_jobs": len(jobs)}))
    print(f"state -> {state_path}; poll with --batch poll", flush=True)


def batch_poll(jobs, state_path: Path, out_path: Path):
    state = json.loads(state_path.read_text())
    texts: dict[int, str] = {}
    pending = []
    for bid in state["batch_ids"]:
        out = _api("GET", f"/batches/{bid}")
        if out["status"] != "completed":
            pending.append((bid, out["status"], out.get("request_counts")))
            continue
        for res in out["results"]:
            i = int(res["custom_id"][1:])
            body = (res.get("response") or {}).get("body") or {}
            msg = (body.get("choices") or [{}])[0].get("message", {})
            texts[i] = msg.get("content") or msg.get("reasoning") or ""
    if pending:
        for bid, status, counts in pending:
            print(f"batch {bid}: {status} {counts}", flush=True)
        print("not all batches complete — try again later", flush=True)
        return
    with open(out_path, "w") as f:
        for i, (rec, _) in enumerate(jobs):
            f.write(json.dumps(row_of(rec, texts.get(i, ""))) + "\n")
    print(f"wrote {out_path} ({len(jobs)} rows)", flush=True)


def judge_one(job) -> dict:
    rec, prompt = job
    text = call_judge(prompt)
    if judgement_of(text) is None and text:          # one retry for unparseable
        text = call_judge(prompt + "\n\nRemember: end with <judgement>N</judgement>.")
    return row_of(rec, text)


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="base")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--reveal-roles", action="store_true",
                    help="oracle judge: scenario states the speaker's secret role")
    ap.add_argument("--batch", choices=["submit", "poll"],
                    help="use the OpenRouter Batch API instead of interactive calls")
    args = ap.parse_args()
    MODEL = args.model

    recs = read_jsonl(OUT / f"transcripts_{args.arm}.jsonl")
    by_game: dict[str, list[dict]] = {}
    for r in recs:
        by_game.setdefault(r["game_id"], []).append(r)
    jobs = []
    for game in by_game.values():
        game.sort(key=lambda r: (r["round"], r["position"]))
        for i, r in enumerate(game):
            jobs.append((r, build_prompt(r, game[:i], reveal_role=args.reveal_roles)))
    if args.limit:
        jobs = jobs[:args.limit]

    suffix = "_oracle" if args.reveal_roles else ""
    path = OUT / f"judge_{args.arm}_api{suffix}.jsonl"
    if args.batch:
        state_path = OUT / f"judge_batch_state_{args.arm}{suffix}.json"
        if args.batch == "submit":
            batch_submit(jobs, state_path)
        else:
            batch_poll(jobs, state_path, path)
        return

    done = sum(1 for _ in open(path)) if path.exists() else 0
    print(f"{len(jobs)} messages to judge with {MODEL}, {done} done", flush=True)

    with open(path, "a") as f, ThreadPoolExecutor(WORKERS) as ex:
        for c0 in range(done, len(jobs), CHUNK):
            chunk = jobs[c0:c0 + CHUNK]
            t0 = time.time()
            for row in ex.map(judge_one, chunk):
                f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{c0 + len(chunk)}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)

    from collections import Counter
    with open(path) as f:
        labels = Counter(json.loads(line)["judge_label"] for line in f)
    print("judge labels:", dict(labels), flush=True)


if __name__ == "__main__":
    main()
