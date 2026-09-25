"""Grade-annotated auditor via OpenRouter: repeated seeds per game.

Same prompts as the local run (built by the `auditor` package from this
folder's suspicion_scale.json), but the auditor model runs on OpenRouter,
enabling many independent audits per game: n_games games (balanced across
seed groups: game_idx < n_games/5) x len(conditions) x n_auditor_seeds.
Thinking stays enabled (as locally); the guess is parsed from the reconstructed
<think>...</think>public turn with one different-seed retry, as locally.

Config: config_api.json (ApiAuditorConfig = AuditorConfig + api fields).
Requires OPENROUTER_API_KEY in the repo-root .env (gitignored) or env.
Appends to auditor_api_{arm}.jsonl in job order — safe to re-run to resume.

Run:  uv run --no-sync python experiments/2026-09-25/run_auditor_api.py \
          --arm base [--limit 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
DATA22 = OUT.parents[0] / "2026-09-22"
sys.path.insert(0, str(DATA22))          # common (load_env)
sys.path.insert(0, str(REPO))

from common import load_env  # noqa: E402

from auditor.config import ApiAuditorConfig  # noqa: E402
from auditor.prompts import (AUDITOR_SYSTEM, AUDITOR_USER, DETECTOR_NOTE,  # noqa: E402
                             build_transcript, parse_guess)
from auditor.run import build_grade_maps, read_jsonl  # noqa: E402
from auditor.scale import load_scale  # noqa: E402

load_env()
URL = "https://openrouter.ai/api/v1/chat/completions"
CHUNK = 100


def call_api(conv: list[dict], cfg: ApiAuditorConfig, seed: int,
             attempts: int = 5) -> tuple[str, str]:
    """-> (reasoning, content); empty strings after exhausted retries."""
    body = json.dumps({
        "model": cfg.api_model, "temperature": cfg.temperature,
        "top_p": cfg.top_p, "max_tokens": cfg.api_max_tokens, "seed": seed,
        "messages": conv}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json"})
    for a in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.load(resp)
            msg = out["choices"][0]["message"]
            return msg.get("reasoning") or "", msg.get("content") or ""
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                TimeoutError, json.JSONDecodeError) as e:
            if a == attempts - 1:
                print(f"giving up after {attempts} attempts: {e}", flush=True)
                return "", ""
            time.sleep(2 ** a * 2)
    return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["base", "steered"], default="base")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N jobs")
    args = ap.parse_args()

    cfg = ApiAuditorConfig.load(OUT / "config_api.json")
    data = cfg.path(cfg.data_dir)
    scale = load_scale(OUT / "suspicion_scale.json")

    recs = read_jsonl(data / f"transcripts_{args.arm}.jsonl")
    games_meta = json.load(open(data / f"games_{args.arm}.json"))["games"]
    parsnip = {g["game_id"]: g["parsnip"] for g in games_meta}
    player_names = list(games_meta[0]["roles"])
    num_rounds = max(r["round"] for r in recs) + 1
    by_game = defaultdict(list)
    for r in recs:
        by_game[r["game_id"]].append(r)
    for msgs in by_game.values():
        msgs.sort(key=lambda m: (m["round"], m["position"]))

    per_group = cfg.n_games // cfg.seed_groups
    game_ids = sorted(g for g in by_game
                      if int(g.split("g")[1]) < per_group)
    assert len(game_ids) == cfg.n_games, len(game_ids)

    with open(data / f"scores_{args.arm}.csv") as f:
        score_rows = list(csv.DictReader(f))
    grade_by_cond = build_grade_maps(score_rows, scale, cfg.conditions)

    def build_conv(gid: str, cond: str) -> list[dict]:
        transcript = build_transcript(by_game[gid], grade_by_cond.get(cond))
        system = AUDITOR_SYSTEM.format(
            players=", ".join(player_names), num_rounds=num_rounds,
            detector_note=DETECTOR_NOTE.format(legend=cfg.grade_legend)
            if cond != "none" else "")
        user = AUDITOR_USER.format(
            transcript=transcript,
            and_scores=" and the suspicion grades" if cond != "none" else "",
            example=player_names[0])
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    jobs = [(gid, cond, si) for gid in game_ids for cond in cfg.conditions
            for si in range(cfg.n_auditor_seeds)]
    if args.limit:
        jobs = jobs[: args.limit]
    else:                            # re-stamp the api config (NOT config.json)
        (OUT / "config_api.json").write_text(cfg.to_json() + "\n")

    def run_job(job) -> dict:
        gid, cond, si = job
        conv = build_conv(gid, cond)
        seed = cfg.seed + 100 * si
        reasoning, content = call_api(conv, cfg, seed)
        raw = f"<think>{reasoning}</think>{content}" if reasoning else content
        guess = parse_guess(raw, player_names)
        if guess is None and (reasoning or content):   # one different-seed retry
            reasoning, content = call_api(conv, cfg, seed + 20_000)
            raw = f"<think>{reasoning}</think>{content}" if reasoning else content
            guess = parse_guess(raw, player_names)
        return {"arm": args.arm, "game_id": gid, "condition": cond,
                "auditor_seed": si, "guess": guess, "parsnip": parsnip[gid],
                "correct": bool(guess == parsnip[gid]),
                "parse_failed": guess is None, "judge_model": cfg.api_model,
                "reasoning": reasoning[-2000:], "public": content}

    path = OUT / f"auditor_api_{args.arm}{'_smoke' if args.limit else ''}.jsonl"
    done = sum(1 for _ in open(path)) if path.exists() and not args.limit else 0
    print(f"{len(jobs)} jobs ({len(game_ids)} games x {cfg.conditions} x "
          f"{cfg.n_auditor_seeds} seeds), {done} done -> {path}", flush=True)

    mode = "a" if done else "w"
    with open(path, mode) as f, ThreadPoolExecutor(cfg.workers) as ex:
        for c0 in range(done, len(jobs), CHUNK):
            chunk = jobs[c0:c0 + CHUNK]
            t0 = time.time()
            n_ok = 0
            for row in ex.map(run_job, chunk):
                f.write(json.dumps(row) + "\n")
                n_ok += row["correct"]
            f.flush()
            print(f"{c0 + len(chunk)}/{len(jobs)} ({time.time() - t0:.0f}s, "
                  f"{n_ok}/{len(chunk)} correct in chunk)", flush=True)

    from collections import Counter
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    acc = Counter()
    n = Counter()
    for r in rows:
        acc[r["condition"]] += r["correct"]
        n[r["condition"]] += 1
    for cond in cfg.conditions:
        if n[cond]:
            print(f"{args.arm}/{cond}: {acc[cond]}/{n[cond]} correct "
                  f"({acc[cond] / n[cond]:.3f})", flush=True)


if __name__ == "__main__":
    main()
