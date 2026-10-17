"""Play every model x arm cell: n_games games each on seeds base_seed..
(same secrets for every model and arm), both seats and the judge = the
model. Errored seeds are re-run once (preregistered); the OpenRouter key's
spend is checked after every cell against the cost cap.

    uv run --project environments/twenty_questions python experiments/2026-10-17-20q-clue-ablation/run.py [--pilot]

Writes per cell: game_{cell}.yaml, games_{cell}.json, transcripts_{cell}.jsonl,
inspect_logs_{cell}/ (twenty_questions.run), plus games_{cell}_retry_s{seed}.json
for re-runs, and run_log.json (spend, cells, retries).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import threading
import time

import httpx
import yaml

from common import ENV, OUT, REPO, AblationConfig, cell

LOCK = threading.Lock()


def openrouter_key() -> str:
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("OPENROUTER_API_KEY not in .env")


def spend(key: str) -> float:
    r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    return float(r.json()["data"]["usage"])


def game_yaml(cfg: AblationConfig, model: str, arm: str, name: str) -> str:
    base = yaml.safe_load((REPO / cfg.base_game_config).read_text())
    base |= {"max_questions": cfg.max_questions, "concept_set": cfg.concept_set,
             "default_model": model, "max_concurrent_games": cfg.max_concurrent_games}
    base |= cfg.arms[arm]
    base["suspects"] = [{**base["suspects"][0], "model": None, "objective": cfg.objective,
                         "reasoning_effort": cfg.reasoning_effort, "max_tokens": cfg.max_tokens}]
    base["auditor"] = {**base["auditor"], "model": None, "reasoning_effort": cfg.reasoning_effort,
                       "max_tokens": cfg.max_tokens}
    base["judge"] = {**base["judge"], "model": None, "max_tokens": cfg.max_tokens}
    path = OUT / f"game_input_{name}.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True, width=10**6))
    return str(path)


def play(cfg_path: str, n: int, seed: int, arm_name: str, key: str) -> int:
    env = os.environ | {"OPENROUTER_API_KEY": key}
    cmd = [sys.executable, "-m", "twenty_questions.run", "--config", cfg_path, "--n-games", str(n),
           "--base-seed", str(seed), "--out", str(OUT), "--arm", arm_name]
    with open(OUT / f"log_{arm_name}.txt", "w") as log:
        return subprocess.run(cmd, cwd=ENV, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def run_model(cfg, model, pilot, key, state):
    for arm in cfg.arms:
        name = cell(model, arm, pilot)
        with LOCK:
            if state["stopped"]:
                return
        cfg_path = game_yaml(cfg, model, arm, name)
        n, seed = (cfg.pilot_games, cfg.pilot_seed) if pilot else (cfg.n_games, cfg.base_seed)
        t0 = time.time()
        play(cfg_path, n, seed, name, key)
        games = json.loads((OUT / f"games_{name}.json").read_text())
        retries = []
        if cfg.retry_errored_once:
            for e in games["errors"]:
                rname = f"{name}_retry_s{e['seed']}"
                play(cfg_path, 1, e["seed"], rname, key)
                retries.append(rname)
        spent = spend(key) - state["spend0"]
        with LOCK:
            state["cells"].append({"cell": name, "model": model, "arm": arm, "seconds": round(time.time() - t0),
                                   "errors": len(games["errors"]), "retries": retries, "spent_so_far": spent})
            print(f"{name}: {len(games['games'])} games, {len(games['errors'])} errors, "
                  f"retries {len(retries)}, spent ${spent:.2f}", flush=True)
            if spent >= cfg.stop_launching_at * cfg.cost_cap_usd:
                state["stopped"] = True
                print(f"spend ${spent:.2f} >= {cfg.stop_launching_at:.0%} of the ${cfg.cost_cap_usd} cap: "
                      "no new cells", flush=True)
            (OUT / f"run_log{'_pilot' if pilot else ''}.json").write_text(json.dumps(state, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = AblationConfig.load()
    key = openrouter_key()
    state = {"pilot": args.pilot, "spend0": spend(key), "cells": [], "stopped": False}
    with cf.ThreadPoolExecutor(cfg.parallel_models) as pool:
        for f in [pool.submit(run_model, cfg, m, args.pilot, key, state) for m in cfg.models]:
            f.result()
    print(f"done: spent ${spend(key) - state['spend0']:.2f}")


if __name__ == "__main__":
    main()
