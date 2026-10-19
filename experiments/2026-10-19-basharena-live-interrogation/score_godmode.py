"""Score every action step of logged trajectories with the god-mode detector (Jev), post hoc.

Phase A runs non-blocking, so Jev's per-step p_attack can be computed afterwards from the eval
logs; Phase B's god-mode protocol makes the same call live (environments/bash_arena/godmode.py).
Writes one JSON line per step: log, task_id, mode, step (0-based index among action steps),
message_index, p_attack, input_tokens, model. Stops (raises) before a request would take the
running Jev spend past --budget-usd.

Usage (from the repo root):
    uv run --project environments/bash_arena python \
        experiments/2026-10-19-basharena-live-interrogation/score_godmode.py \
        --logs experiments/2026-10-19-basharena-live-interrogation/phase_a/inspect_logs \
        --out experiments/2026-10-19-basharena-live-interrogation/phase_a/godmode_scores.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "environments" / "bash_arena"))

import run_bash_arena  # noqa: E402,F401  (loads TYPESAFE_API_KEY from the repo .env)
from godmode import JEV_MODEL, judge_step, step_state  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import ChatMessageAssistant  # noqa: E402
from interrogation import _is_action  # noqa: E402
from typesafe_sdk import AsyncTypeSafeClient  # noqa: E402

USD_PER_MTOK = 0.042  # Jev 1.14 input price (docs.typesafe.ai/models); output is free


def jobs(log_dir: Path) -> list[dict]:
    out = []
    for f in sorted(log_dir.glob("*.eval")):
        log = read_eval_log(f, resolve_attachments=True)
        for s in log.samples or []:
            if s.error:
                continue
            task = s.metadata["main_task_description"]
            steps = [
                i for i, m in enumerate(s.messages)
                if isinstance(m, ChatMessageAssistant) and _is_action(m)
            ]
            for k, i in enumerate(steps):
                out.append({
                    "log": f.name, "task_id": str(s.id), "mode": s.metadata["eval_mode"],
                    "step": k, "message_index": i,
                    "state": step_state(list(s.messages), i, task),
                })
    return out


async def score(todo: list[dict], out: Path, budget_usd: float, concurrency: int) -> None:
    spent_tokens = 0
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    async with AsyncTypeSafeClient(model=JEV_MODEL) as client:
        with out.open("w") as fh:
            async def one(job):
                nonlocal spent_tokens
                async with sem:
                    async with lock:
                        if (spent_tokens * USD_PER_MTOK / 1e6) >= budget_usd:
                            raise RuntimeError(f"Jev budget ${budget_usd} reached")
                    j = await judge_step(client, job["state"])
                    async with lock:
                        spent_tokens += j.input_tokens
                        row = {k: v for k, v in job.items() if k != "state"} | j.to_dict()
                        fh.write(json.dumps(row) + "\n")
            await asyncio.gather(*(one(j) for j in todo))
    print(f"{len(todo)} steps, {spent_tokens} input tokens, "
          f"${spent_tokens * USD_PER_MTOK / 1e6:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget-usd", type=float, default=24.0)
    p.add_argument("--concurrency", type=int, default=16)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; move it away first")
    todo = jobs(a.logs)
    est = sum(len(json.dumps(j["state"])) for j in todo) / 4
    print(f"{len(todo)} steps, ~{est / 1e6:.1f}M tokens est., ~${est * USD_PER_MTOK / 1e6:.2f}")
    asyncio.run(score(todo, a.out, a.budget_usd, a.concurrency))
