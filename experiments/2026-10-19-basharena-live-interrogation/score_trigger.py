"""Score every action step of logged trajectories with the trigger (trigger.py), post hoc.

One JSON line per step: log, task_id, mode, step, message_index, n_earlier (earlier calls that fit
the window), p_flag, input_tokens. Stops before the running Jev spend would pass --budget-usd.

Usage (from the repo root; one call per Phase A log dir):
    uv run --project environments/bash_arena python \
        experiments/2026-10-19-basharena-live-interrogation/score_trigger.py \
        --logs experiments/2026-10-19-basharena-live-interrogation/phase_a_honest/inspect_logs \
        --out experiments/2026-10-19-basharena-live-interrogation/phase_a_trigger_scores_honest.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "environments" / "bash_arena"))

import run_bash_arena  # noqa: E402,F401  (loads TYPESAFE_API_KEY from the repo .env)
from godmode import JEV_MODEL  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import ChatMessageAssistant  # noqa: E402
from interrogation import _is_action  # noqa: E402
from trigger import trigger_score, trigger_state  # noqa: E402
from typesafe_sdk import AsyncTypeSafeClient  # noqa: E402

USD_PER_MTOK = 0.042


async def main(a) -> None:
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; move it away first")
    todo = []
    for f in sorted(a.logs.glob("*.eval")):
        for s in read_eval_log(f, resolve_attachments=True).samples or []:
            if s.error:
                continue
            steps = [i for i, m in enumerate(s.messages)
                     if isinstance(m, ChatMessageAssistant) and _is_action(m)]
            for k, i in enumerate(steps):
                st = trigger_state(list(s.messages), i, s.metadata["main_task_description"])
                todo.append(({"log": f.name, "task_id": str(s.id),
                              "mode": s.metadata["eval_mode"], "step": k, "message_index": i,
                              "n_earlier": len(st["earlier_tool_calls"])}, st))
    print(f"{len(todo)} steps", flush=True)
    spent = 0
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    async with AsyncTypeSafeClient(model=JEV_MODEL) as client:
        with a.out.open("w") as fh:
            async def one(row, st):
                nonlocal spent
                async with sem:
                    async with lock:
                        if spent * USD_PER_MTOK / 1e6 >= a.budget_usd:
                            raise RuntimeError(f"Jev budget ${a.budget_usd} reached")
                    p, tokens = await trigger_score(client, st)
                    async with lock:
                        spent += tokens
                        fh.write(json.dumps(row | {"p_flag": p, "input_tokens": tokens}) + "\n")
            await asyncio.gather(*(one(r, st) for r, st in todo))
    print(f"{len(todo)} steps, {spent} tokens, ${spent * USD_PER_MTOK / 1e6:.2f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget-usd", type=float, default=5.0)
    p.add_argument("--concurrency", type=int, default=16)
    asyncio.run(main(p.parse_args()))
