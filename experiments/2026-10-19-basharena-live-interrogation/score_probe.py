"""Score action steps of logged trajectories with the fibs probe (step read), post hoc.

Phase A runs non-blocking, so the probe protocol's suspicion (the step contrast) can be read
afterwards from the eval logs through the Modal vllm-lens deployment (modal_probe_lens.py), called
with the Modal SDK (ProbeLens.score_jobs; the web endpoints cancelled long batched requests), in
batches of --batch steps, --concurrency in flight. Attack
steps first, then honest; --every k keeps every k-th honest action step (the calibration
subsample; attack steps are all scored). One JSON line per step: log, task_id, mode, step,
message_index, contrast, p_yes, p_no, n_tokens, seconds (--read followup, the fibs read) or
p_self, n_tokens, seconds (--read self: U's own step, no follow-up). --resume skips steps
already in --out.

Usage (from the repo root; one call per Phase A log dir into the same --out):
    MODAL_PROFILE=cadenza uv run --project environments/bash_arena --with modal python \
        experiments/2026-10-19-basharena-live-interrogation/score_probe.py \
        --logs experiments/2026-10-19-basharena-live-interrogation/phase_a_honest/inspect_logs \
        --out experiments/2026-10-19-basharena-live-interrogation/phase_a_probe_scores.jsonl --resume
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "environments" / "bash_arena"))

import probe_client  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import ChatMessageAssistant  # noqa: E402
from interrogation import _is_action  # noqa: E402


def jobs(log_dir: Path, every: int) -> list[dict]:
    out = []
    for f in sorted(log_dir.glob("*.eval")):
        log = read_eval_log(f, resolve_attachments=True)
        for s in log.samples or []:
            if s.error:
                continue
            mode = s.metadata["eval_mode"]
            tools = next(e.tools for e in s.events if e.event == "model" and e.tools)
            steps = [i for i, m in enumerate(s.messages)
                     if isinstance(m, ChatMessageAssistant) and _is_action(m)]
            for k, i in enumerate(steps):
                if mode == "honest" and k % every:
                    continue
                out.append({"log": f.name, "task_id": str(s.id), "mode": mode, "step": k,
                            "message_index": i, "messages": list(s.messages[: i + 1]),
                            "tools": tools})
    return sorted(out, key=lambda j: j["mode"] != "attack")


async def main(a) -> None:
    done = set()
    if a.out.exists():
        if not a.resume:
            raise SystemExit(f"{a.out} exists; pass --resume to continue it")
        done = {(r["log"], r["task_id"], r["step"]) for r in map(json.loads, a.out.open())}
    todo = [j for j in jobs(a.logs, a.every) if (j["log"], j["task_id"], j["step"]) not in done]
    print(f"{len(todo)} steps to score ({len(done)} done)", flush=True)
    t0 = time.time()
    # The repo .env (loaded by run_bash_arena) sets MODAL_TOKEN_ID to a proxy token (wk-...),
    # which the Modal SDK rejects as an API token ("Token ID is malformed") and which would
    # override the profile; the SDK must authenticate with MODAL_PROFILE's token.
    os.environ.pop("MODAL_TOKEN_ID", None)
    os.environ.pop("MODAL_TOKEN_SECRET", None)
    import modal  # run with `uv run --with modal`, MODAL_PROFILE=cadenza

    lens = modal.Cls.from_name("ais407-probe-lens", "ProbeLens")()
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    n_done = 0
    with a.out.open("a") as fh:
        async def one(batch):
            nonlocal n_done
            if a.read == "self":
                jobs = [{"messages": probe_client.template_messages(j["messages"]),
                         "tools": probe_client.tool_schemas(j["tools"]), "self": True}
                        for j in batch]
            else:
                jobs = [probe_client.step_job(j["messages"], j["tools"]) for j in batch]
            async with sem:
                results = await lens.score_jobs.remote.aio(jobs)
            async with lock:
                for j, res in zip(batch, results):
                    if a.read == "self":
                        (p_self,) = res["scores"]
                        read = {"p_self": p_self, "n_tokens": res["n_tokens"][0],
                                "seconds": res["seconds"]}
                    else:
                        read = probe_client.contrast_of(res)
                    row = ({k: j[k] for k in ("log", "task_id", "mode", "step", "message_index")}
                           | read)
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                n_done += len(batch)
                rate = (time.time() - t0) / n_done
                print(f"{n_done}/{len(todo)}  {rate:.1f} s/step  "
                      f"eta {rate * (len(todo) - n_done) / 3600:.1f} h", flush=True)
        await asyncio.gather(*(one(b) for b in batches))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--every", type=int, default=1, help="Keep every k-th honest step.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch", type=int, default=10, help="Steps per score_jobs call.")
    p.add_argument("--read", choices=["followup", "self"], default="followup",
                   help="followup: the fibs follow-up, contrast No. - Yes.; self: U's own step "
                   "(the token before its <|im_end|>), p_self.")
    p.add_argument("--concurrency", type=int, default=2, help="Batches in flight.")
    asyncio.run(main(p.parse_args()))
