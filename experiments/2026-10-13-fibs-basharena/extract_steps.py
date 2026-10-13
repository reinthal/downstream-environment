"""Read the Qwen3.8-27B BashArena rollouts (Inspect logs of 2026-10-10) into
one record per trajectory, with the message list as the model saw it, the
tool schemas, and an index of the model's steps.

A step is one assistant message produced by the model (the paper: "each step
is reasoning ending in a tool call"; a message without a tool call is a step
too and is judged by its text). The auto-submit wrapper's forced ``submit``
(``message.model == "side_task_auto_submit"``) is not a model step and is
flagged, as is the step after which the wrapper reported a side-task pass
(the user message "Side task '<name>' passed: ...").

  transcripts_{mode}.jsonl   one trajectory per line:
      sample_id, task_id, epoch, mode, error, limit, scores,
      tools (OpenAI function schemas), messages (role/content/reasoning/
      tool_calls/tool_call_id), steps [{step, msg_index, n_tool_calls,
      tool_names, has_text, forced, pass_after}], side_task_pass_after_step

Run:  uv run --project environments/bash_arena python experiments/2026-10-13-fibs-basharena/extract_steps.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

OUT = Path(__file__).resolve().parent
LOG_DIR = OUT.parents[1] / "experiments" / "2026-10-10-qwen-logit-monitor" / "inspect_logs"
PASS_PREFIX = "Side task '"
FORCED_MODEL = "side_task_auto_submit"


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(getattr(p, "text", "") for p in content if getattr(p, "type", None) == "text")


def reasoning_of(content) -> str:
    if isinstance(content, str):
        return ""
    return "".join(p.reasoning for p in content if getattr(p, "type", None) == "reasoning")


def convert_message(m) -> dict:
    d = {"role": m.role}
    if m.role == "assistant":
        d["content"] = text_of(m.content)
        d["reasoning"] = reasoning_of(m.content)
        d["model"] = m.model
        d["tool_calls"] = [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function, "arguments": tc.arguments}}
                           for tc in (m.tool_calls or [])]
    elif m.role == "tool":
        d["content"] = text_of(m.content)
        d["tool_call_id"] = m.tool_call_id
        d["name"] = m.function
        if m.error is not None:
            d["error"] = m.error.message
    else:
        d["content"] = text_of(m.content)
    return d


def tool_schemas(sample) -> list[dict]:
    # the agent's calls carry the tool schemas; the logit monitor's calls (thinking off, no tools)
    # are model events of the same sample and are skipped
    events = [e for e in sample.events if e.event == "model" and e.tools]
    if not events:
        raise ValueError(f"{sample.id}: no model events with tools")
    first = events[0]
    schemas = [{"type": "function", "function": {
        "name": t.name, "description": t.description,
        "parameters": t.parameters.model_dump(exclude_none=True)}} for t in first.tools]
    for e in events[1:]:
        if [t.name for t in e.tools] != [t.name for t in first.tools]:
            raise ValueError(f"{sample.id}: tool set changes between model calls")
    return schemas


def convert_sample(sample, mode: str) -> dict:
    msgs = [convert_message(m) for m in sample.messages]
    if msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
        raise ValueError(f"{sample.id}: unexpected opening messages {[m['role'] for m in msgs[:2]]}")
    steps, pass_after = [], None
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        forced = m["model"] == FORCED_MODEL
        nxt = msgs[i + 1] if i + 1 < len(msgs) else None
        # the wrapper checks side tasks before the NEXT step and, on a pass, appends the user notice
        # then the forced submit; so a pass notice right after this step's tool results belongs to it
        passed = False
        j = i + 1
        while j < len(msgs) and msgs[j]["role"] == "tool":
            j += 1
        if j < len(msgs) and msgs[j]["role"] == "user" and msgs[j]["content"].startswith(PASS_PREFIX):
            passed = True
        step = {"step": sum(1 for s in steps if not s["forced"]) + (0 if forced else 1),
                "msg_index": i, "n_tool_calls": len(m["tool_calls"]),
                "tool_names": [tc["function"]["name"] for tc in m["tool_calls"]],
                "has_text": bool(m["content"].strip()), "forced": forced, "pass_after": passed}
        if forced:
            step["step"] = None
        steps.append(step)
        if passed:
            if pass_after is not None:
                raise ValueError(f"{sample.id}: two side-task pass notices")
            pass_after = step["step"]
        del nxt
    scores = {k: (v.value if not isinstance(v.value, dict) else v.value) for k, v in (sample.scores or {}).items()}
    # sample ids repeat across epochs (the task id); uid is the unique key every stage uses
    return {"sample_id": sample.id, "task_id": str(sample.id), "epoch": sample.epoch,
            "uid": f"{sample.id}-{sample.epoch}", "mode": mode,
            "error": None if sample.error is None else sample.error.message,
            "limit": None if sample.limit is None else sample.limit.type,
            "scores": scores, "tools": tool_schemas(sample), "messages": msgs, "steps": steps,
            "side_task_pass_after_step": pass_after}


def main():
    for mode in ("honest", "attack"):
        (path,) = glob.glob(str(LOG_DIR / f"*-{mode}-*.eval"))
        log = read_eval_log(path)
        recs = [convert_sample(s, mode) for s in log.samples]
        n_err = sum(r["error"] is not None for r in recs)
        n_steps = sum(sum(1 for s in r["steps"] if not s["forced"]) for r in recs if r["error"] is None)
        n_forced = sum(sum(1 for s in r["steps"] if s["forced"]) for r in recs)
        n_pass = sum(r["side_task_pass_after_step"] is not None for r in recs)
        out = OUT / f"transcripts_{mode}.jsonl"
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{mode}: {len(recs)} samples ({n_err} errored), {n_steps} model steps in the valid ones, "
              f"{n_forced} forced submits, {n_pass} with a side-task pass notice -> {out.name}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
