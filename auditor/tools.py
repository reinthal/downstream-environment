"""Tool-using auditor: a batched agent loop with a sandboxed Python tool.

Each audit is a multi-turn episode in Qwen's native tool format: the model
thinks, may emit a <tool_call> with Python code, the code runs in
auditor.sandbox with the audit's data files, the output comes back as a tool
message, and the model continues — until it answers without a tool call.
All active audits advance one step at a time in a single vLLM batch.

Limits: at most `max_calls` executed tool calls per audit (further calls are
answered "tool limit reached"), and `max_new_tokens` generated tokens in total
across an episode's steps.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from auditor.prompts import ANSWER_CUE, parse_guess
from auditor.sandbox import run_python
from rollouts.generation import CLOSE_CUE

TOOLS = [{"type": "function", "function": {
    "name": "python",
    "description": ("Run Python 3 code (numpy, pandas and scipy are available). The detector "
                    "scores are in the file scores.csv. Use print() to see results."),
    "parameters": {"type": "object", "properties": {"code": {"type": "string",
                                                              "description": "Python code to run"}},
                   "required": ["code"]}}}]
STOP = "</tool_call>"
CODE_RE = re.compile(r"<parameter=code>\n?(.*?)\n?</parameter>", re.DOTALL)
STEP_TOKENS = 4000
LIMIT_MSG = "Tool limit reached. Give your final answer now, without calling the tool again."


def parse_tool_call(out: str) -> tuple[str, str, str | None] | None:
    """(reasoning, content before the call, code) if `out` ends in a tool call,
    else None. code is None when the call is malformed."""
    if "<tool_call>" not in out:
        return None
    head, call = out.split("<tool_call>", 1)
    if "</think>" in head:
        reasoning, content = head.split("</think>", 1)
    else:
        reasoning, content = head, ""
    m = CODE_RE.search(call)
    return reasoning.replace("<think>", "").strip(), content.strip(), (m.group(1) if m else None)


def run_tool_audits(backend, convs: list[list[dict]], files: list[dict[str, str]], *,
                    max_new_tokens: int, max_calls: int, seed: int, player_names: list[str],
                    force_close_tokens: int, answer_followup: bool):
    """Returns per audit: (final raw output, log dict)."""
    n = len(convs)
    msgs = [list(c) for c in convs]
    used, calls = [0] * n, [0] * n
    logs = [{"tool_calls": []} for _ in range(n)]
    final: list[str | None] = [None] * n
    prefix: list[str] = [""] * n
    base_kwargs = dict(backend.gen_kwargs or {})
    tok = backend.tok

    def render(i):
        return tok.apply_chat_template(msgs[i], tools=TOOLS, tokenize=False, add_generation_prompt=True)

    step = 0
    while any(f is None for f in final):
        active = [i for i in range(n) if final[i] is None]
        texts = [render(i) for i in active]
        cap = max(64, min(STEP_TOKENS, max(max_new_tokens - used[i] for i in active)))
        backend.gen_kwargs = {**base_kwargs, "stop": [STOP]}
        outs = backend._generate_texts(texts, cap, seed + 1000 * step, None)
        backend.gen_kwargs = base_kwargs
        pending = []
        for i, text, out in zip(active, texts, outs):
            used[i] += len(tok(out, add_special_tokens=False)["input_ids"])
            parsed = parse_tool_call(out)
            out_of_budget = used[i] >= max_new_tokens
            if parsed is None or out_of_budget:
                if parsed is not None:                 # budget gone mid-call: drop the call
                    out = out.split("<tool_call>", 1)[0]
                final[i], prefix[i] = out, text
                continue
            reasoning, content, code = parsed
            pending.append((i, reasoning, content, code))
        def execute(item):
            i, reasoning, content, code = item
            if code is None:
                return "[error] could not read the tool call; put the code in the 'code' parameter"
            if calls[i] >= max_calls:
                return LIMIT_MSG
            return run_python(code, files[i])
        with ThreadPoolExecutor(16) as ex:
            results = list(ex.map(execute, pending))
        for (i, reasoning, content, code), result in zip(pending, results):
            if code is not None and calls[i] < max_calls:
                calls[i] += 1
                logs[i]["tool_calls"].append({"code": code, "output": result})
            msgs[i] += [{"role": "assistant", "content": content, "reasoning_content": reasoning,
                         "tool_calls": [{"type": "function", "function": {
                             "name": "python", "arguments": {"code": code or ""}}}]},
                        {"role": "tool", "content": result}]
        step += 1

    # close unfinished thinking, then make sure there is an answer tag
    unclosed = [i for i in range(n) if "</think>" not in final[i]]
    if unclosed:
        conts = backend.complete([prefix[i] + final[i] + CLOSE_CUE for i in unclosed],
                                 force_close_tokens, seed=seed + 1)
        for i, c in zip(unclosed, conts):
            final[i] = final[i] + CLOSE_CUE + c
            logs[i]["force_closed"] = True
    if answer_followup:
        todo = [i for i in range(n) if parse_guess(final[i], player_names) is None]
        conts = backend.complete([prefix[i] + final[i] + ANSWER_CUE for i in todo], 10,
                                 seed=seed + 2) if todo else []
        for i, c in zip(todo, conts):
            final[i] = final[i] + ANSWER_CUE + c
            logs[i]["answer_followup"] = True
    for i in range(n):
        logs[i]["n_tool_calls"] = calls[i]
        logs[i]["generated_tokens"] = used[i]
    return list(zip(final, logs))
