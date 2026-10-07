"""Outside auditor on a hosted Claude model via the Anthropic Messages API
(e.g. Claude Opus 5.5 through OpenRouter's Anthropic-compatible endpoint).

Same prompts as the local runner (auditor.run.prepare), same answer parsing,
and for tool conditions the same sandboxed Python tool (auditor.sandbox),
here as a native Claude tool in a manual tool-use loop. Adaptive thinking
with summarised reasoning is logged; token usage and estimated cost are
tracked per audit, and no new audits start once max_cost_usd is reached.
Interrupted runs resume: audits already in the output file are skipped.

Key: never in code or config. Set the variable named by cfg.api_key_env
(default OPENROUTER_API_KEY) in the environment, or put that line in the
repo-root .env (gitignored).

Run (.venv-api: anthropic SDK + numpy):
  .venv-api/bin/python -m auditor.api_run --config experiments/<date>/<dir>/config.json --arm base
Dry run (prompts + cost estimate, no API calls): add --dry-run
Smoke: add --games 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from auditor.config import ApiAuditorConfig  # noqa: E402
from auditor.prompts import parse_guess, score_rows_csv  # noqa: E402
from auditor.run import prepare  # noqa: E402
from auditor.sandbox import run_python  # noqa: E402
from auditor.tools import LIMIT_MSG, TOOLS  # noqa: E402

PY_TOOL = {
    "name": "python",
    "description": TOOLS[0]["function"]["description"],
    "input_schema": {"type": "object",
                     "properties": {"code": {"type": "string", "description": "Python code to run"}},
                     "required": ["code"], "additionalProperties": False},
}
FINAL_ANSWER = ("Give your final answer now: exactly one player name inside answer tags, "
                "e.g. <answer>Alice</answer>.")


def load_key(var: str, cfg_base_url_hint: str = "") -> str:
    """The key from the environment, else from the repo-root .env. No
    cleaning: a malformed value fails here, reporting its shape only."""
    raw, where = os.environ.get(var), "environment"
    env = REPO / ".env"
    if not raw and env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.strip().removeprefix("export ").partition("=")
            if k.strip() == var and v.strip():
                raw, where = v, str(env)
    if not raw:
        sys.exit(f"No API key: set {var} in the environment or add '{var}=...' to {env} (gitignored).")
    key = raw.strip()
    odd = [i for i, ch in enumerate(key) if not (ch.isalnum() or ch in "-_")]
    if odd:
        # report the shape only, never the key itself
        sys.exit(f"The {var} value in {where} is malformed: {len(key)} characters, unexpected "
                 f"characters (quotes, paste markers, spaces?) at positions {odd[:10]}. "
                 f"Re-enter it with a text editor.")
    if "openrouter" in cfg_base_url_hint and not key.startswith("sk-or-"):
        sys.exit(f"The {var} value in {where} is not an OpenRouter key (they start with sk-or-; "
                 f"{len(key)} characters). An exported variable takes precedence over .env.")
    return key


class Spend:
    def __init__(self, cfg):
        self.cfg, self.lock, self.usd = cfg, threading.Lock(), 0.0

    def cost(self, u) -> float:
        c = self.cfg
        return ((u["input_tokens"] * c.price_in_per_m + u["output_tokens"] * c.price_out_per_m
                 + u["cache_read_input_tokens"] * c.price_cache_read_per_m
                 + u["cache_creation_input_tokens"] * c.price_cache_write_per_m) / 1e6)

    def add(self, usd: float):
        with self.lock:
            self.usd += usd

    def over(self) -> bool:
        return self.usd >= self.cfg.max_cost_usd


def audit(client, cfg, conv, files, player_names) -> dict:
    """One audit: a single call, or a tool-use loop for tool conditions."""
    kwargs = dict(model=cfg.api_model, max_tokens=cfg.api_max_tokens, system=conv[0]["content"],
                  thinking={"type": "adaptive", "display": "summarized"},
                  output_config={"effort": cfg.api_effort})
    if cfg.api_cache:
        kwargs["cache_control"] = {"type": "ephemeral"}
    if files is not None:
        kwargs["tools"] = [PY_TOOL]
    messages = [{"role": "user", "content": conv[1]["content"]}]
    usage = dict(input_tokens=0, output_tokens=0, cache_read_input_tokens=0,
                 cache_creation_input_tokens=0)
    thinking, tool_calls, n_calls, stop = [], [], 0, None

    def call():
        r = client.messages.create(messages=messages, **kwargs)
        for k in usage:
            usage[k] += getattr(r.usage, k) or 0     # cache fields are None when unused
        thinking.extend(b.thinking for b in r.content if b.type == "thinking" and b.thinking)
        return r

    resp = call()
    while resp.stop_reason == "tool_use" and len(tool_calls) <= cfg.max_tool_calls:
        uses = [b for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in uses:
            code = tu.input.get("code") if isinstance(tu.input, dict) else None
            if n_calls >= cfg.max_tool_calls:
                out = LIMIT_MSG
            elif not isinstance(code, str):
                out = "[error] the tool needs a 'code' string"
            else:
                out = run_python(code, files)
                n_calls += 1
                tool_calls.append({"code": code, "output": out})
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
        messages.append({"role": "user", "content": results})
        resp = call()
    stop = resp.stop_reason
    text = "".join(b.text for b in resp.content if b.type == "text")
    source = "tag" if parse_guess(text, player_names) else None
    if source is None and stop != "refusal":
        messages += [{"role": "assistant", "content": resp.content},
                     {"role": "user", "content": FINAL_ANSWER}]
        resp = call()
        text += "\n" + "".join(b.text for b in resp.content if b.type == "text")
        source = "followup" if parse_guess(text, player_names) else None
    return {"text": text, "thinking": "\n\n".join(thinking), "stop_reason": stop,
            "answer_source": source, "usage": usage,
            "tool": {"n_tool_calls": n_calls, "tool_calls": tool_calls} if files is not None else None}


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", default="base")
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--games", type=int, default=0, help="limit games (smoke test)")
    ap.add_argument("--max-cost", type=float, default=None, help="override max_cost_usd")
    ap.add_argument("--dry-run", action="store_true", help="print prompts + estimate, no API calls")
    args = ap.parse_args(argv)

    cfg = ApiAuditorConfig.load(args.config)
    if args.max_cost is not None:
        cfg.max_cost_usd = args.max_cost
    conditions = args.conditions.split(",") if args.conditions else cfg.conditions
    P = prepare(cfg, args.arm, conditions, games=args.games)
    seeds = range(cfg.n_auditor_seeds)
    jobs = [(i, s) for s in seeds for i in range(len(P.jobs))]
    out = cfg.path(cfg.out_dir) / f"auditor_{args.arm}{'_smoke' if args.games else ''}.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            done.add((r["game_id"], r["condition"], r["auditor_seed"]))
    todo = [(i, s) for i, s in jobs if (*P.jobs[i], s) not in done]
    print(f"{len(jobs)} audits ({len(P.game_ids)} games x {conditions} x {len(seeds)} seed(s)); "
          f"{len(todo)} to run -> {out}", flush=True)

    if args.dry_run:
        c = P.convs[-1]
        print("=== system ===\n" + c[0]["content"] + "\n=== user (first 2500 chars) ===\n"
              + c[1]["content"][:2500])
        n_tool = sum(P.jobs[i][1] in cfg.tool_conditions for i, _ in todo)
        est = ((len(todo) - n_tool) * (1500 * cfg.price_in_per_m + 6500 * cfg.price_out_per_m)
               + n_tool * (8500 * cfg.price_in_per_m + 3300 * cfg.price_out_per_m)) / 1e6
        print(f"\nestimated cost if it thinks as much as Qwen did: ${est:.2f} "
              f"(roughly half that if it thinks half as much); cap ${cfg.max_cost_usd:.2f}")
        return

    import anthropic
    client = anthropic.Anthropic(base_url=cfg.api_base_url, auth_token=load_key(cfg.api_key_env, cfg.api_base_url),
                                 api_key=None, max_retries=6, timeout=900)
    if not args.games:
        cfg.save(cfg.path(cfg.out_dir))
    spend, lock, t0 = Spend(cfg), threading.Lock(), time.time()

    def run(job):
        i, s = job
        if spend.over():
            return None
        gid, cond = P.jobs[i]
        files = ({"scores.csv": score_rows_csv(P.raw_scores[cond][gid])}
                 if cond in cfg.tool_conditions else None)
        res = audit(client, cfg, P.convs[i], files, P.player_names)
        usd = spend.cost(res["usage"])
        spend.add(usd)
        guess = parse_guess(res["text"], P.player_names)
        row = {"arm": args.arm, "game_id": gid, "condition": cond, "auditor_seed": s,
               "guess": guess, "parsnip": P.parsnip[gid], "correct": guess == P.parsnip[gid],
               "parse_failed": guess is None, "answer_source": res["answer_source"],
               "stop_reason": res["stop_reason"], "tool": res["tool"], "usage": res["usage"],
               "cost_usd": round(usd, 5), "model": cfg.api_model, "effort": cfg.api_effort,
               "reasoning": res["thinking"], "public": res["text"]}
        with lock, open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    n = correct = 0
    with ThreadPoolExecutor(cfg.workers) as ex:
        futures = [ex.submit(run, j) for j in todo]
        for fut in as_completed(futures):
            row = fut.result()
            if row is None:
                continue
            n += 1
            correct += row["correct"]
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} done, {correct} correct, est. spend ${spend.usd:.2f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    if spend.over():
        print(f"stopped at the cost cap (${cfg.max_cost_usd:.2f}); rerun to resume", flush=True)
    print(f"finished: {n} audits this run, est. spend ${spend.usd:.2f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
