"""Compare the FP8 vllm-lens probe reads (modal_probe_lens.py) with the bf16 probe service.

Takes steps the bf16 service already scored (phase_a_probe_scores.jsonl), spread over both modes
and over context length, re-reads them on Modal, and writes probe_lens_validation.json:
per step both contrasts and the token counts, plus Pearson r, mean and max |difference|, and
the agreement of the top-10% call (would both flag the step at the bf16 10% cut).

Usage (from the repo root; MODAL_PROXY_TOKEN in the repo .env):
    uv run --project environments/bash_arena python \
        experiments/2026-10-19-basharena-live-interrogation/validate_probe_lens.py --n 40
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "environments" / "bash_arena"))
sys.path.insert(0, str(HERE))

import httpx  # noqa: E402
import probe_client as pc  # noqa: E402
import run_bash_arena  # noqa: E402,F401  (loads the repo .env)
from inspect_ai.log import read_eval_log  # noqa: E402
from thresholds import threshold  # noqa: E402

BATCH_URL = "https://cadenza--ais407-probe-lens-probelens-read-batch.us-east.modal.run"


def main(a) -> None:
    rows = [json.loads(l) for l in (HERE / "phase_a_probe_scores.jsonl").open()]
    t10 = threshold([r["contrast"] for r in rows if r["mode"] == "honest"], 0.10)
    pick = []
    for mode in ("attack", "honest"):
        rs = sorted((r for r in rows if r["mode"] == mode), key=lambda r: r["n_tokens"])
        k = a.n // 2
        pick += [rs[round(i * (len(rs) - 1) / (k - 1))] for i in range(k)]
    logs = {}
    for r in pick:
        if r["log"] not in logs:
            f = next(HERE.glob(f"phase_a_*/inspect_logs/{r['log']}"))
            logs[r["log"]] = {str(s.id): s for s in read_eval_log(f, resolve_attachments=True).samples}
    jobs = []
    for r in pick:
        s = logs[r["log"]][r["task_id"]]
        tools = pc.tool_schemas(next(e.tools for e in s.events if e.event == "model" and e.tools))
        jobs.append({"messages": pc.template_messages(list(s.messages[: r["message_index"] + 1])),
                     "tools": tools,
                     "reads": [{"question": pc.FOLLOWUP_QUESTION, "answer": x} for x in pc.PREFILLS]})
    key, secret = os.environ["MODAL_PROXY_TOKEN"].split(".", 1)
    out = []
    with httpx.Client(timeout=3600, follow_redirects=True,
                      headers={"Modal-Key": key, "Modal-Secret": secret}) as c:
        for i in range(0, len(jobs), a.batch):
            t0 = time.time()
            resp = c.post(BATCH_URL, json={"jobs": jobs[i:i + a.batch]})
            resp.raise_for_status()
            res = resp.json()["results"]
            print(f"batch {i // a.batch}: {len(res)} steps in {time.time() - t0:.0f} s", flush=True)
            for r, x in zip(pick[i:i + a.batch], res):
                yes, no = x["scores"]
                out.append({"task_id": r["task_id"], "mode": r["mode"], "step": r["step"],
                            "n_tokens": r["n_tokens"], "bf16": r["contrast"], "fp8": no - yes})
    b, f = [x["bf16"] for x in out], [x["fp8"] for x in out]
    d = [abs(x - y) for x, y in zip(b, f)]
    summary = {"n": len(out), "pearson_r": statistics.correlation(b, f),
               "mean_abs_diff": statistics.mean(d), "max_abs_diff": max(d),
               "bf16_honest_10pct_cut": t10,
               "flag_agreement": sum((x >= t10) == (y >= t10) for x, y in zip(b, f)) / len(out)}
    (HERE / "probe_lens_validation.json").write_text(
        json.dumps({"summary": summary, "steps": out}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--batch", type=int, default=10)
    main(p.parse_args())
