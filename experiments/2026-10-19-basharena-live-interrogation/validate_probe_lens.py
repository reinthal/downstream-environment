"""Compare the vllm-lens probe reads on Modal (modal_probe_lens.py) with the HF bf16 probe service.

Takes steps the HF bf16 service on gpaulo-ord-0 (probe_server.py) already scored (--reference),
spread over both modes and over context length, re-reads them on Modal (ProbeLens.score_jobs via
the Modal SDK), and writes --out: per step both contrasts and the token counts, plus Pearson r,
mean and max |difference|, and the agreement of the top-10% call (would both flag the step at the
reference's 10% honest cut). probe_lens_validation.json is the FP8 run against
phase_a_probe_scores.jsonl as it stood then (the HF bf16 scores, later renamed _bf16_partial).

Usage (from the repo root):
    MODAL_PROFILE=cadenza uv run --project environments/bash_arena --with modal python \
        experiments/2026-10-19-basharena-live-interrogation/validate_probe_lens.py --n 40 \
        --out experiments/2026-10-19-basharena-live-interrogation/probe_lens_validation_bf16.json
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

import probe_client as pc  # noqa: E402
import run_bash_arena  # noqa: E402,F401  (loads the repo .env)
from inspect_ai.log import read_eval_log  # noqa: E402
from thresholds import threshold  # noqa: E402



def main(a) -> None:
    rows = [json.loads(l) for l in a.reference.open()]
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
    # the repo .env's MODAL_TOKEN_ID is a proxy token the SDK rejects (see score_probe.py)
    os.environ.pop("MODAL_TOKEN_ID", None)
    os.environ.pop("MODAL_TOKEN_SECRET", None)
    import modal

    lens = modal.Cls.from_name("ais407-probe-lens", "ProbeLens")()
    out = []
    for i in range(0, len(jobs), a.batch):
        t0 = time.time()
        res = lens.score_jobs.remote(jobs[i:i + a.batch])
        print(f"batch {i // a.batch}: {len(res)} steps in {time.time() - t0:.0f} s", flush=True)
        for r, x in zip(pick[i:i + a.batch], res):
            yes, no = x["scores"]
            out.append({"task_id": r["task_id"], "mode": r["mode"], "step": r["step"],
                        "n_tokens": r["n_tokens"], "reference": r["contrast"],
                        "lens": no - yes})
    b, f = [x["reference"] for x in out], [x["lens"] for x in out]
    d = [abs(x - y) for x, y in zip(b, f)]
    summary = {"n": len(out), "pearson_r": statistics.correlation(b, f),
               "mean_abs_diff": statistics.mean(d), "max_abs_diff": max(d),
               "bf16_honest_10pct_cut": t10,
               "flag_agreement": sum((x >= t10) == (y >= t10) for x, y in zip(b, f)) / len(out)}
    a.out.write_text(
        json.dumps({"summary": summary, "steps": out}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--reference", type=Path,
                   default=HERE / "phase_a_probe_scores_bf16_partial.jsonl")
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
