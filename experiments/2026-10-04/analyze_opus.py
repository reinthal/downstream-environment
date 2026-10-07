"""Opus 5.5 auditor vs Qwen3.5-27B auditor on the same raw-score conditions (CPU).

Same metrics as analyze_audits.py for audit_opus/auditor_base.jsonl, side by
side with the Qwen results (audit_a + audit_b), plus cost and how the
analysis was done (round / position adjustments in the code it ran).

Output: results_opus.json

Run:  .venv/bin/python experiments/2026-10-04/analyze_opus.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_audits import RAW, ci, picks

OUT = Path(__file__).resolve().parent


def summarise(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    res = {}
    for cond, rs in by.items():
        d = {"n": len(rs), "accuracy": round(float(np.mean([r["correct"] for r in rs])), 3), "ci95": ci(rs)}
        if cond in RAW:
            data = json.loads((OUT / f"{RAW[cond]}_base.json").read_text())
            pk = {g: picks(v) for g, v in data.items()}
            follow = [r["guess"] == pk[r["game_id"]][0] for r in rs]
            follow_pos = [r["guess"] == pk[r["game_id"]][1] for r in rs]
            over = [r for r, f in zip(rs, follow) if not f]
            d.update(picks_raw_average_top=round(float(np.mean(follow)), 3),
                     picks_position_corrected_top=round(float(np.mean(follow_pos)), 3),
                     overrides=len(over), overrides_correct=int(sum(r["correct"] for r in over)))
            tools = [r["tool"] for r in rs if r.get("tool")]
            if tools:
                codes = [" ".join(c["code"] for c in t["tool_calls"]) for t in tools]
                d["mean_tool_calls"] = round(float(np.mean([t["n_tool_calls"] for t in tools])), 2)
                # adjusts by position / round: groups, indexes or normalises by that column
                adj = lambda col: r"groupby\(\s*\[?['\"]" + col + r"|\[['\"]" + col + r"['\"]\]\s*==|\." + col + r"\b"  # noqa: E731
                d["code_adjusts_position"] = round(float(np.mean([bool(re.search(adj("position"), c)) for c in codes])), 3)
                d["code_adjusts_round"] = round(float(np.mean([bool(re.search(adj("round"), c)) for c in codes])), 3)
        if "cost_usd" in rs[0]:
            d["cost_usd"] = round(float(sum(r["cost_usd"] for r in rs)), 2)
        res[cond] = d
    return res


def main():
    opus = [json.loads(l) for l in open(OUT / "audit_opus" / "auditor_base.jsonl")]
    qwen = [json.loads(l) for d in ("audit_a", "audit_b") for l in open(OUT / d / "auditor_base.jsonl")]
    out = {"opus": summarise(opus), "qwen": summarise(qwen),
           "opus_total_cost_usd": round(sum(r["cost_usd"] for r in opus), 2),
           "opus_stop_reasons": {k: sum(r["stop_reason"] == k for r in opus)
                                 for k in {r["stop_reason"] for r in opus}}}
    (OUT / "results_opus.json").write_text(json.dumps(out, indent=1) + "\n")
    for cond in out["opus"]:
        o, q = out["opus"][cond], out["qwen"].get(cond, {})
        print(f"{cond:10s} OPUS {o['accuracy']:.2f} {o['ci95']} | QWEN {q.get('accuracy', float('nan')):.2f} {q.get('ci95')}")
        for k in ("picks_raw_average_top", "picks_position_corrected_top", "overrides", "overrides_correct",
                  "mean_tool_calls", "code_adjusts_round", "code_adjusts_position", "cost_usd"):
            if k in o:
                print(f"           {k}: opus {o[k]} | qwen {q.get(k)}")
    print("total opus cost $", out["opus_total_cost_usd"], "| stop reasons", out["opus_stop_reasons"])


if __name__ == "__main__":
    main()
