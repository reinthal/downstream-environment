"""Convert Inspect eval logs to auditor_api_{arm}.jsonl for analyze_auditor_api.py.

Picks the newest completed carrot_auditor log per arm from inspect_logs/
(each sample id is "<game_id>:<condition>", epochs are the repeated auditor
seeds) and writes the same row schema as run_auditor_api.py.

Run:  uv run --no-sync python experiments/2026-09-25/inspect_to_jsonl.py
"""
from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.log import read_eval_log

OUT = Path(__file__).resolve().parent
LOGS = OUT / "inspect_logs"


def main():
    by_arm: dict[str, object] = {}
    for path in sorted(LOGS.glob("*.eval")):        # sorted by timestamp prefix
        log = read_eval_log(str(path), header_only=True)
        if log.status != "success" or log.eval.task != "carrot_auditor":
            continue
        arm = log.eval.task_args.get("arm", "base")
        n = log.results.total_samples if log.results else 0
        if n >= 2000:                               # full runs only, newest wins
            by_arm[arm] = path

    for arm, path in by_arm.items():
        log = read_eval_log(str(path))
        rows = []
        for s in log.samples:
            gid, cond = str(s.id).split(":")
            sc = s.scores["parsnip_scorer"]
            guess = None if sc.answer == "(unparsed)" else sc.answer
            rows.append({"arm": arm, "game_id": gid, "condition": cond,
                         "auditor_seed": s.epoch - 1, "guess": guess,
                         "parsnip": s.target, "correct": sc.value == "C",
                         "parse_failed": guess is None,
                         "judge_model": log.eval.model,
                         "reasoning": "", "public": (s.output.completion or "")[-500:]})
        rows.sort(key=lambda r: (r["game_id"], r["condition"], r["auditor_seed"]))
        out_path = OUT / f"auditor_api_{arm}.jsonl"
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{arm}: {len(rows)} rows from {path.name} -> {out_path}")


if __name__ == "__main__":
    main()
