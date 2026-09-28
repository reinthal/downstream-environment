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
    # merge every successful carrot_auditor log; for a duplicate
    # (game, condition, epoch) the newest log (timestamp-sorted) wins
    merged: dict[str, dict] = {}
    for path in sorted(LOGS.glob("*.eval")):        # sorted by timestamp prefix
        head = read_eval_log(str(path), header_only=True)
        if head.status != "success" or head.eval.task != "carrot_auditor":
            continue
        arm = head.eval.task_args.get("arm", "base")
        log = read_eval_log(str(path))
        n = 0
        for s in log.samples:
            gid, cond = str(s.id).split(":")
            sc = s.scores["parsnip_scorer"]
            guess = None if sc.answer == "(unparsed)" else sc.answer
            merged.setdefault(arm, {})[(gid, cond, s.epoch)] = {
                "arm": arm, "game_id": gid, "condition": cond,
                "auditor_seed": s.epoch - 1, "guess": guess,
                "parsnip": s.target, "correct": sc.value == "C",
                "parse_failed": guess is None, "judge_model": log.eval.model,
                "reasoning": "", "public": (s.output.completion or "")[-500:]}
            n += 1
        print(f"{path.name[:45]}: {n} samples ({arm})")

    for arm, rows in merged.items():
        out_path = OUT / f"auditor_api_{arm}.jsonl"
        with open(out_path, "w") as f:
            for k in sorted(rows):
                f.write(json.dumps(rows[k]) + "\n")
        print(f"{arm}: {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
