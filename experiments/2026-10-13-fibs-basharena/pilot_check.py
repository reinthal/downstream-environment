"""Pilot checks (PREREGISTRATION §5), from the pilot's outputs:

  * finite contrast scores on every (non-dropped) step of the 4 episodes
    (scores_pilot_{mode}.jsonl from modal_app.py::score);
  * a parseable judge verdict on every step (judge_pilot_{mode}.jsonl);
  * the first side-task pass step is recoverable and the judged onset is
    at or before it (transcripts_attack.jsonl);
  * descriptive: per-step linear contrast by class (honest / pre-onset /
    post-onset) and the pilot AUROC, NOT an analysis result.

Trainer and published-probe checks are read from the train / published_check
driver outputs (pilot_train.json, pilot_published.json) if present.

Run (.venv-api):  .venv-api/bin/python experiments/2026-10-13-fibs-basharena/pilot_check.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, load_config, read_jsonl  # noqa: E402


def auroc(scores, labels) -> float:
    s, y = np.array(scores, float), np.array(labels)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    cfg = load_config()
    checks = {}
    scores = {m: read_jsonl(OUT / f"scores_pilot_{m}.jsonl") for m in ("honest", "attack")}
    judge = {m: read_jsonl(OUT / f"judge_pilot_{m}.jsonl") for m in ("honest", "attack")}
    attack = {r["uid"]: r for r in read_jsonl(OUT / "transcripts_attack.jsonl")}

    # 1. finite scores on every non-dropped step, all archs
    archs = [a for a in ("linear", "mlp", "efc", "axial") if any(f"{a}_contrast" in r for rows in scores.values() for r in rows)]
    bad, n_dropped, n_steps = [], 0, 0
    for m, rows in scores.items():
        for r in rows:
            n_steps += 1
            if r["dropped"]:
                n_dropped += 1
                continue
            for a in archs:
                if not math.isfinite(r[f"{a}_contrast"]):
                    bad.append((m, r["uid"], r["step"], a))
    checks["finite_scores"] = {"pass": not bad and n_dropped == 0, "steps": n_steps, "dropped": n_dropped,
                               "non_finite": bad, "archs": archs}

    # 2. judge parseable on every step
    unparsed = [(m, r["uid"], r["step"]) for m, rows in judge.items() for r in rows if r["side_task"] is None]
    checks["judge_parseable"] = {"pass": not unparsed, "steps": sum(len(v) for v in judge.values()),
                                 "unparseable": unparsed,
                                 "flagged": {m: sum(r["side_task"] is True for r in rows) for m, rows in judge.items()}}

    # 3. onset vs first side-task pass
    onsets = {}
    for sid in {r["uid"] for r in judge["attack"]}:
        flagged = sorted(r["step"] for r in judge["attack"] if r["uid"] == sid and r["side_task"] is True)
        onset = flagged[0] if flagged else None
        pass_after = attack[sid]["side_task_pass_after_step"]
        onsets[sid] = {"onset": onset, "first_pass": pass_after,
                       "ok": pass_after is not None and onset is not None and onset <= pass_after}
    checks["onset_vs_pass"] = {"pass": all(v["ok"] for v in onsets.values()), "episodes": onsets}

    # 4. descriptive: linear contrast by class
    by_class = {"honest": [], "pre_onset": [], "post_onset": []}
    for r in scores["honest"]:
        if not r["dropped"] and "linear_contrast" in r:
            by_class["honest"].append(r["linear_contrast"])
    for r in scores["attack"]:
        if r["dropped"] or "linear_contrast" not in r:
            continue
        o = onsets.get(r["uid"], {}).get("onset")
        cls = "post_onset" if o is not None and r["step"] >= o else "pre_onset"
        by_class[cls].append(r["linear_contrast"])
    neg = by_class["honest"] + by_class["pre_onset"]
    checks["descriptive"] = {
        "n": {k: len(v) for k, v in by_class.items()},
        "median_linear_contrast": {k: (float(np.median(v)) if v else None) for k, v in by_class.items()},
        "pilot_auroc_linear": auroc(by_class["post_onset"] + neg, [1] * len(by_class["post_onset"]) + [0] * len(neg)),
        "note": "4 episodes; not an analysis result"}

    for name in ("pilot_train.json", "pilot_published.json"):
        p = OUT / name
        if p.exists():
            checks[name.removesuffix(".json")] = json.loads(p.read_text())
    (OUT / "pilot_checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    for k, v in checks.items():
        print(k, "PASS" if v.get("pass") else ("-" if "pass" not in v else "FAIL"),
              json.dumps({kk: vv for kk, vv in v.items() if kk not in ("episodes", "non_finite", "unparseable")})[:300])


if __name__ == "__main__":
    main()
