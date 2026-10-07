"""Shared pieces for the follow-up-question ("did you lie") probe, after
Goldowsky-Dill et al. (2025). The follow-up machinery was lifted into
`linear_probes.dyl_probe` (DYLProbe) — this module re-exports it under the
names the scripts here were written against.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
VENDOR = REPO / "vendor" / "deception-detection"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from linear_probes.dyl_probe import (  # noqa: E402
    FOLLOWUP_ANSWER as FOLLOWUP_A,
    FOLLOWUP_QUESTION as FOLLOWUP_Q,
    encode_no_span,  # noqa: F401
)
from linear_probes.dyl_probe import with_followup as _with_followup  # noqa: E402

# RolePlayingDataset.system_postfix
ROLEPLAY_POSTFIX = "Answer the question directly, without asides or internal thoughts."


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def roleplay_scenarios() -> list[dict]:
    import yaml
    with open(VENDOR / "data" / "roleplaying" / "dataset.yaml") as f:
        return yaml.safe_load(f)


def roleplay_prompt(s: dict) -> list[dict]:
    """System + user; the reply is generated after the answer prefix."""
    return [{"role": "system", "content": s["scenario"] + "\n\n" + ROLEPLAY_POSTFIX},
            {"role": "user", "content": s["question"]}]


def with_followup(messages: list[dict], reply: str) -> list[dict]:
    """A finished dialogue (prompt messages + the model's public reply) plus
    the follow-up question and the fixed "No"."""
    return _with_followup([*messages, {"role": "assistant", "content": reply}])
