"""Shared pieces for the Carrot-Parsnip auditor-win-rate experiment.

One model (Qwen/Qwen3.5-27B) plays all five agents and the auditor. Games are
discussion-only: 3 rounds of public announcements, no votes — the game stops
before the elimination phase and an outside auditor names the Parsnip instead.

Generation goes through ``rollouts.generation`` (vLLM for bulk rollouts, HF
transformers when steering); this module keeps the experiment constants and
the probe-side helpers (span extraction + activation collection).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
VENDOR = REPO / "vendor" / "deception-detection"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rollouts.generation import load_hf_model, split_think  # noqa: E402,F401  (re-export)
from linear_probes.activations import (collect_activations,  # noqa: E402,F401  (moved there;
                                       encode_with_spans)    #  re-exported for the scripts here)

MODEL_ID = "Qwen/Qwen3.5-27B"
PLAYER_NAMES = ["Alice", "Bob", "Charlie", "Dave", "Eve"]
NUM_ROUNDS = 3
SEED_GROUPS = 5
GAMES_PER_SEED = 20

# probe read-out layers considered during training; the winner is stored in the
# probe npz and every later stage reads it from there
CANDIDATE_LAYERS = [16, 24, 32, 40, 48]

MAX_NEW_TOKENS = 1800        # think block + short announcement
AUDITOR_MAX_NEW_TOKENS = 2000


def game_seed(seed_group: int, game_idx: int) -> int:
    return 1000 * seed_group + game_idx


def load_model():
    """(model, tokenizer) for activation collection — HF transformers env."""
    return load_hf_model(MODEL_ID)


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from the repo-root ``.env`` (gitignored — API keys
    live there, never in the repo) into ``os.environ``. Already-set variables
    win; ``export`` prefixes and quotes are tolerated so the file can also be
    ``source``d."""
    import os

    path = path or REPO / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


# ── transcript I/O ───────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: list[dict]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
