"""Shared config and paths for the 2026-10-09 3-player game experiment."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
ENGINE = REPO / "environments" / "carrot_parsnip"
for p in (REPO, ENGINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from linear_probes.config import ExperimentConfig  # noqa: E402


@dataclass
class GameRunConfig(ExperimentConfig):
    game_config: str = ""            # repo-relative GameConfig YAML (the game's own hyperparameters)
    n_games: int = 100               # game i uses seed `seed + i`
    pilot_seed: int = 0              # pilot games use `pilot_seed + i`; never analysed
    # RunPod serverless endpoint serving the L18 follow-up (DYL) probe
    probe_endpoint_id: str = ""
    probe_hf_repo: str = ""          # tokenizer + chat template for client-side rendering
    probe_timeout_s: int = 900       # covers a cold start (~4 min measured 2026-10-08)
    probe_batch: int = 16            # dialogues per /classify request
    # analysis (preregistered): the pick rate counts as unbiased when its whole
    # 95% interval lies within chance +- bias_band; runs with more than
    # void_fraction of games or audits excluded are void
    bias_band: float = 0.10
    void_fraction: float = 0.10
    n_boot: int = 2000


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
