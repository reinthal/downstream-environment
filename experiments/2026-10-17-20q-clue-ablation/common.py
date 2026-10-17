"""Shared config for the clue ablation. Runs in the twenty_questions env:
    uv run --project environments/twenty_questions python experiments/2026-10-17-20q-clue-ablation/run.py
(no probes in this experiment, so no linear_probes ExperimentConfig: it
imports torch, which this env does not carry.)"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
ENV = REPO / "environments" / "twenty_questions"


@dataclass
class AblationConfig:
    # model ladder, in this fixed order (figure x-axis); OpenRouter ids
    models: list = field(default_factory=lambda: [
        "openrouter/qwen/qwen3.5-9b",
        "openrouter/qwen/qwen3.5-27b",
        "openrouter/qwen/qwen3.5-35b-a3b",
        "openrouter/qwen/qwen3.5-122b-a10b",
        "openrouter/qwen/qwen3.5-397b-a17b",
        "openrouter/z-ai/glm-5.3-flash",
        "openrouter/z-ai/glm-5.3",
    ])
    # arm -> GameConfig overrides
    arms: dict = field(default_factory=lambda: {
        "clues": {"clues": True, "clue_requests": True},
        "no_clues": {"clues": False, "clue_requests": False},
    })
    base_game_config: str = "environments/twenty_questions/configs/default.yaml"
    max_questions: int = 12
    concept_set: str = "concepts/things_n512_s0.json"
    objective: str = "competitive"
    reasoning_effort: str | None = None     # provider default
    max_tokens: int = 16000                 # per call, every seat and the judge
    n_games: int = 10                       # per model x arm; seeds base_seed.. -> same secrets everywhere
    base_seed: int = 0
    pilot_seed: int = 1000
    pilot_games: int = 1
    max_concurrent_games: int = 10
    parallel_models: int = 7                # models run concurrently, arms sequentially within a model
    cost_cap_usd: float = 50.0
    stop_launching_at: float = 0.8          # fraction of the cap after which no new cell starts
    retry_errored_once: bool = True
    void_excluded_fraction: float = 0.2     # a cell with more excluded games is void
    too_easy_threshold: float = 3.9         # clue-arm mean end turn below this = "too easy"

    def save(self) -> Path:
        path = OUT / "config.json"
        path.write_text(json.dumps(asdict(self), indent=1) + "\n")
        return path

    @classmethod
    def load(cls) -> "AblationConfig":
        raw = json.loads((OUT / "config.json").read_text())
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config fields {sorted(unknown)}")
        return cls(**raw)


def slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "_")


def cell(model: str, arm: str, pilot: bool) -> str:
    return f"{'pilot_' if pilot else ''}{slug(model)}__{arm}"
