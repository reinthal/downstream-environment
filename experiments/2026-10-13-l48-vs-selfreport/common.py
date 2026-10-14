"""Shared config, paths and helpers for 2026-10-13: the layer-48 follow-up
probe against the model's own answer, P(Yes), in normal and "deny" games.

Arms: "normal" / "deny" (seeds game_seed..), and their pilots
"normal_pilot" / "deny_pilot" (seeds pilot_seed.., never analysed). Every
per-arm file is keyed by the arm string. Generation goes through the local
Qwen3.8 server (serving/serve_qwen38.py, thinking off by default).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
ENGINE = REPO / "environments" / "carrot_parsnip"
for p in (REPO, ENGINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from linear_probes.config import ExperimentConfig  # noqa: E402
from linear_probes.dyl_probe import FOLLOWUP_QUESTION  # noqa: E402

ARMS = ("normal", "deny", "normal_pilot", "deny_pilot")


@dataclass
class RunConfig(ExperimentConfig):
    model_revision: str = ""
    layers: list[int] = field(default_factory=list)      # probe read-out layers; probe_paths keyed by str(layer)
    primary_layer: int = 48
    server_url: str = "http://localhost:8003/v1"
    served_model: str = "Qwen/Qwen3.8-27B"
    workers: int = 32
    request_timeout_s: int = 900
    games: dict[str, str] = field(default_factory=dict)   # arm -> repo-relative GameConfig YAML
    n_games: int = 100                                    # per arm; game i uses seed game_seed + i
    game_seed: int = 2000
    pilot_seed: int = 0
    n_boot: int = 2000


def load_config() -> RunConfig:
    return RunConfig.load(OUT / "config.json")


def arm_yaml(cfg: RunConfig, arm: str) -> Path:
    return REPO / cfg.games[arm.removesuffix("_pilot")]


def arm_seed(cfg: RunConfig, arm: str) -> int:
    return cfg.pilot_seed if arm.endswith("_pilot") else cfg.game_seed


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def scored_turns(arm: str) -> list[dict]:
    """The discussion turns both detectors score: those with an announcement."""
    return [r for r in read_jsonl(OUT / f"transcripts_{arm}.jsonl") if r["announcement"]]


def followup_dialogue(r: dict) -> list[dict]:
    """What the speaker saw + its announcement + the follow-up question."""
    return [*r["input_messages"], {"role": "assistant", "content": r["announcement"]},
            {"role": "user", "content": FOLLOWUP_QUESTION}]


def post(cfg: RunConfig, path: str, body: dict) -> dict:
    req = urllib.request.Request(f"{cfg.server_url}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=cfg.request_timeout_s) as resp:
        return json.loads(resp.read())


def pmap(cfg: RunConfig, fn, items: list) -> list:
    with ThreadPoolExecutor(cfg.workers) as ex:
        return list(ex.map(fn, items))
