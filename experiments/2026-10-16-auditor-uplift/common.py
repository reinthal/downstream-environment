"""Shared config, paths and clients for 2026-10-16: auditor uplift over the
detector it is given — the 5-player game, five rounds, S2 / S4 with six
detectors of controlled quality, plus a passive (S0) replay over a quality
ladder.

Generation: the local Qwen3.8 server (serving/serve_qwen38.py, thinking off
by default; the auditor turns thinking on per request). Live raw probe
values: scorer.py, an HTTP service holding Qwen3.8 truncated at layer 48 on
its own GPUs. Detector scores shown to the auditor: detectors.py, from
calibration.json (calibrate.py).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
ENGINE = REPO / "environments" / "carrot_parsnip"
for p in (REPO, ENGINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from linear_probes.config import ExperimentConfig  # noqa: E402


@dataclass
class RunConfig(ExperimentConfig):
    model_revision: str = ""
    game_config: str = ""                # repo-relative GameConfig YAML (5 players, 5 rounds)
    server_url: str = "http://localhost:8003/v1"
    served_model: str = "Qwen/Qwen3.8-27B"
    scorer_url: str = "http://localhost:8010"
    request_timeout_s: int = 1800
    max_concurrent_games: int = 20
    player_attempts: int = 3             # engine: up to 3 tries for an invalid tool call
    auditor_max_tokens: int = 12000      # thinking + answer, per auditor call
    auditor_reasoning_effort: str = "medium"
    auditor_attempts: int = 2            # a malformed auditor reply gets one corrective retry
    answer_max_tokens: int = 1000        # a player's reply to an auditor question
    budget_questions: int = 5            # unused here (S3 not run); kept for game.py
    n_games: int = 50                    # per game set; game i uses seed game_seed + i
    game_seed: int = 6000                # = 2026-10-15, so its text-only S2 / S4 games pair by seed
    pilot_seed: int = 0
    n_boot: int = 2000
    # probes and detectors
    layers: list[int] = field(default_factory=lambda: [32, 48])
    scenarios: list[str] = field(default_factory=lambda: ["S2", "S4"])
    detectors: list[str] = field(default_factory=lambda: ["l32", "l48", "rand", "mix", "oracle", "inv"])
    passive_detectors: list[str] = field(default_factory=lambda: [
        "none", "l32", "l48", "rand", "mix30", "mix", "mix70", "mix90", "oracle", "inv"])
    mix_targets: dict[str, float] = field(default_factory=lambda: {
        "mix30": 0.3, "mix": 0.5, "mix70": 0.7, "mix90": 0.9})   # probe-alone top-1 on calibration games
    n_random_candidates: int = 200
    random_seed: int = 16
    # calibration (2026-10-15 text-only games, never the evaluation games) and passive replay source
    source_dir: str = "experiments/2026-10-15-auditor-l32-5rounds"
    calib_sets: list[str] = field(default_factory=lambda: ["S2_none", "S4_none"])
    calib_games: int = 50                # first N games of each calibration set
    passive_runs: int = 3                # auditor runs per (game, detector) in the replay
    interactive_games: int = 50          # seeds analysed per interactive cell (32 after the 2026-10-19 stop)
    collect_shards: int = 3


def load_config() -> RunConfig:
    return RunConfig.load(OUT / "config.json")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _post(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(cfg: RunConfig, messages: list[dict], *, max_tokens: int, tools: list[dict] | None = None,
         thinking: bool = False) -> dict:
    """One chat completion at the server's default sampling (as the engine
    sends no temperature). With thinking on, the reasoning effort is
    cfg.auditor_reasoning_effort (only the auditor thinks). Returns
    {content, reasoning, tool_calls, finish_reason}."""
    kwargs = {"enable_thinking": thinking}
    if thinking:
        kwargs["reasoning_effort"] = cfg.auditor_reasoning_effort
    body = {"model": cfg.served_model, "messages": messages, "max_tokens": max_tokens,
            "chat_template_kwargs": kwargs}
    if tools is not None:
        body.update(tools=tools, tool_choice="auto")
    (choice,) = _post(f"{cfg.server_url}/chat/completions", body, cfg.request_timeout_s)["choices"]
    m = choice["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
            "tool_calls": m.get("tool_calls") or [], "finish_reason": choice["finish_reason"]}


def probe_raw(cfg: RunConfig, dialogue: list[dict]) -> dict:
    """Raw values {l32, l48, rand} for a dialogue ending with the speaker's
    reply (scorer.py appends the follow-up question and "No")."""
    return _post(f"{cfg.scorer_url}/score", {"dialogue": dialogue}, cfg.request_timeout_s)
