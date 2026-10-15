"""Shared config, paths and clients for 2026-10-15: the auditor inside the
5-player game, five rounds (five interaction scenarios + passive, with and without the layer-32
follow-up probe).

Generation: the local Qwen3.8 server (serving/serve_qwen38.py, thinking off
by default; the auditor turns thinking on per request). Live probe scores:
scorer.py, an HTTP service holding the probe model on its own GPUs.
"""
from __future__ import annotations

import json
import sys
import urllib.request
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
class RunConfig(ExperimentConfig):
    model_revision: str = ""
    game_config: str = ""                # repo-relative GameConfig YAML (5 players)
    server_url: str = "http://localhost:8003/v1"
    served_model: str = "Qwen/Qwen3.8-27B"
    scorer_url: str = "http://localhost:8010"
    request_timeout_s: int = 1800
    max_concurrent_games: int = 20
    player_attempts: int = 3             # engine: up to 3 tries for an invalid tool call
    auditor_max_tokens: int = 12000      # thinking + answer, per auditor call
    auditor_reasoning_effort: str = "medium"   # Qwen3.8 template: xhigh (default) | medium | low;
                                               # at xhigh the text-only auditor overran 12k tokens (pilot)
    auditor_attempts: int = 2            # a malformed auditor reply gets one corrective retry
    answer_max_tokens: int = 1000        # a player's reply to an auditor question
    budget_questions: int = 5            # S3: questions for the whole game
    n_games: int = 50                    # per game set; game i uses seed game_seed + i
    game_seed: int = 5000
    pilot_seed: int = 0
    n_boot: int = 2000


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


def probe_score(cfg: RunConfig, dialogue: list[dict]) -> float:
    """Layer-48 follow-up-probe log-odds for a dialogue ending with the
    speaker's reply (scorer.py appends the follow-up question and "No")."""
    return _post(f"{cfg.scorer_url}/score", {"dialogue": dialogue}, cfg.request_timeout_s)["score"]
