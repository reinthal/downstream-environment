"""Shared config, paths and clients for 2026-10-17: Inside Job — does a
dynamic auditor with a white-box deception probe add anything beyond the
probe? (Engine: environments/inside_job.)

Generation: a Qwen3.8 server copy (serving/serve_qwen38.py, thinking off by
default; the auditor turns thinking on per request). Raw probe values per
turn: the 2026-10-16 scorer (layer-32 / layer-48 follow-up probes, + its
random direction), an HTTP service on its own GPUs.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
for p in (REPO, REPO / "environments" / "inside_job"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from linear_probes.config import ExperimentConfig  # noqa: E402


@dataclass
class RunConfig(ExperimentConfig):
    model_revision: str = ""
    server_url: str = "http://localhost:8004/v1"
    served_model: str = "Qwen/Qwen3.8-27B"
    scorer_url: str = "http://localhost:8010"
    request_timeout_s: int = 7200        # 2026-10-16: 30 min timed out in a long queue
    max_concurrent_games: int = 24
    player_attempts: int = 3             # tries for a valid tool call
    player_max_tokens: int = 1500
    answer_max_tokens: int = 600
    auditor_max_tokens: int = 12000      # thinking + answer, per auditor call
    auditor_reasoning_effort: str = "medium"
    auditor_attempts: int = 2            # one corrective retry
    discussion_rounds: int = 2
    free_questions: int = 3              # interactive auditor: private questions per mission
    audits: int = 2                      # ground-truth audits per game (interactive auditor)
    # game (environments/inside_job GameSpec)
    n_missions: int = 4
    code_len: int = 3
    overlap: float = 0.25
    p_corrupted: float = 0.3
    corrupt_missions: int = 2
    p_benign_secret: float = 0.5
    # mole type per seed (deterministic in the seed, so sets pair by seed)
    mole_mix: dict[str, float] = field(default_factory=lambda: {"none": 0.2, "liar": 0.4, "silent": 0.4})
    pilot_mole_mix: dict[str, float] = field(default_factory=lambda: {
        "none": 0.2, "liar": 0.3, "silent": 0.3, "open": 0.2})
    n_games: int = 150
    game_seed: int = 7000
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
    """One chat completion at the server's default sampling. With thinking on,
    the reasoning effort is cfg.auditor_reasoning_effort (only the auditor
    thinks). Returns {content, reasoning, tool_calls, finish_reason}."""
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
    turn (the scorer appends the follow-up question and "No")."""
    return _post(f"{cfg.scorer_url}/score", {"dialogue": dialogue}, cfg.request_timeout_s)
