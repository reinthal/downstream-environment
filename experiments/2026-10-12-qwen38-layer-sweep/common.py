"""Shared config, paths and the Qwen3.8 server client for the 2026-10-12
follow-up-probe layer sweep.

All generation goes through the local OpenAI-compatible vLLM server
(`serving/serve_qwen38.py`, started with thinking off by default — see
README). Prompts that need the 2026-10-01 raw-completion recipe (answer
prefix, forced close / score) are rendered client-side with the model's own
chat template and sent to /v1/completions.
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
VENDOR = REPO / "vendor" / "deception-detection"
for p in (REPO, ENGINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from linear_probes.config import ExperimentConfig  # noqa: E402


@dataclass
class SweepConfig(ExperimentConfig):
    model_revision: str = ""                    # HF revision, same as the server's
    layers: list[int] = field(default_factory=list)   # probe read-out layers (block outputs)
    server_url: str = "http://localhost:8003/v1"
    served_model: str = "Qwen/Qwen3.8-27B"
    workers: int = 32                           # concurrent requests to the server
    request_timeout_s: int = 900
    # grading (2026-10-01 local grader: the policy model, thinking on, temperature 0)
    grader_max_tokens: int = 2000
    # games (engine GameConfig YAML; game i uses seed game_seed + i)
    game_config: str = ""
    n_games: int = 100
    game_seed: int = 1000
    pilot_seed: int = 0
    # analysis
    n_random: int = 100                         # random directions per layer
    n_boot: int = 2000


def load_config() -> SweepConfig:
    return SweepConfig.load(OUT / "config.json")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def tokenizer(cfg: SweepConfig):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)


def render(tok, messages: list[dict], thinking: bool) -> str:
    """Chat-template text ending at the assistant generation prompt."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)


def _post(cfg: SweepConfig, path: str, body: dict) -> dict:
    req = urllib.request.Request(f"{cfg.server_url}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=cfg.request_timeout_s) as resp:
        return json.loads(resp.read())


def complete(cfg: SweepConfig, prompts: list[str], max_tokens: int, *, temperature: float,
             top_p: float, seed: int) -> list[dict]:
    """Raw continuations of rendered prompts: [{"text", "finish_reason"}] in
    input order; prompt i is sampled with seed `seed + i`."""
    def one(i: int) -> dict:
        r = _post(cfg, "/completions", {
            "model": cfg.served_model, "prompt": prompts[i], "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "seed": seed + i,
            "skip_special_tokens": False})
        (choice,) = r["choices"]
        return {"text": choice["text"], "finish_reason": choice["finish_reason"]}
    with ThreadPoolExecutor(cfg.workers) as ex:
        return list(ex.map(one, range(len(prompts))))


def chat(cfg: SweepConfig, conversations: list[list[dict]], *, tools: list[dict] | None = None,
         max_tokens: int) -> list[dict]:
    """Chat completions with the server's default sampling and thinking off
    (the engine sends neither temperature nor top_p). Returns each choice's
    message dict, in input order."""
    def one(msgs: list[dict]) -> dict:
        body = {"model": cfg.served_model, "messages": msgs, "max_tokens": max_tokens}
        if tools is not None:
            body.update(tools=tools, tool_choice="auto")
        (choice,) = _post(cfg, "/chat/completions", body)["choices"]
        return {**choice["message"], "finish_reason": choice["finish_reason"]}
    with ThreadPoolExecutor(cfg.workers) as ex:
        return list(ex.map(one, conversations))
