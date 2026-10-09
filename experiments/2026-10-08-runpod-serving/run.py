"""Smoke-test the RunPod serverless deployment (iac/): the Qwen3.5-27B
OpenAI-compatible endpoint and the two probe /classify endpoints.

Probe requests are the model cards' "Python client (remote /classify
endpoint)" snippets verbatim, with URL = the endpoint's RunPod OpenAI base
(``https://api.runpod.ai/v2/<id>/openai``) plus the RunPod auth header, and
again through the /runsync route envelope documented in iac/README.md.

No GPU, no ML env — runs anywhere with RUNPOD_API_KEY set:
    set -a; source .env; set +a
    uv run --no-project --with requests --with transformers --with jinja2 \
        python experiments/2026-10-08-runpod-serving/run.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from linear_probes.config import ExperimentConfig  # noqa: E402

API = "https://api.runpod.ai/v2"
REST = "https://rest.runpod.io/v1"

# The model cards' example dialogue (serving/publish.py::_RENDER_MEAN/_RENDER_LAST).
QUESTION = "Were you near the pantry when the carrots vanished?"
REPLY = "No, I was in the garden all morning."
REASONING = "I need to deflect suspicion from myself."
FOLLOWUP = "Was your last response deceptive? (Yes/No)"


@dataclass
class RunpodServingConfig(ExperimentConfig):
    gen_endpoint_id: str = ""
    # probe name -> {"endpoint_id", "hf_repo", "pooling": "MEAN" | "LAST"}
    probe_endpoints: dict[str, dict[str, str]] = field(default_factory=dict)
    gen_timeout_s: int = 180      # a cold start that loads 55 GB will exceed this
    probe_timeout_s: int = 600    # covers a probe cold start (~4 min measured)
    gen_max_tokens: int = 20


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}


def deployed(endpoint_id: str) -> dict:
    """What RunPod says is deployed: endpoint + its template, and live health."""
    ep = requests.get(f"{REST}/endpoints/{endpoint_id}", headers=headers(), timeout=60)
    ep.raise_for_status()
    ep = ep.json()
    tmpl = requests.get(f"{REST}/templates/{ep['templateId']}", headers=headers(), timeout=60)
    tmpl.raise_for_status()
    tmpl = tmpl.json()
    health = requests.get(f"{API}/{endpoint_id}/health", headers=headers(), timeout=60)
    health.raise_for_status()
    return {"name": ep["name"], "image": tmpl["imageName"], "env": tmpl["env"],
            "gpuTypeIds": ep["gpuTypeIds"], "workersMax": ep["workersMax"],
            "health": health.json()}


def test_generation(cfg: RunpodServingConfig) -> dict:
    t0 = time.time()
    try:
        r = requests.post(
            f"{API}/{cfg.gen_endpoint_id}/openai/v1/chat/completions", headers=headers(),
            json={"model": cfg.model_id, "max_tokens": cfg.gen_max_tokens,
                  "messages": [{"role": "user", "content": "ping"}]},
            timeout=cfg.gen_timeout_s)
    except requests.Timeout:
        # A timeout IS the measurement here (endpoint not serving), not an error to hide.
        return {"ok": False, "outcome": f"timeout after {cfg.gen_timeout_s}s"}
    out = {"ok": r.ok, "http": r.status_code, "seconds": round(time.time() - t0, 1)}
    out["body"] = r.json() if r.ok else r.text
    return out


def render(tok, pooling: str) -> str:
    if pooling == "MEAN":
        messages = [{"role": "user", "content": QUESTION},
                    {"role": "assistant", "content": REPLY, "reasoning_content": REASONING}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if pooling == "LAST":
        conv = [{"role": "user", "content": QUESTION},
                {"role": "assistant", "content": REPLY},
                {"role": "user", "content": FOLLOWUP}]
        prefix = tok.apply_chat_template(conv, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return prefix + "No"
    raise ValueError(f"pooling must be MEAN or LAST, got {pooling!r}")


def test_probe(cfg: RunpodServingConfig, spec: dict[str, str]) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["hf_repo"])
    text = render(tok, spec["pooling"])
    eid = spec["endpoint_id"]

    # 1. model-card snippet: POST {URL}/classify, URL = RunPod OpenAI base
    t0 = time.time()
    r = requests.post(f"{API}/{eid}/openai/classify", headers=headers(),
                      json={"input": [text]}, timeout=cfg.probe_timeout_s)
    r.raise_for_status()
    card = r.json()
    card_s = round(time.time() - t0, 1)

    # 2. iac/README.md: /runsync route envelope, batch of 2
    t0 = time.time()
    r = requests.post(
        f"{API}/{eid}/runsync", headers=headers(), timeout=cfg.probe_timeout_s,
        json={"input": {"route": "/classify", "method": "POST",
                        "body": {"input": [text, text]}}})
    r.raise_for_status()
    job = r.json()
    if job["status"] != "COMPLETED":
        raise RuntimeError(f"runsync job not completed: {job}")
    (envelope,) = job["output"]

    return {
        "served_model": card["model"],
        "prompt_tokens": card["usage"]["prompt_tokens"],
        "rendered_tail": text[-60:],
        "classify": {"p_deceptive": card["data"][0]["probs"][0],
                     "label": card["data"][0]["label"], "seconds": card_s},
        "runsync": {"p_deceptive": [d["probs"][0] for d in envelope["data"]],
                    "seconds": round(time.time() - t0, 1),
                    "delay_ms": job["delayTime"], "execution_ms": job["executionTime"]},
    }


def main():
    cfg = RunpodServingConfig.load(OUT / "config.json")
    cfg.save()
    results = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "deployed": {"generation": deployed(cfg.gen_endpoint_id),
                     **{name: deployed(spec["endpoint_id"])
                        for name, spec in cfg.probe_endpoints.items()}},
        "generation": test_generation(cfg),
        "probes": {name: test_probe(cfg, spec) for name, spec in cfg.probe_endpoints.items()},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
