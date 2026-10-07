"""Publish a probe as a servable HF classification model.

Takes a trained :class:`linear_probes.probes.LinearProbe` (npz), its layer and
the base HF model id, and builds a checkpoint containing only decoder blocks
``0..layer`` plus one fp32 ``score`` Linear whose weights are the probe with
its standardizer folded in:

    score.weight = w / sd            (1, d)
    score.bias   = b - (mu/sd) @ w   (1,)

so ``sigmoid(x @ score.weight.T + score.bias) == probe.predict_proba(x)``
exactly, where ``x`` is the mean over ALL processed tokens of the raw
(pre-final-norm) block-``layer`` residual stream. Span selection is the
client's job: the served model scores whatever text it is sent.

The checkpoint is served by vLLM through the in-repo plugin
(serving/vllm_plugin), which registers ``Qwen3_5ProbeForSequenceClassification``
— the plugin replaces the final RMSNorm with a residual-add passthrough so the
pooled states match the activations the probe was trained on.

Run (in .venv):
    uv run --no-sync python -m serving.publish \
        --config experiments/<date>/config.json [--local-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np

from linear_probes.probes import LinearProbe
from serving.config import ServingConfig

REPO = Path(__file__).resolve().parents[1]

ARCHITECTURE = "Qwen3_5ProbeForSequenceClassification"

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "chat_template.jinja")


# ── pure transforms (CPU-tested in tests/test_probe_publish.py) ─────────────

def fold_probe_head(probe: LinearProbe) -> tuple[np.ndarray, np.ndarray]:
    """([1, d] fp32 weight, [1] fp32 bias) such that
    ``sigmoid(X @ W.T + b) == probe.predict_proba(X)``."""
    w = np.asarray(probe.w, dtype=np.float64) / np.asarray(probe.sd, dtype=np.float64)
    bias = float(probe.b) - float(np.asarray(probe.mu, dtype=np.float64) @ w)
    return w[None, :].astype(np.float32), np.array([bias], dtype=np.float32)


def remap_weight_name(name: str, layer: int) -> str | None:
    """Published tensor name for a base-checkpoint tensor, or None to drop it.

    Handles the Qwen3.5 VLM layout (``model.language_model.*``) and text-only
    layouts (``model.*``). Raises on names it was not written for — extend it
    deliberately for a new model family instead of silently dropping weights.
    """
    if name == "lm_head.weight" or name.startswith("mtp."):
        return None
    for pref in ("model.language_model.", "model."):
        if name.startswith(pref):
            core = name[len(pref):]
            break
    else:
        raise ValueError(f"unexpected tensor name {name!r}")
    if core.startswith("visual."):
        return None
    if core == "norm.weight":
        return None   # final norm — replaced by a residual-add passthrough at serve time
    if core == "embed_tokens.weight":
        return "model.embed_tokens.weight"
    m = re.match(r"layers\.(\d+)\.", core)
    if m:
        return f"model.{core}" if int(m.group(1)) <= layer else None
    raise ValueError(f"unexpected tensor name {name!r}")


def build_config(src: dict, layer: int, probe_meta: dict, pooling: str = "MEAN") -> dict:
    """Text-only classifier config from the base model's config.json dict."""
    if pooling not in ("MEAN", "LAST"):
        raise ValueError(f"pooling must be MEAN or LAST, got {pooling!r}")
    text = dict(src["text_config"]) if "text_config" in src else dict(src)
    n = layer + 1
    if text["num_hidden_layers"] < n:
        raise ValueError(f"layer {layer} out of range for "
                         f"num_hidden_layers={text['num_hidden_layers']}")
    text["num_hidden_layers"] = n
    text["layer_types"] = list(text["layer_types"])[:n]
    text["mtp_num_hidden_layers"] = 0
    # Text-only model: mrope reduces to standard RoPE (all three channels
    # equal the text position), but its presence makes vLLM's runner demand
    # M-RoPE support from the model class — drop it.
    if "rope_parameters" in text:
        rope = dict(text["rope_parameters"])
        rope.pop("mrope_section", None)
        rope.pop("mrope_interleaved", None)
        text["rope_parameters"] = rope
    text["architectures"] = [ARCHITECTURE]
    text["id2label"] = {"0": "deceptive"}
    text["label2id"] = {"deceptive": 0}
    text["probe_pooling"] = pooling
    text["probe_meta"] = probe_meta
    return text


# ── checkpoint streaming (never more than one source shard in RAM) ──────────

def select_weights(snapshot: Path, layer: int) -> Iterator[tuple[str, "torch.Tensor"]]:
    """Stream the wanted tensors out of the base checkpoint's shards."""
    from safetensors import safe_open

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(name)
    for shard in sorted(by_shard):
        with safe_open(snapshot / shard, framework="pt", device="cpu") as f:
            for name in by_shard[shard]:
                new = remap_weight_name(name, layer)
                if new is not None:
                    yield new, f.get_tensor(name)


def write_sharded(out_dir: Path, weights: Iterator, max_shard_bytes: int = 4 << 30) -> int:
    """model-0000k-of-0000n.safetensors + index. Returns total bytes."""
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    paths: list[Path] = []
    shard: dict = {}
    shard_bytes = 0
    total = 0

    def flush():
        nonlocal shard, shard_bytes
        if not shard:
            return
        path = out_dir / f"model-{len(paths) + 1:05d}.safetensors"
        save_file(shard, str(path), metadata={"format": "pt"})
        paths.append(path)
        for k in shard:
            weight_map[k] = path.name
        shard, shard_bytes = {}, 0

    for name, t in weights:
        t = t.contiguous()
        nb = t.numel() * t.element_size()
        if shard and shard_bytes + nb > max_shard_bytes:
            flush()
        shard[name] = t
        shard_bytes += nb
        total += nb
    flush()

    n = len(paths)
    renames = {p.name: f"model-{i + 1:05d}-of-{n:05d}.safetensors"
               for i, p in enumerate(paths)}
    for p in paths:
        p.rename(out_dir / renames[p.name])
    index = {"metadata": {"total_size": total},
             "weight_map": {k: renames[v] for k, v in weight_map.items()}}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=1) + "\n")
    return total


# ── model card ───────────────────────────────────────────────────────────────

_RENDER_MEAN = '''messages = [
    {"role": "user", "content": "Were you near the pantry when the carrots vanished?"},
    {"role": "assistant",
     "content": "No, I was in the garden all morning.",
     "reasoning_content": "I need to deflect suspicion from myself."},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)'''

_RENDER_LAST = '''messages = [   # finished dialogue, last turn = the assistant's public reply
    {"role": "user", "content": "Were you near the pantry when the carrots vanished?"},
    {"role": "assistant", "content": "No, I was in the garden all morning."},
]
conv = messages + [{"role": "user", "content": "Was your last response deceptive? (Yes/No)"},
                   {"role": "assistant", "content": "No"}]
prefix = tok.apply_chat_template(conv[:-1], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
text = prefix + "No"   # MUST end exactly at the "No" — the probe reads its token'''

_CONTRACT_MEAN = """- The probe reads the **raw (pre-final-norm) residual stream** after block
  {layer}, **mean-pooled over ALL tokens** of the submitted prompt.
- The client decides what text to score. To reproduce training conditions,
  render a finished conversation with the bundled chat template
  (`add_generation_prompt=False`) and submit the full rendered string — the
  score then covers the whole conversation."""

_CONTRACT_LAST = """- This is a follow-up ("did you lie") probe: it reads the **raw
  (pre-final-norm) residual stream** after block {layer} at the **LAST token**
  of the submitted prompt.
- The client appends `user: Was your last response deceptive? (Yes/No)` and
  the fixed answer `No` to a finished dialogue, renders with the bundled chat
  template, and sends text ending **exactly at the "No"** (see the snippet
  below) — the probe reads that token's activation."""


def model_card(cfg: ServingConfig, probe_meta: dict) -> str:
    contract = (_CONTRACT_LAST if cfg.pooling == "LAST" else _CONTRACT_MEAN).format(layer=cfg.layer)
    render = _RENDER_LAST if cfg.pooling == "LAST" else _RENDER_MEAN
    return f"""---
base_model: {cfg.model_id}
pipeline_tag: text-classification
tags:
- vllm
- linear-probe
- interpretability
- deception-detection
---

# Deception-probe read-out of {cfg.model_id} (layer {cfg.layer})

{cfg.model_id} truncated to decoder blocks 0..{cfg.layer} with a single-logit
`score` head whose weights are a trained linear deception probe
({probe_meta["kind"]}, standardizer folded in). Given a prompt, the model
returns `p(deceptive)` = sigmoid(probe logit) instead of generating text.

## Scoring contract

{contract}
- The checkpoint's hidden states are unnormalized by design; only the
  classification output is meaningful. There is no `lm_head` — this model
  cannot generate.

## Serving (vLLM)

This architecture (`{ARCHITECTURE}`) is out-of-tree: install the plugin from
the source repo first (it replaces the final RMSNorm with a residual-add
passthrough and registers the model class):

```bash
uv pip install --python <venv>/bin/python -e serving/vllm_plugin
vllm serve {cfg.hf_repo_id or "<repo>"} --runner pooling --max-model-len {cfg.max_model_len}
curl -s localhost:8000/classify -H 'Content-Type: application/json' \\
    -d '{{"input": ["<rendered conversation>"]}}'
```

## Python (local vLLM, offline)

Same plugin requirement as serving; then:

```python
from transformers import AutoTokenizer
from vllm import LLM

REPO = "{cfg.hf_repo_id or "<repo>"}"
tok = AutoTokenizer.from_pretrained(REPO)
llm = LLM(model=REPO, runner="pooling", enforce_eager=True, max_model_len={cfg.max_model_len})

{render}

(out,) = llm.classify([text])
p_deceptive = out.outputs.probs[0]
```

## Python client (remote /classify endpoint)

The endpoint does NOT apply the chat template — render the conversation
client-side with this repo's tokenizer, then send the string:

```python
import requests
from transformers import AutoTokenizer

REPO = "{cfg.hf_repo_id or "<repo>"}"
URL = "https://<your-vllm-server>"   # host running `vllm serve REPO --runner pooling`

tok = AutoTokenizer.from_pretrained(REPO)
{render}

resp = requests.post(f"{{URL}}/classify", json={{"input": [text]}})
p_deceptive = resp.json()["data"][0]["probs"][0]
```

Batch by passing multiple strings in `input`.

## Provenance

```json
{json.dumps(probe_meta, indent=2)}
```
"""


# ── publish ──────────────────────────────────────────────────────────────────

def build(cfg: ServingConfig) -> Path:
    """Build the checkpoint into ``cfg.publish_dir``. Returns the dir."""
    import torch
    from huggingface_hub import snapshot_download

    probe_path = REPO / cfg.probe_paths[cfg.probe_name]
    probe = LinearProbe.load(str(probe_path))
    if probe.config.model_id != cfg.model_id:
        raise ValueError(f"probe was trained on {probe.config.model_id!r}, "
                         f"config says {cfg.model_id!r}")
    if probe.config.layer != cfg.layer:
        raise ValueError(f"probe reads layer {probe.config.layer}, config says {cfg.layer}")

    snapshot = Path(snapshot_download(cfg.model_id))
    src_cfg = json.loads((snapshot / "config.json").read_text())
    hidden = (src_cfg.get("text_config") or src_cfg)["hidden_size"]
    if len(probe.w) != hidden:
        raise ValueError(f"probe dim {len(probe.w)} != hidden_size {hidden}")

    probe_meta = {
        "kind": probe.kind,
        "layer": cfg.layer,
        "base_model_id": cfg.model_id,
        "probe_npz": str(probe_path.resolve().relative_to(REPO)),
        "probe_npz_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "published": date.today().isoformat(),
    }

    out = REPO / cfg.publish_dir
    if out.exists():
        shutil.rmtree(out)   # build artifact dir — always rebuilt from scratch
    out.mkdir(parents=True)

    weight, bias = fold_probe_head(probe)
    extra = [("score.weight", torch.from_numpy(weight)),
             ("score.bias", torch.from_numpy(bias))]

    import itertools
    total = write_sharded(out, itertools.chain(select_weights(snapshot, cfg.layer), extra))
    (out / "config.json").write_text(
        json.dumps(build_config(src_cfg, cfg.layer, probe_meta, cfg.pooling), indent=1) + "\n")
    for fname in TOKENIZER_FILES:
        src = snapshot / fname
        if not src.exists():
            raise FileNotFoundError(f"{fname} missing from base snapshot {snapshot}")
        shutil.copyfile(src, out / fname)
    (out / "README.md").write_text(model_card(cfg, probe_meta))
    print(f"built {out} ({total / 2**30:.1f} GiB)")
    return out


def push(cfg: ServingConfig) -> str:
    """Upload an already-built ``cfg.publish_dir`` to ``cfg.hf_repo_id``."""
    from huggingface_hub import HfApi

    out = REPO / cfg.publish_dir
    if not (out / "model.safetensors.index.json").exists():
        raise FileNotFoundError(f"{out} has no built checkpoint; run build first")
    api = HfApi()
    api.create_repo(cfg.hf_repo_id, private=cfg.private, exist_ok=True)
    api.upload_folder(repo_id=cfg.hf_repo_id, folder_path=str(out))
    url = f"https://huggingface.co/{cfg.hf_repo_id}"
    print(f"pushed to {url}")
    return url


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--local-only", action="store_true",
                    help="build the checkpoint but do not upload")
    args = ap.parse_args()
    cfg = ServingConfig.load(args.config)
    build(cfg)
    if not args.local_only:
        push(cfg)


if __name__ == "__main__":
    main()
