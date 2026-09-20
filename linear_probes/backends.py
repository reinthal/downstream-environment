"""Activation backends: where the forward pass runs.

The trainer never knows whether activations come from a local GPU or a remote
NDIF session — it is handed an :class:`ActivationBackend` at runtime and calls
``collect``. Both backends return the same thing: one pooled residual-stream
vector per conversation, read at ``config.layer`` and mean-pooled over the
final assistant response's tokens.

Heavy deps (torch, transformers, nnsight) are imported lazily inside the
backend that needs them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import ProbeConfig


class ActivationBackend(ABC):
    """Interface the runtime injects: local or NDIF, same contract."""

    def __init__(self, config: ProbeConfig):
        self.config = config

    @abstractmethod
    def collect(self, conversations: list[list[dict]]) -> np.ndarray:
        """``conversations``: chat ``messages`` lists. Returns (N, d_model)
        activations at ``config.layer``, mean-pooled over each conversation's
        final assistant turn, in input order."""


# ── shared tokenization / batching ──────────────────────────────────────────

def tokenize_conversation(messages: list, tokenizer, max_len: int):
    """Chat-template tokenize + locate the final assistant turn. Returns
    ``(token_ids, (start, end))``; left-trimmed to ``max_len`` so the response
    survives. Falls back to "the whole text is the span" for tokenizers
    without a chat template."""
    try:
        full = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True)["input_ids"]
        prefix = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
        ids, s, e = list(full), len(prefix), len(full)
    except Exception:
        text = "\n".join(m.get("content", "") for m in messages)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = ids or [tokenizer.eos_token_id or 0]
        s, e = 0, len(ids)
    if max_len and len(ids) > max_len:
        cut = len(ids) - max_len
        ids, s, e = ids[cut:], max(0, s - cut), e - cut
    if e <= s:                                   # no assistant turn → pool everything
        s, e = 0, len(ids)
    return ids, (s, e)


def iter_batches(toks: list, spans: list, pad_id: int, batch_size: int):
    """Length-sorted, right-padded batches. Yields
    ``(positions, input_ids, attention_mask, batch_spans)``; ``positions`` maps
    batch rows back to input order."""
    import torch

    order = sorted(range(len(toks)), key=lambda i: len(toks[i]))
    for b0 in range(0, len(order), batch_size):
        bpos = order[b0:b0 + batch_size]
        w = max(len(toks[i]) for i in bpos)
        input_ids = torch.full((len(bpos), w), pad_id, dtype=torch.long)
        attn = torch.zeros((len(bpos), w), dtype=torch.long)
        for r, i in enumerate(bpos):
            input_ids[r, :len(toks[i])] = torch.tensor(toks[i], dtype=torch.long)
            attn[r, :len(toks[i])] = 1
        yield bpos, input_ids, attn, [spans[i] for i in bpos]


def _pad_id(tokenizer) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


# ── local ────────────────────────────────────────────────────────────────────

class LocalBackend(ActivationBackend):
    """Plain transformers forward on this machine (GPU or CPU)."""

    def collect(self, conversations):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.config
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        kwargs = {"dtype": getattr(torch, cfg.dtype), "device_map": cfg.device}
        if cfg.truncate_layers:
            from transformers import AutoConfig
            mcfg = AutoConfig.from_pretrained(cfg.model_id)
            mcfg.num_hidden_layers = cfg.layer + 1   # blocks > layer are never read; skip loading them
            kwargs["config"] = mcfg
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
        except TypeError:                        # transformers < 5 spells it torch_dtype
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
        if cfg.lora_id:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, cfg.lora_id)
        model.eval()

        toks, spans = zip(*(tokenize_conversation(m, tokenizer, cfg.max_len) for m in conversations))
        rows = {}
        with torch.no_grad():
            for bpos, input_ids, attn, bspans in iter_batches(list(toks), list(spans),
                                                              _pad_id(tokenizer), cfg.batch_size):
                out = model(input_ids=input_ids.to(model.device),
                            attention_mask=attn.to(model.device),
                            output_hidden_states=True)
                # hidden_states[0] is the embeddings, so [layer + 1] is block `layer`'s output
                h = out.hidden_states[cfg.layer + 1]
                for r, i in enumerate(bpos):
                    s, e = bspans[r]
                    rows[i] = h[r, s:e].float().mean(0).cpu().numpy()
        return np.stack([rows[i] for i in range(len(toks))])


# ── NDIF ─────────────────────────────────────────────────────────────────────

class NDIFBackend(ActivationBackend):
    """nnsight traces against a model served remotely on NDIF. All batches run
    inside ONE remote session (one queue wait, one LoRA hotswap)."""

    def collect(self, conversations):
        import torch

        cfg = self.config
        model = _build_nnsight_model(cfg.model_id, cfg.lora_id)
        layers = _decoder_layers(model)
        tokenizer = model.tokenizer

        toks, spans = zip(*(tokenize_conversation(m, tokenizer, cfg.max_len) for m in conversations))
        batches = list(iter_batches(list(toks), list(spans), _pad_id(tokenizer), cfg.batch_size))

        with model.session(remote=True):
            pieces, poses = [], []
            for bpos, input_ids, attn, bspans in batches:
                with model.trace({"input_ids": input_ids, "attention_mask": attn}) as tracer:
                    h = layers[cfg.layer].output
                    if not hasattr(h, "shape"):  # some archs wrap the residual in a tuple
                        h = h[0]
                    pooled = torch.stack([h[r, s:e].float().mean(0)
                                          for r, (s, e) in enumerate(bspans)])
                    tracer.stop()                # skip layers > cfg.layer and the LM head
                pieces.append(pooled)
                poses.append(torch.tensor(bpos))
            feats = torch.cat(pieces).save()     # only saved values survive the session
            pos = torch.cat(poses).save()

        F = feats.cpu().float().numpy()
        out = np.zeros_like(F)
        out[pos.cpu().numpy()] = F               # back to input order
        return out


def _build_nnsight_model(model_id: str, lora_id: str | None):
    """nnsight handle: config/tokenizer load locally, weights live on NDIF.
    VLM repos (e.g. Gemma-3) need the VisionLanguageModel wrapper."""
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(model_id)
        archs = getattr(cfg, "architectures", None) or []
        is_vlm = bool(getattr(cfg, "vision_config", None)) or any(
            ("ConditionalGeneration" in a) or ("VisionLanguage" in a) for a in archs)
    except Exception:
        is_vlm = False
    try:
        from nnsight import LanguageModel, VisionLanguageModel
    except ImportError:                          # nnsight 0.7.x drops the top-level re-export
        from nnsight.modeling.language import LanguageModel
        from nnsight.modeling.vlm import VisionLanguageModel

    Wrapper = VisionLanguageModel if is_vlm else LanguageModel
    return Wrapper(model_id, **({"peft": lora_id} if lora_id else {}))


def _decoder_layers(model):
    """The text decoder's ``layers`` ModuleList, found by search — the nesting
    varies with family, VLM vs text, and PEFT wrapping. Picks the largest
    non-vision ModuleList named ``layers`` whose first block looks like a
    decoder block."""
    candidates = []
    for name, child in model.model.named_modules():
        if name.rsplit(".", 1)[-1] != "layers" or "vis" in name.lower():
            continue
        kids = list(child.children())
        if kids and ("Decoder" in type(kids[0]).__name__
                     or hasattr(kids[0], "mlp") or hasattr(kids[0], "mixer")):
            candidates.append((len(kids), child))
    if not candidates:
        raise AttributeError(f"no decoder `layers` ModuleList found in {type(model.model).__name__}")
    return max(candidates, key=lambda t: t[0])[1]


def make_backend(name: str, config: ProbeConfig) -> ActivationBackend:
    """Runtime injection point: ``"local"`` or ``"ndif"``."""
    backends = {"local": LocalBackend, "ndif": NDIFBackend}
    if name not in backends:
        raise ValueError(f"unknown backend {name!r}; expected one of {sorted(backends)}")
    return backends[name](config)
