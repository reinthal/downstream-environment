"""Span encoding + bare-decoder activation collection (HF transformers env).

Lifted from experiments/2026-09-22/common.py so experiments stop copying it.
Two facts of life are baked in (see AGENTS.md and the 2026-09-20 log):

  * forwards run the BARE decoder (no LM head) — the vocab-sized logits are
    never read and CUDA-launch-fail at long sequences;
  * models can be loaded truncated at the probe layer
    (:func:`load_truncated_decoder`) — blocks above it are never read, so a
    27B probed at layer 16 loads ~1/4 of the decoder.

Heavy deps (torch, transformers) are imported lazily.
"""
from __future__ import annotations

import numpy as np


def load_truncated_decoder(model_id: str, layer: int, *, dtype: str = "bfloat16",
                           device: str = "auto", truncate: bool = True):
    """(model, tokenizer) with only decoder blocks ``0..layer`` loaded.
    Activations at <= ``layer`` are unchanged by the truncation. Handles
    composite (VLM) configs whose decoder depth lives in ``text_config``."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    mcfg = AutoConfig.from_pretrained(model_id)
    if truncate:
        tcfg = getattr(mcfg, "text_config", None) or mcfg
        tcfg.num_hidden_layers = layer + 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=mcfg, dtype=getattr(torch, dtype), device_map=device)
    if truncate:
        # the decoder applies its final RMSNorm after the last block, so the
        # last hidden_states entry — our hidden_states[layer + 1] — would be
        # norm(block output) instead of the raw residual stream the probes
        # were trained on; neutralize it
        inner = model
        while hasattr(inner, "model"):
            inner = inner.model
        for attr in ("norm", "final_layernorm"):
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        else:
            raise AttributeError(f"no final norm found on {type(inner).__name__}; "
                                 "cannot truncate safely")
    model.eval()
    return model, tok


def encode_with_spans(tok, messages: list[dict], max_len: int = 4096):
    """Tokenize a finished conversation whose final turn is an assistant
    message with `reasoning_content` + `content`. Returns
    (ids, full_span, public_span): full_span covers the whole final assistant
    turn (reasoning + announcement), public_span only the post-</think>
    announcement tokens."""
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    p = len(tok(prefix, add_special_tokens=False)["input_ids"])
    et = tok("</think>", add_special_tokens=False)["input_ids"]
    pub = None
    for j in range(len(ids) - len(et), p - 1, -1):        # last occurrence
        if ids[j:j + len(et)] == et:
            pub = j + len(et)
            break
    if pub is None or pub >= len(ids):
        pub = p
    if max_len and len(ids) > max_len:
        cut = len(ids) - max_len
        ids, p, pub = ids[cut:], max(0, p - cut), max(0, pub - cut)
    return ids, (p, len(ids)), (pub, len(ids))


def collect_activations(model, tok, encoded: list[tuple], layers: list[int],
                        batch_size: int = 8) -> dict[int, np.ndarray]:
    """`encoded`: list of (ids, span) or (ids, [span, ...]) pairs. Returns
    {layer: (N, d)} — or {layer: (N, S, d)} when spans are lists — activations
    mean-pooled over each span, teacher-forced in one forward per batch."""
    import torch

    multi = isinstance(encoded[0][1], list)
    rows = {L: {} for L in layers}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # bare decoder: the LM head's vocab-sized logits are never read and cause
    # CUDA launch failures at long sequences (see 2026-09-20 log)
    inner = model
    while hasattr(inner, "model"):
        inner = inner.model
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    for b0 in range(0, len(order), batch_size):
        bpos = order[b0:b0 + batch_size]
        w = max(len(encoded[i][0]) for i in bpos)
        input_ids = torch.full((len(bpos), w), pad, dtype=torch.long)
        attn = torch.zeros((len(bpos), w), dtype=torch.long)
        for r, i in enumerate(bpos):
            ids = encoded[i][0]
            input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[r, :len(ids)] = 1
        with torch.no_grad():
            out = inner(input_ids=input_ids.to(model.device),
                        attention_mask=attn.to(model.device), output_hidden_states=True)
        for L in layers:
            h = out.hidden_states[L + 1]          # [0] is embeddings
            for r, i in enumerate(bpos):
                n = len(encoded[i][0])
                spans = encoded[i][1] if multi else [encoded[i][1]]
                pooled = []
                for s, e in spans:
                    e = min(e, n)
                    s = min(s, e - 1) if e > 0 else 0   # empty span -> last token
                    pooled.append(h[r, s:e].float().mean(0).cpu().numpy())
                rows[L][i] = np.stack(pooled) if multi else pooled[0]
        del out
    return {L: np.stack([rows[L][i] for i in range(len(encoded))]) for L in layers}
