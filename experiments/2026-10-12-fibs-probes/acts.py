"""Rendering, read windows and window-activation collection for the paper's
probes (probe_inference activation contract, vendor/caught-in-the-act-probes
README "Activation contract"):

  * the follow-up is a user turn + a prefilled assistant answer rendered by
    the model's own chat template with an empty think block
    (``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nYes.<|im_end|>``);
  * layer k is decoder block k's output = HF ``hidden_states[k + 1]``; the
    model runs in bf16 and the activations are stored in bf16;
  * the read window is the final user turn through the end-of-turn token;
    linear/MLP read the token before ``<|im_end|>`` (the ``.``), EFC/axial
    read the whole window. Only the window is kept: probe_inference compacts
    the selected tokens to the front before scoring, so positions outside it
    never reach a probe.

Everything here runs in ``.venv`` (HF transformers); the bare decoder is used
as in linear_probes.activations.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

IM_END = "<|im_end|>"


@dataclass
class Rendered:
    ids: list[int]
    followup_start: int      # first token of the final user turn
    completion_start: int    # first token of the prefilled answer
    n_tokens: int


def render_followup(tok, messages: list[dict], question: str, answer: str,
                    tools: list[dict] | None = None) -> Rendered:
    """``messages`` is a finished dialogue ending with the assistant's reply
    (or its tool call plus the tool result); the follow-up question and
    prefilled answer are appended and tokenized. ``tools`` are the schemas
    the provider rendered into the system prompt. Fails loudly if the
    template does not nest as the contract expects."""
    conv = [*messages, {"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    kw = {"tools": tools} if tools else {}
    full = tok.apply_chat_template(conv, tokenize=False, **kw)
    if not full.endswith(IM_END + "\n"):
        raise ValueError(f"unexpected template ending {full[-40:]!r}")
    full = full[:-1]                                   # end at the end-of-turn token
    gen = tok.apply_chat_template(conv[:-1], tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False, **kw)
    if not (full.startswith(gen) and gen.endswith("<think>\n\n</think>\n\n")):
        raise ValueError("chat template does not render the follow-up as the probes expect")
    ids = tok(full, add_special_tokens=False)["input_ids"]
    cs = len(tok(gen, add_special_tokens=False)["input_ids"])
    # the final user turn starts at the second-to-last <|im_start|> (the last one opens the answer)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    starts = [i for i, t_ in enumerate(ids) if t_ == im_start]
    if len(starts) < 2:
        raise ValueError("fewer than two <|im_start|> tokens in the rendered follow-up")
    fs = starts[-2]
    if not (0 <= fs < cs < len(ids)) or tok.decode(ids[fs:fs + 2]) != "<|im_start|>user" \
            or tok.decode(ids[starts[-1]:starts[-1] + 2]) != "<|im_start|>assistant" \
            or tok.decode(ids[-1:]) != IM_END:
        raise ValueError(f"token boundaries off: followup_start={fs}, completion_start={cs}, n={len(ids)}")
    if len(ids) - cs < 2:
        raise ValueError("the answer must have at least one token before <|im_end|>")
    return Rendered(ids, fs, cs, len(ids))


def window_masks(r: Rendered) -> tuple[torch.Tensor, torch.Tensor]:
    """(prompt_mask, completion_mask) over the window ``ids[followup_start:]``."""
    w = r.n_tokens - r.followup_start
    completion = torch.zeros(w, dtype=torch.bool)
    completion[r.completion_start - r.followup_start:] = True
    return ~completion, completion


def load_model(model_path: str, top_layer: int, dtype: str = "bfloat16", device: str = "auto"):
    """Decoder truncated above ``top_layer`` with the final norm neutralized —
    linear_probes.activations.load_truncated_decoder, i.e. hidden_states[k+1]
    is block k's raw residual output for every k <= top_layer."""
    from linear_probes.activations import load_truncated_decoder
    return load_truncated_decoder(model_path, top_layer, dtype=dtype, device=device)


@torch.no_grad()
def collect_windows(model, tok, rendered: list[Rendered], layers: list[int],
                    batch_tokens: int, max_batch: int, log=print) -> list[torch.Tensor]:
    """For each rendered input, a bf16 tensor (len(layers), window, d_model)
    holding the residual stream at ``layers`` over the read window.
    Right-padded batches under a token budget, longest first."""
    inner = model
    while hasattr(inner, "model"):
        inner = inner.model
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(len(rendered)), key=lambda i: -rendered[i].n_tokens)
    out: list[torch.Tensor | None] = [None] * len(rendered)
    i = 0
    while i < len(order):
        w = rendered[order[i]].n_tokens
        bs = min(max_batch, max(1, batch_tokens // w))
        batch = order[i:i + bs]
        i += bs
        input_ids = torch.full((len(batch), w), pad, dtype=torch.long)
        attn = torch.zeros((len(batch), w), dtype=torch.long)
        for r, j in enumerate(batch):
            ids = rendered[j].ids
            input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[r, :len(ids)] = 1
        res = inner(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
                    output_hidden_states=True)
        for r, j in enumerate(batch):
            rj = rendered[j]
            out[j] = torch.stack([res.hidden_states[L + 1][r, rj.followup_start:rj.n_tokens]
                                  .to(torch.bfloat16).cpu() for L in layers])
        del res
        log(f"  collected {min(i, len(order))}/{len(order)} (batch {len(batch)} x {w} tokens)")
    return out  # type: ignore[return-value]
