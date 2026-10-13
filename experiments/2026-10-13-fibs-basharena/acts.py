"""Rendering, read windows and window-activation collection for the paper's
probes on Qwen3.8-27B. Adapted from experiments/2026-10-12-fibs-probes/acts.py
(probe_inference activation contract, vendor/caught-in-the-act-probes README
"Activation contract"):

  * the follow-up is a user turn + a prefilled assistant answer rendered by
    the model's own chat template with an empty think block
    (``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nYes.<|im_end|>``);
  * Qwen3.8's template differs from Qwen3.5's: ``preserve_thinking`` keeps
    earlier assistant turns' reasoning unless False, and thinking enabled
    injects a "Reasoning effort" line into the system prompt — so both
    renders take ``template_kwargs`` (enable_thinking=False,
    preserve_thinking=False), which reproduces the paper's Qwen3.5 read: no
    step's reasoning in the input, and the prefilled answer after an empty
    think block;
  * layer k is decoder block k's output = HF ``hidden_states[k + 1]``; bf16;
  * the read window is the final user turn through the end-of-turn token;
    linear/MLP read the token before ``<|im_end|>`` (the ``.``), EFC/axial
    read the whole window. Only the window is kept.

Imports torch lazily so the config/driver side can import this module.
"""
from __future__ import annotations

from dataclasses import dataclass

IM_END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"


@dataclass
class Rendered:
    ids: list[int]
    followup_start: int      # first token of the final user turn
    completion_start: int    # first token of the prefilled answer
    n_tokens: int


def render_followup(tok, messages: list[dict], question: str, answer: str,
                    tools: list[dict] | None = None, template_kwargs: dict | None = None) -> Rendered:
    """``messages`` is a finished dialogue ending with the assistant's reply
    (or its tool call plus the tool result); the follow-up question and
    prefilled answer are appended and tokenized. ``tools`` are the schemas
    the provider rendered into the system prompt."""
    return render_completion(tok, [*messages, {"role": "user", "content": question}], answer, tools, template_kwargs)


def render_completion(tok, messages: list[dict], answer: str,
                      tools: list[dict] | None = None, template_kwargs: dict | None = None) -> Rendered:
    """``messages`` is the transcript the answer follows (its last message is
    usually the follow-up user turn; in fibs' tool follow-up rows it is a
    tool result, which the template renders as a user turn); ``answer`` is
    the prefilled assistant completion. The read window starts at the last
    user-rendered turn before the answer. Fails loudly if the template does
    not nest as the contract expects."""
    kw = dict(template_kwargs or {})
    if tools:
        kw["tools"] = tools
    conv = [*messages, {"role": "assistant", "content": answer, "reasoning_content": ""}]
    full = tok.apply_chat_template(conv, tokenize=False, **kw)
    if not full.endswith(IM_END + "\n"):
        raise ValueError(f"unexpected template ending {full[-40:]!r}")
    full = full[:-1]                                   # end at the end-of-turn token
    gen = tok.apply_chat_template(conv[:-1], tokenize=False, add_generation_prompt=True, **kw)
    if not gen.endswith(EMPTY_THINK):
        raise ValueError("the generation prompt does not end with an empty think block; "
                         "template_kwargs must disable thinking")
    if not full.startswith(gen):
        raise ValueError("chat template does not render the follow-up as the probes expect")
    for m in conv[:-1]:      # no earlier turn's reasoning may be in the input (the paper's read)
        rc = (m.get("reasoning_content") or "").strip() if m.get("role") == "assistant" else ""
        if rc and rc in full:
            raise ValueError("an earlier turn's reasoning is in the rendered input; "
                             "template_kwargs must set preserve_thinking=False")
    ids = tok(full, add_special_tokens=False)["input_ids"]
    cs = len(tok(gen, add_special_tokens=False)["input_ids"])
    # the window starts at the last user-rendered turn before the answer (the last <|im_start|> opens the answer)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    starts = [i for i, t_ in enumerate(ids) if t_ == im_start]
    if len(starts) < 2:
        raise ValueError("fewer than two <|im_start|> tokens in the rendered input")
    user_starts = [i for i in starts[:-1] if tok.decode(ids[i:i + 2]) == "<|im_start|>user"]
    if not user_starts:
        raise ValueError("no user turn before the answer")
    fs = user_starts[-1]
    if not (0 <= fs < cs < len(ids)) \
            or tok.decode(ids[starts[-1]:starts[-1] + 2]) != "<|im_start|>assistant" \
            or tok.decode(ids[-1:]) != IM_END:
        raise ValueError(f"token boundaries off: followup_start={fs}, completion_start={cs}, n={len(ids)}")
    if len(ids) - cs < 2:
        raise ValueError("the answer must have at least one token before <|im_end|>")
    return Rendered(ids, fs, cs, len(ids))


def window_masks(r: Rendered):
    """(prompt_mask, completion_mask) over the window ``ids[followup_start:]``."""
    import torch
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


def _decoder(model):
    inner = model
    while hasattr(inner, "model"):
        inner = inner.model
    return inner


class _Capture:
    """Forward hooks on the probe layers' decoder blocks; ``take()`` returns
    the captured outputs (bf16, cpu) and clears them. Reading block outputs
    directly (= HF hidden_states[k+1]) avoids materialising every layer's
    hidden state for 100k-token inputs."""

    def __init__(self, inner, layers: list[int]):
        self.layers, self.out, self.on = layers, {}, False
        self.handles = [inner.layers[L].register_forward_hook(self._hook(L)) for L in layers]

    def _hook(self, L):
        def hook(mod, args, out):
            if self.on:
                h = out[0] if isinstance(out, tuple) else out
                self.out[L] = h
        return hook

    def take(self):
        o, self.out = self.out, {}
        return o

    def remove(self):
        for h in self.handles:
            h.remove()


def collect_trajectory(model, tok, steps: list[dict], layers: list[int], chunk: int = 8192, log=print) -> dict:
    """Window activations for every step of one trajectory with one pass
    over the trajectory: step k's rendered prefix (``ids[:followup_start]``)
    is a prefix of step k+1's, so the model's KV/recurrent cache advances
    over the shared tokens once and each prefill branch runs on a copy of
    the cache over its own window tokens only. Exact for a causal model
    (verified token-level nesting; raises otherwise).

    ``steps``: [{"step": k, "rendered": {prefill: Rendered}}] in trajectory
    order. Returns {(step, prefill): (len(layers), window, d_model) bf16}."""
    import copy
    import time
    import torch
    inner = _decoder(model)
    cap = _Capture(inner, layers)
    out, cache, done, prev = {}, None, 0, []
    t0 = time.time()
    timing = {"advance": 0.0, "copy": 0.0, "branch": 0.0, "advance_tokens": 0, "branch_tokens": 0}
    try:
        with torch.no_grad():
            for s in steps:
                rds = s["rendered"]
                starts = {r.followup_start for r in rds.values()}
                if len(starts) != 1:
                    raise ValueError(f"step {s['step']}: prefills disagree on followup_start {starts}")
                fs = starts.pop()
                base = next(iter(rds.values())).ids[:fs]
                for r in rds.values():
                    if r.ids[:fs] != base:
                        raise ValueError(f"step {s['step']}: prefills disagree on the shared prefix")
                if fs < done or base[:done] != prev[:done]:
                    raise ValueError(f"step {s['step']}: rendered prefix is not nested in the previous step's")
                # advance the main cache over the new shared tokens, chunked
                t = time.time()
                for a in range(done, fs, chunk):
                    ids = torch.tensor([base[a:min(a + chunk, fs)]], device=model.device)
                    res = inner(input_ids=ids, past_key_values=cache, use_cache=True)
                    cache = res.past_key_values
                    del res
                torch.cuda.synchronize()
                timing["advance"] += time.time() - t
                timing["advance_tokens"] += fs - done
                done, prev = fs, base
                for prefill, r in rds.items():
                    t = time.time()
                    branch = copy.deepcopy(cache)
                    torch.cuda.synchronize()
                    timing["copy"] += time.time() - t
                    t = time.time()
                    ids = torch.tensor([r.ids[fs:]], device=model.device)
                    cap.on = True
                    res = inner(input_ids=ids, past_key_values=branch, use_cache=True)
                    cap.on = False
                    got = cap.take()
                    out[(s["step"], prefill)] = torch.stack([got[L][0].to(torch.bfloat16).cpu() for L in layers])
                    del res, branch
                    torch.cuda.synchronize()
                    timing["branch"] += time.time() - t
                    timing["branch_tokens"] += len(r.ids) - fs
                log(f"  step {s['step']}: prefix {fs} tokens, {time.time() - t0:.0f}s")
    finally:
        cap.remove()
    out["timing"] = {k: round(v, 2) if isinstance(v, float) else v for k, v in timing.items()}
    return out


def collect_windows(model, tok, rendered: list[Rendered], layers: list[int],
                    batch_tokens: int, max_batch: int, log=print, pad_to: int = 4096) -> list:
    """For each rendered input, a bf16 tensor (len(layers), window, d_model)
    holding the residual stream at ``layers`` over the read window.
    Right-padded batches under a token budget, longest first; the padded
    width is a multiple of ``pad_to`` so the Triton kernels' per-shape
    autotuning happens once per bucket (right padding leaves the real
    tokens' activations unchanged in a causal model)."""
    import torch
    inner = _decoder(model)
    cap = _Capture(inner, layers)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(len(rendered)), key=lambda i: -rendered[i].n_tokens)
    out: list = [None] * len(rendered)
    i = 0
    try:
        with torch.no_grad():
            while i < len(order):
                w = rendered[order[i]].n_tokens
                bs = min(max_batch, max(1, batch_tokens // w))
                batch = order[i:i + bs]
                i += bs
                wp = -(-w // pad_to) * pad_to
                input_ids = torch.full((len(batch), wp), pad, dtype=torch.long)
                attn = torch.zeros((len(batch), wp), dtype=torch.long)
                for r, j in enumerate(batch):
                    ids = rendered[j].ids
                    input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attn[r, :len(ids)] = 1
                cap.on = True
                res = inner(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
                            use_cache=False)
                cap.on = False
                got = cap.take()
                for r, j in enumerate(batch):
                    rj = rendered[j]
                    out[j] = torch.stack([got[L][r, rj.followup_start:rj.n_tokens].to(torch.bfloat16).cpu()
                                          for L in layers])
                del res, got
                log(f"  collected {min(i, len(order))}/{len(order)} (batch {len(batch)} x {w} tokens, padded {wp})")
    finally:
        cap.remove()
    return out
