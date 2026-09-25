"""Generation backends: chat conversations in, raw thinking-model completions out.

Mirrors the `linear_probes.backends` pattern: callers pick a backend at runtime
via ``make_backend`` and get the same contract from both.

  * :class:`VLLMBackend` — bulk generation (~10x HF throughput). Runs under the
    vLLM env (`.venv-vllm`, vllm 0.21.0+cu129 — the last CUDA-12.x build, which
    this box's 535 driver requires). No steering, no activations.
  * :class:`HFBackend` — plain transformers ``generate``. Slow, but supports
    residual-stream steering hooks (:meth:`HFBackend.attach_steering`) and
    shares the weights/env with activation collection.

Both render prompts with the model's chat template (thinking enabled), so
completions start inside an open ``<think>`` block. If a completion never
closes its think block, ``generate`` force-closes it (append a wrap-up cue +
``</think>``, then a bounded continuation) so the public message still gets
produced; the count lands in ``backend.n_force_closed``.

Heavy deps (torch, transformers, vllm) are imported lazily inside the backend
that needs them.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

GEN_KWARGS = dict(temperature=0.7, top_p=0.95)
FORCE_CLOSE_TOKENS = 220
CLOSE_CUE = ("\nTime is up — I now write only my short public message, as direct "
             "speech, without mentioning my private reasoning.\n</think>\n\n")

_THINK_RE = re.compile(r"^(.*?)</think>", re.DOTALL)


def split_think(raw: str) -> tuple[str, str]:
    """(reasoning, public). If the think block never closes, the whole output
    is reasoning and public is empty."""
    m = _THINK_RE.match(raw)
    if not m:
        return raw.strip(), ""
    reasoning = m.group(1).replace("<think>", "").strip()
    return reasoning, raw[m.end():].strip()


def load_hf_model(model_id: str):
    """(model, tokenizer): bf16, sharded over visible GPUs, left padding."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    return model, tok


def decoder_layers(model):
    """The text decoder's `layers` ModuleList (nesting varies by family)."""
    cands = []
    for name, child in model.named_modules():
        if name.rsplit(".", 1)[-1] != "layers" or "vis" in name.lower():
            continue
        kids = list(child.children())
        if kids and (hasattr(kids[0], "mlp") or "Decoder" in type(kids[0]).__name__):
            cands.append((len(kids), child))
    if not cands:
        raise AttributeError("no decoder layers ModuleList found")
    return max(cands, key=lambda t: t[0])[1]


class GenerationBackend(ABC):
    """Shared contract: ``generate(convs, max_new_tokens)`` -> raw completions."""

    # instance-level overrides for ExperimentConfig injection; None means the
    # module globals stay authoritative (scripts patch those, e.g. 2026-09-24
    # sets generation.FORCE_CLOSE_TOKENS = 800)
    gen_kwargs: dict | None = None
    force_close_tokens: int | None = None

    def __init__(self, model_id: str):
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        self.n_force_closed = 0

    def render(self, conv: list[dict]) -> str:
        return self.tok.apply_chat_template(conv, tokenize=False,
                                            add_generation_prompt=True)

    def generate(self, convs: list[list[dict]], max_new_tokens: int, *,
                 seed: int = 0, coefs: list[float] | None = None) -> list[str]:
        """``convs``: chat ``messages`` lists. Returns raw completions
        (think block + public text) in input order, force-close applied.
        ``coefs``: per-conversation steering coefficients (HF backend only)."""
        raws = self._generate_texts([self.render(c) for c in convs],
                                    max_new_tokens, seed, coefs)
        idxs = [i for i, r in enumerate(raws) if "</think>" not in r]
        self.n_force_closed = len(idxs)
        if idxs:
            fct = self.force_close_tokens if self.force_close_tokens is not None \
                else FORCE_CLOSE_TOKENS
            cont = self._generate_texts(
                [self.render(convs[i]) + raws[i] + CLOSE_CUE for i in idxs],
                fct, seed + 1,
                [coefs[i] for i in idxs] if coefs is not None else None)
            for i, c in zip(idxs, cont):
                raws[i] = raws[i] + CLOSE_CUE + c
        return raws

    @abstractmethod
    def _generate_texts(self, texts: list[str], max_new_tokens: int,
                        seed: int, coefs: list[float] | None) -> list[str]:
        """Sample a completion for each raw prompt string."""


# ── vLLM ─────────────────────────────────────────────────────────────────────

class VLLMBackend(GenerationBackend):
    """Offline vLLM engine. Run under `.venv-vllm` (see AGENTS.md)."""

    def __init__(self, model_id: str, tensor_parallel_size: int | None = None,
                 max_model_len: int = 8192, gpu_memory_utilization: float = 0.90,
                 enforce_eager: bool = False):
        super().__init__(model_id)
        import os

        # flashinfer's sampler JIT-compiles against CUDA-12 headers the system
        # nvcc (11.8) doesn't have; the native torch sampler needs no JIT
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        import torch
        from vllm import LLM

        tp = tensor_parallel_size or max(1, torch.cuda.device_count())
        self.llm = LLM(model=model_id, tensor_parallel_size=tp,
                       max_model_len=max_model_len, dtype="bfloat16",
                       gpu_memory_utilization=gpu_memory_utilization,
                       enforce_eager=enforce_eager)

    def _generate_texts(self, texts, max_new_tokens, seed, coefs):
        if coefs is not None:
            raise ValueError("steering requires the HF backend (--backend hf)")
        from vllm import SamplingParams

        sps = [SamplingParams(max_tokens=max_new_tokens,
                              seed=seed * 100_003 + i,
                              **(self.gen_kwargs or GEN_KWARGS))
               for i in range(len(texts))]
        outs = self.llm.generate(texts, sps)
        return [o.outputs[0].text for o in outs]


# ── HF transformers ──────────────────────────────────────────────────────────

class HFBackend(GenerationBackend):
    """Plain transformers ``generate``; supports residual-stream steering."""

    def __init__(self, model_id: str, chunk: int = 25):
        super().__init__(model_id)
        self.chunk = chunk
        self.model, self.tok = load_hf_model(model_id)
        self.steerer: _Steerer | None = None

    def attach_steering(self, layer: int, vec) -> None:
        """Add ``coefs[i] * vec`` to the residual stream at ``layer`` during
        conversation *i*'s generation (all token positions)."""
        self.steerer = _Steerer(self.model, layer, vec)

    def _generate_texts(self, texts, max_new_tokens, seed, coefs):
        import torch

        if coefs is not None and self.steerer is None:
            raise ValueError("coefs given but no steering attached")
        torch.manual_seed(seed)
        outs = [""] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        for c0 in range(0, len(order), self.chunk):
            idxs = order[c0:c0 + self.chunk]
            enc = self.tok([texts[i] for i in idxs], return_tensors="pt",
                           padding=True, add_special_tokens=False).to(self.model.device)
            if self.steerer is not None and coefs is not None:
                self.steerer.set([coefs[i] for i in idxs])
            try:
                with torch.no_grad():
                    gen = self.model.generate(
                        **enc, max_new_tokens=max_new_tokens, do_sample=True,
                        pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                        **(self.gen_kwargs or GEN_KWARGS))
            finally:
                if self.steerer is not None:
                    self.steerer.clear()
            new = gen[:, enc["input_ids"].shape[1]:]
            for r, i in enumerate(idxs):
                outs[i] = self.tok.decode(new[r], skip_special_tokens=True)
        return outs


class _Steerer:
    """Forward hook adding a per-row coefficient times a direction vector to
    one decoder layer's output."""

    def __init__(self, model, layer: int, vec):
        import torch
        self.v = torch.tensor(vec, dtype=torch.bfloat16)
        self.coef = None
        self._moved = False
        self.handle = decoder_layers(model)[layer].register_forward_hook(self._hook)

    def set(self, coefs):
        import torch
        self.coef = torch.tensor(coefs, dtype=torch.bfloat16)

    def clear(self):
        self.coef = None

    def _hook(self, module, args, output):
        if self.coef is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if not self._moved:
            self.v = self.v.to(h.device)
            self._moved = True
        add = self.coef.to(h.device).view(-1, 1, 1) * self.v.view(1, 1, -1)
        h = h + add.to(h.dtype)
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h


def make_backend(name: str, model_id: str, **kwargs) -> GenerationBackend:
    """Runtime injection point: ``"vllm"`` or ``"hf"``."""
    backends = {"vllm": VLLMBackend, "hf": HFBackend}
    if name not in backends:
        raise ValueError(f"unknown backend {name!r}; expected one of {sorted(backends)}")
    return backends[name](model_id, **kwargs)
