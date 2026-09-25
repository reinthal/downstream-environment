"""LMProbe: a model truncated at the probe layer + a linear read-out.

Runs in the HF transformers env (`.venv`) — vLLM exposes no hidden states.
Construction is cheap; the (truncated) model loads lazily on first use. Scores
are ``p(deceptive)`` per conversation, mean-pooled over a span of the final
assistant turn ("full" = think + public text, "public" = post-</think> only).

    probe = LMProbe("Qwen/Qwen3.5-27B", 16, "probes/probe_logistic_regression.npz")
    p = probe.score(conversations)              # (N,) p(deceptive), full span

Several read-outs at the same layer share one model via ``with_probe``:

    dom = probe.with_probe("probes/probe_diff_of_means.npz")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .activations import collect_activations, encode_with_spans, load_truncated_decoder
from .config import ExperimentConfig
from .probes import LinearProbe

SPANS = ("full", "public")


class LMProbe:
    """HF slug + decoder layer + trained :class:`LinearProbe` (LR or DoM)."""

    def __init__(self, model_id: str, layer: int, probe: LinearProbe | str | Path, *,
                 batch_size: int = 8, max_len: int = 4096, dtype: str = "bfloat16",
                 device: str = "auto", truncate_layers: bool = True):
        if isinstance(probe, (str, Path)):
            probe = LinearProbe.load(str(probe))
        if probe.config.layer != layer:
            raise ValueError(f"probe was trained at layer {probe.config.layer}, "
                             f"but LMProbe is set to layer {layer}")
        self.model_id, self.layer, self.probe = model_id, layer, probe
        self.batch_size, self.max_len = batch_size, max_len
        self.dtype, self.device, self.truncate_layers = dtype, device, truncate_layers
        self._model = None
        self._tok = None

    @classmethod
    def from_config(cls, cfg: ExperimentConfig, probe: str | None = None) -> "LMProbe":
        """``probe``: key into ``cfg.probe_paths`` (optional when there is
        exactly one entry)."""
        if probe is None:
            if len(cfg.probe_paths) != 1:
                raise ValueError(f"cfg.probe_paths has {sorted(cfg.probe_paths)}; "
                                 "pass probe=<name>")
            probe = next(iter(cfg.probe_paths))
        return cls(cfg.model_id, cfg.layer, cfg.probe_paths[probe],
                   batch_size=cfg.batch_size, max_len=cfg.max_len,
                   dtype=cfg.dtype, device=cfg.device,
                   truncate_layers=cfg.truncate_layers)

    def with_probe(self, probe: LinearProbe | str | Path) -> "LMProbe":
        """A second read-out at the same layer, sharing this instance's model."""
        other = LMProbe(self.model_id, self.layer, probe,
                        batch_size=self.batch_size, max_len=self.max_len,
                        dtype=self.dtype, device=self.device,
                        truncate_layers=self.truncate_layers)
        other._model, other._tok = self._ensure_model()
        return other

    def _ensure_model(self):
        if self._model is None:
            self._model, self._tok = load_truncated_decoder(
                self.model_id, self.layer, dtype=self.dtype, device=self.device,
                truncate=self.truncate_layers)
        return self._model, self._tok

    def collect(self, conversations: list[list[dict]],
                spans: tuple[str, ...] = ("full",)) -> np.ndarray:
        """Pooled activations at the probe layer: (N, d), or (N, S, d) when
        several spans are asked for. Final assistant turns may carry
        ``reasoning_content`` (see :func:`encode_with_spans`)."""
        bad = set(spans) - set(SPANS)
        if bad:
            raise ValueError(f"unknown spans {sorted(bad)}; expected among {SPANS}")
        model, tok = self._ensure_model()
        encoded = []
        for msgs in conversations:
            ids, full, pub = encode_with_spans(tok, msgs, max_len=self.max_len)
            by_name = {"full": full, "public": pub}
            encoded.append((ids, [by_name[s] for s in spans]))
        acts = collect_activations(model, tok, encoded, [self.layer],
                                   batch_size=self.batch_size)[self.layer]
        return acts[:, 0] if len(spans) == 1 else acts

    def score(self, conversations: list[list[dict]], span: str = "full") -> np.ndarray:
        """(N,) ``p(deceptive)`` per conversation."""
        return self.probe.predict_proba(self.collect(conversations, (span,)))
