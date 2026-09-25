"""LMSteering: vLLM generation with a diff-of-means direction added to the
residual stream.

Runs in the vLLM env (`.venv-vllm`). The steering vector MUST come from a
:class:`~linear_probes.probes.DiffOfMeansProbe` — either its saved probe npz
(carries ``raw_diff`` since 2026-09-24) or a raw steer npz with a ``steer_vec``
key (``train_probes.py`` writes these); other probe kinds and bare arrays are
rejected because their raw-activation-space provenance can't be verified.

How steering reaches the workers: vLLM 0.21 has no public activation-hook API,
but ``LLM.apply_model(fn)`` ships ``fn`` to every (TP) worker and calls it on
the ``nn.Module``. We use it to register a plain forward hook on decoder block
``layer`` — its output tuple's ``hidden_states`` element feeds the residual
stream, so adding ``coef * vec`` there matches the HF ``_Steerer`` exactly.
``fn`` crosses two boundaries: client -> engine core (cloudpickle, gated
behind ``VLLM_ALLOW_INSECURE_SERIALIZATION=1``) and engine core -> workers
(PLAIN pickle) — hence the module-level worker functions below; closures
don't survive the second hop. Two more constraints follow:

  * the engine runs ``enforce_eager=True``: under the default torch.compile /
    CUDA-graph path, Python hooks are baked in at capture and later coef
    changes are silently ignored;
  * the coefficient is engine-global, so ``generate`` groups conversations by
    coefficient and runs one engine pass per distinct value (per-conversation
    seeds stay tied to input position, not group). Grouping changes vLLM's
    batch composition, and vLLM sampling is only bit-reproducible under
    identical batching — so a coef-0 conversation in a mixed batch won't
    bitwise-match an all-zero run (the hook itself IS a bit-exact no-op at
    coef 0: verified against an unhooked engine with matched batching).

Callers need an ``if __name__ == "__main__":`` guard — vLLM workers re-import
the main module.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np

from linear_probes.config import DIFF_OF_MEANS, ExperimentConfig
from linear_probes.probes import DiffOfMeansProbe, LinearProbe

from .generation import GEN_KWARGS, VLLMBackend, decoder_layers


def load_steering_vector(source) -> tuple[np.ndarray, int, str]:
    """``(vec, layer, model_id)`` from a DoM probe (instance or npz) or a raw
    steer npz (``steer_vec`` key). Rejects anything else."""
    if isinstance(source, DiffOfMeansProbe):
        if source.raw_diff is None:
            raise ValueError("DiffOfMeansProbe has no raw_diff (not fitted, or "
                             "loaded from an npz saved before raw_diff was stored)")
        return (np.asarray(source.raw_diff), source.config.layer,
                source.config.model_id)
    if isinstance(source, LinearProbe):
        raise ValueError(f"steering vector must come from a diff-of-means probe, "
                         f"got {source.kind!r}")
    if isinstance(source, (str, Path)):
        z = np.load(source, allow_pickle=False)
        if "steer_vec" in z:
            return z["steer_vec"], int(z["layer"]), str(z["model_id"])
        if "kind" in z:
            if str(z["kind"]) != DIFF_OF_MEANS:
                raise ValueError(f"steering vector must come from a diff-of-means "
                                 f"probe, got {str(z['kind'])!r} in {source}")
            if "raw_diff" not in z:
                raise ValueError(f"{source} is a diff-of-means probe npz without "
                                 "raw_diff — refit/re-save it, or point at the "
                                 "steer_diff_of_means.npz next to it")
            return z["raw_diff"], int(z["layer"]), str(z["model_id"])
        raise ValueError(f"{source} is neither a steer npz (steer_vec) nor a probe npz (kind)")
    raise TypeError(f"expected a DiffOfMeansProbe or an npz path, got {type(source).__name__} "
                    "(bare vectors are rejected: provenance can't be checked)")


# ── worker-side functions (must stay module-level: plain-pickled to workers) ─

def _install_hook(model, layer: int, vec: np.ndarray) -> str:
    import torch

    m = decoder_layers(model)[layer]
    dev = next(m.parameters()).device
    m._steer_vec = torch.tensor(vec, dtype=torch.bfloat16, device=dev)
    m._steer_coef = 0.0

    def hook(module, args, output):
        if module._steer_coef == 0.0:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + module._steer_coef * module._steer_vec.to(h.dtype)
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h

    m._steer_handle = m.register_forward_hook(hook)
    return f"{type(m).__name__}[{layer}] on {dev}"


def _set_coef(model, layer: int, coef: float) -> float:
    decoder_layers(model)[layer]._steer_coef = coef
    return coef


class LMSteering(VLLMBackend):
    """vLLM generation backend steered along a trained DoM direction.

    Same ``generate(convs, max_new_tokens, seed=..., coefs=...)`` contract as
    the other backends; ``coefs[i]`` multiplies the raw class-gap vector for
    conversation *i* (e.g. the 2026-09-22 steered arm used ±8)."""

    def __init__(self, model_id: str, layer: int, vector, *,
                 tensor_parallel_size: int | None = None, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.90):
        vec, vec_layer, vec_model = load_steering_vector(vector)
        if vec_layer != layer:
            raise ValueError(f"steering vector was trained at layer {vec_layer}, "
                             f"but LMSteering is set to layer {layer}")
        if vec_model != model_id:
            raise ValueError(f"steering vector was trained on {vec_model!r}, "
                             f"but LMSteering runs {model_id!r}")
        self.layer, self.vec = layer, vec
        import os
        # apply_model sends callables to the workers via cloudpickle, which
        # vLLM gates behind this env var (local single-user engine: fine).
        # Must be set before the engine spawns so workers inherit it.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        super().__init__(model_id, tensor_parallel_size, max_model_len,
                         gpu_memory_utilization, enforce_eager=True)
        self.llm.apply_model(partial(_install_hook, layer=layer, vec=vec))

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> "LMSteering":
        if not cfg.steer_path:
            raise ValueError("cfg.steer_path is not set")
        obj = cls(cfg.model_id, cfg.layer, cfg.steer_path,
                  tensor_parallel_size=cfg.tensor_parallel_size,
                  max_model_len=cfg.max_model_len)
        obj.gen_kwargs = dict(temperature=cfg.temperature, top_p=cfg.top_p)
        obj.force_close_tokens = cfg.force_close_tokens
        return obj

    def _generate_texts(self, texts, max_new_tokens, seed, coefs):
        from vllm import SamplingParams

        if coefs is None:
            coefs = [0.0] * len(texts)
        if len(coefs) != len(texts):
            raise ValueError(f"{len(coefs)} coefs for {len(texts)} conversations")
        outs = [""] * len(texts)
        try:
            for c in dict.fromkeys(coefs):           # one engine pass per distinct coef
                idxs = [i for i, k in enumerate(coefs) if k == c]
                self.llm.apply_model(partial(_set_coef, layer=self.layer, coef=float(c)))
                sps = [SamplingParams(max_tokens=max_new_tokens,
                                      seed=seed * 100_003 + i,
                                      **(self.gen_kwargs or GEN_KWARGS))
                       for i in idxs]
                res = self.llm.generate([texts[i] for i in idxs], sps)
                for i, o in zip(idxs, res):
                    outs[i] = o.outputs[0].text
        finally:
            self.llm.apply_model(partial(_set_coef, layer=self.layer, coef=0.0))
        return outs
