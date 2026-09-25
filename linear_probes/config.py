"""Run configuration for probe training and full experiments.

Everything a training run needs — which model to read, where in the network to
read it, and which probe to fit — lives in one dataclass. The runtime builds it
once and injects it into both the activation backend and the probe factory.

:class:`ExperimentConfig` extends :class:`ProbeConfig` with everything else an
experiment injects (generation, steering, seeds, output dir) and JSON
round-trips, so each ``experiments/<date>/`` folder carries one ``config.json``
that fully reproduces its run. Experiments needing extra hyperparameters
subclass it and add fields; ``load`` works on the subclass.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DIFF_OF_MEANS = "diff_of_means"
LOGISTIC_REGRESSION = "logistic_regression"


@dataclass
class ProbeConfig:
    # model parameters
    model_id: str                 # HF repo id of the model whose activations are read
    layer: int                    # decoder block index; the probe reads that block's output (residual stream)
    lora_id: str | None = None    # optional adapter to attach on top of model_id
    batch_size: int = 8
    max_len: int = 256            # token budget per conversation (left-trimmed so the response survives)

    # local backend only (the NDIF backend runs on the server's placement)
    device: str = "auto"
    dtype: str = "bfloat16"
    truncate_layers: bool = True  # load only decoder blocks 0..layer — activations at <= layer are
                                  # unchanged and a 70B probed at layer 22 needs ~1/4 of the weights

    # probe
    probe_type: str = LOGISTIC_REGRESSION   # DIFF_OF_MEANS | LOGISTIC_REGRESSION
    reg_coeff: float = 10.0       # L2 strength lambda for the LR probe (sklearn C = 1/lambda)
    normalize: bool = True        # standardize activations (per-dim mean/std) before fitting


@dataclass
class ExperimentConfig(ProbeConfig):
    """One experiment's full hyperparameter surface. ``layer`` doubles as the
    probe read-out layer and the steering layer (they are the same direction's
    home). Save on run start (``cfg.save(out_dir)``) so the folder is
    self-describing."""

    # trained artifacts (paths, usually into experiments/<date>/probes/)
    probe_paths: dict[str, str] = field(default_factory=dict)  # probe name -> npz
    steer_path: str | None = None    # DoM steering vector npz (see LMSteering)
    steer_coef: float = 0.0          # per-conversation coefs are caller-side; this is the magnitude

    # generation (rollouts.generation / LMSteering)
    gen_backend: str = "vllm"        # "vllm" | "hf"
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 1800
    force_close_tokens: int = 220    # bounded continuation after an unclosed <think>
    tensor_parallel_size: int | None = None   # None -> all visible GPUs
    max_model_len: int = 8192        # vLLM context budget
    gen_chunk: int = 25              # HF generation batch (25 is the long-context ceiling)

    # experiment bookkeeping
    seed: int = 0
    seed_groups: int = 1
    dataset_sizes: dict[str, int] = field(default_factory=dict)
    out_dir: str = "."

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> "ExperimentConfig":
        raw = json.loads(text)
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config fields {sorted(unknown)} for {cls.__name__}")
        return cls(**raw)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_json(Path(path).read_text())

    def save(self, out_dir: str | Path | None = None) -> Path:
        """Write ``config.json`` into ``out_dir`` (default: ``self.out_dir``)."""
        out = Path(out_dir if out_dir is not None else self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "config.json"
        path.write_text(self.to_json() + "\n")
        return path
