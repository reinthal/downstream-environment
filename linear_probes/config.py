"""Run configuration for probe training.

Everything a training run needs — which model to read, where in the network to
read it, and which probe to fit — lives in one dataclass. The runtime builds it
once and injects it into both the activation backend and the probe factory.
"""
from __future__ import annotations

from dataclasses import dataclass

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

    # probe
    probe_type: str = LOGISTIC_REGRESSION   # DIFF_OF_MEANS | LOGISTIC_REGRESSION
    reg_coeff: float = 10.0       # L2 strength lambda for the LR probe (sklearn C = 1/lambda)
    normalize: bool = True        # standardize activations (per-dim mean/std) before fitting
