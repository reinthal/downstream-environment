"""Minimal linear-probe training on residual-stream activations, local or NDIF."""
from .activations import collect_activations, encode_with_spans, load_truncated_decoder
from .backends import ActivationBackend, LocalBackend, NDIFBackend, make_backend
from .config import DIFF_OF_MEANS, LOGISTIC_REGRESSION, ExperimentConfig, ProbeConfig
from .lm_probe import LMProbe
from .probes import DiffOfMeansProbe, LinearProbe, LogisticRegressionProbe, make_probe
from .train import evaluate, train_probe

__all__ = [
    "ActivationBackend", "LocalBackend", "NDIFBackend", "make_backend",
    "DIFF_OF_MEANS", "LOGISTIC_REGRESSION", "ExperimentConfig", "ProbeConfig",
    "DiffOfMeansProbe", "LinearProbe", "LMProbe", "LogisticRegressionProbe", "make_probe",
    "collect_activations", "encode_with_spans", "load_truncated_decoder",
    "evaluate", "train_probe",
]
