"""Minimal linear-probe training on residual-stream activations, local or NDIF."""
from .backends import ActivationBackend, LocalBackend, NDIFBackend, make_backend
from .config import DIFF_OF_MEANS, LOGISTIC_REGRESSION, ProbeConfig
from .probes import DiffOfMeansProbe, LinearProbe, LogisticRegressionProbe, make_probe
from .train import evaluate, train_probe

__all__ = [
    "ActivationBackend", "LocalBackend", "NDIFBackend", "make_backend",
    "DIFF_OF_MEANS", "LOGISTIC_REGRESSION", "ProbeConfig",
    "DiffOfMeansProbe", "LinearProbe", "LogisticRegressionProbe", "make_probe",
    "evaluate", "train_probe",
]
