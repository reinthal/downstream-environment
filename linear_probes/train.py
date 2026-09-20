"""Train and evaluate a probe through an injected backend."""
from __future__ import annotations

import numpy as np

from .backends import ActivationBackend
from .config import ProbeConfig
from .probes import LinearProbe, make_probe


def train_probe(config: ProbeConfig, backend: ActivationBackend,
                conversations: list[list[dict]], labels) -> LinearProbe:
    """Collect activations via ``backend`` (local or NDIF — the caller decides)
    and fit the probe named by ``config.probe_type``. ``labels``: 1 = deceptive."""
    X = backend.collect(conversations)
    return make_probe(config).fit(X, np.asarray(labels))


def evaluate(probe: LinearProbe, X, y) -> dict:
    """Accuracy, balanced accuracy, and AUROC of ``probe`` on held-out
    activations ``X`` / labels ``y``."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y = np.asarray(y).astype(int)
    p = probe.predict_proba(X)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
    }
