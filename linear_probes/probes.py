"""Linear probes on residual-stream activations.

Both probes follow Goldowsky-Dill et al. (2025), "Detecting Strategic Deception
Using Linear Probes" (arXiv:2502.03407): a probe is one direction ``w`` (plus
bias ``b``) in activation space, read as ``p(deceptive) = sigmoid(x_std @ w + b)``
where ``x_std`` is the (optionally standardized) activation.

  * :class:`DiffOfMeansProbe` — ``w`` is proportional to
    ``mean(deceptive) - mean(honest)``; bias and scale are set so the midpoint
    of the class means maps to p = 0.5 and training projections have unit spread.
  * :class:`LogisticRegressionProbe` — ``w``, ``b`` fit by L2-regularized
    logistic regression (lambda = ``config.reg_coeff``) on standardized
    activations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import DIFF_OF_MEANS, LOGISTIC_REGRESSION, ProbeConfig


class LinearProbe(ABC):
    """Shared linear read-out: subclasses only decide how ``w``/``b`` are fit."""

    kind: str

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.mu = None    # (d,) standardizer mean
        self.sd = None    # (d,) standardizer std
        self.w = None     # (d,) direction, in standardized space
        self.b = 0.0

    @abstractmethod
    def fit(self, X, y) -> "LinearProbe":
        """``X``: (N, d) activations; ``y``: (N,) labels, 1 = deceptive."""

    def _fit_standardizer(self, X: np.ndarray) -> np.ndarray:
        if self.config.normalize:
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        else:
            self.mu, self.sd = np.zeros(X.shape[1]), np.ones(X.shape[1])
        return (X - self.mu) / self.sd

    def predict_proba(self, X) -> np.ndarray:
        Xs = (np.asarray(X, dtype=float) - self.mu) / self.sd
        z = Xs @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(X) >= threshold

    def save(self, path: str) -> str:
        np.savez(path, kind=self.kind, mu=self.mu, sd=self.sd, w=self.w, b=self.b,
                 model_id=self.config.model_id, layer=self.config.layer,
                 lora_id=self.config.lora_id or "")
        return path

    @staticmethod
    def load(path: str) -> "LinearProbe":
        z = np.load(path, allow_pickle=False)
        config = ProbeConfig(model_id=str(z["model_id"]), layer=int(z["layer"]),
                             lora_id=str(z["lora_id"]) or None, probe_type=str(z["kind"]))
        probe = make_probe(config)
        probe.mu, probe.sd, probe.w, probe.b = z["mu"], z["sd"], z["w"], float(z["b"])
        return probe


class DiffOfMeansProbe(LinearProbe):
    """Difference-of-means direction: ``w`` points from the honest-class mean to
    the deceptive-class mean; the midpoint between the projected class means is
    the p = 0.5 decision boundary."""

    kind = DIFF_OF_MEANS

    def fit(self, X, y) -> "DiffOfMeansProbe":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(bool)
        Xs = self._fit_standardizer(X)
        d = Xs[y].mean(0) - Xs[~y].mean(0)
        p = Xs @ d
        mid = (p[y].mean() + p[~y].mean()) / 2.0
        scale = p.std() + 1e-9
        self.w, self.b = d / scale, -mid / scale
        return self


class LogisticRegressionProbe(LinearProbe):
    """L2-regularized logistic regression on standardized activations."""

    kind = LOGISTIC_REGRESSION

    def fit(self, X, y) -> "LogisticRegressionProbe":
        from sklearn.linear_model import LogisticRegression

        Xs = self._fit_standardizer(np.asarray(X, dtype=float))
        clf = LogisticRegression(C=1.0 / self.config.reg_coeff, max_iter=2000)
        clf.fit(Xs, np.asarray(y).astype(int))
        self.w, self.b = clf.coef_[0], float(clf.intercept_[0])
        return self


def make_probe(config: ProbeConfig) -> LinearProbe:
    probes = {DIFF_OF_MEANS: DiffOfMeansProbe, LOGISTIC_REGRESSION: LogisticRegressionProbe}
    if config.probe_type not in probes:
        raise ValueError(f"unknown probe_type {config.probe_type!r}; expected one of {sorted(probes)}")
    return probes[config.probe_type](config)
