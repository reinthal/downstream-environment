"""Synthetic-data checks for the probe classes (no model, no network)."""
import numpy as np
import pytest

from linear_probes import (DIFF_OF_MEANS, LOGISTIC_REGRESSION, LinearProbe,
                           ProbeConfig, make_probe)
from linear_probes.train import evaluate


def _separable_data(n=200, d=32, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, d))
    X[y == 1, 0] += 3.0                      # class signal along one dimension
    return X, y


@pytest.mark.parametrize("probe_type", [DIFF_OF_MEANS, LOGISTIC_REGRESSION])
def test_fit_separates_classes(probe_type):
    X, y = _separable_data()
    config = ProbeConfig(model_id="test/model", layer=0, probe_type=probe_type)
    probe = make_probe(config).fit(X, y)
    metrics = evaluate(probe, X, y)
    assert metrics["auroc"] > 0.95
    assert metrics["balanced_accuracy"] > 0.9
    p = probe.predict_proba(X)
    assert p.shape == (len(y),) and (0 <= p).all() and (p <= 1).all()


@pytest.mark.parametrize("probe_type", [DIFF_OF_MEANS, LOGISTIC_REGRESSION])
def test_save_load_roundtrip(tmp_path, probe_type):
    X, y = _separable_data()
    config = ProbeConfig(model_id="test/model", layer=7, probe_type=probe_type)
    probe = make_probe(config).fit(X, y)
    path = probe.save(str(tmp_path / "probe.npz"))
    loaded = LinearProbe.load(path)
    assert type(loaded) is type(probe)
    assert loaded.config.layer == 7
    np.testing.assert_allclose(loaded.predict_proba(X), probe.predict_proba(X))


def test_unknown_probe_type():
    with pytest.raises(ValueError):
        make_probe(ProbeConfig(model_id="test/model", layer=0, probe_type="mlp"))
