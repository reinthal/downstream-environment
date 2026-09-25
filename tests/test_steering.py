"""LMSteering vector-loading / DoM-only enforcement (no vLLM, no GPU).

LMSteering validates the vector BEFORE building the engine, so the rejection
paths are testable in the probe env where vllm is not installed."""
import numpy as np
import pytest

from conftest import fit_probe

from linear_probes import DIFF_OF_MEANS, LOGISTIC_REGRESSION
from rollouts.steering import LMSteering, load_steering_vector

D = 16


def _steer_npz(tmp_path, layer=16, model_id="test/model"):
    path = tmp_path / "steer.npz"
    np.savez(path, steer_vec=np.arange(D, dtype=float), layer=layer, model_id=model_id)
    return path


def test_dom_probe_instance_accepted():
    dom = fit_probe("test/model", 16, DIFF_OF_MEANS)
    vec, layer, model_id = load_steering_vector(dom)
    assert layer == 16 and model_id == "test/model"
    np.testing.assert_allclose(vec, dom.raw_diff)
    assert np.argmax(np.abs(vec)) == 0            # class gap lives on dim 0


def test_dom_probe_npz_accepted(tmp_path):
    dom = fit_probe("test/model", 16, DIFF_OF_MEANS)
    path = dom.save(str(tmp_path / "dom.npz"))
    vec, layer, model_id = load_steering_vector(path)
    np.testing.assert_allclose(vec, dom.raw_diff)


def test_steer_npz_accepted(tmp_path):
    vec, layer, model_id = load_steering_vector(_steer_npz(tmp_path))
    assert (layer, model_id) == (16, "test/model")


def test_lr_probe_rejected(tmp_path):
    lr = fit_probe("test/model", 16, LOGISTIC_REGRESSION)
    with pytest.raises(ValueError, match="diff-of-means"):
        load_steering_vector(lr)
    with pytest.raises(ValueError, match="diff-of-means"):
        load_steering_vector(lr.save(str(tmp_path / "lr.npz")))


def test_legacy_dom_npz_without_raw_diff_rejected(tmp_path):
    dom = fit_probe("test/model", 16, DIFF_OF_MEANS)
    path = tmp_path / "old.npz"
    np.savez(path, kind=DIFF_OF_MEANS, mu=dom.mu, sd=dom.sd, w=dom.w, b=dom.b,
             model_id="test/model", layer=16, lora_id="")
    with pytest.raises(ValueError, match="raw_diff"):
        load_steering_vector(path)


def test_bare_vector_rejected():
    with pytest.raises(TypeError, match="provenance"):
        load_steering_vector(np.ones(D))


def test_lmsteering_validates_before_engine(tmp_path):
    """Layer/model mismatches raise before any vLLM import happens."""
    with pytest.raises(ValueError, match="layer 16"):
        LMSteering("test/model", 3, _steer_npz(tmp_path))
    with pytest.raises(ValueError, match="trained on"):
        LMSteering("other/model", 16, _steer_npz(tmp_path))
