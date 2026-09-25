"""LMProbe on the tiny offline checkpoint (CPU, no network)."""
import numpy as np
import pytest

from conftest import TINY_LAYERS, fit_probe

from linear_probes import (DIFF_OF_MEANS, LOGISTIC_REGRESSION, ExperimentConfig,
                           LMProbe, load_truncated_decoder)

LAYER = 1


def _lm_probe(tiny_model_dir, probe=None, **kw):
    probe = probe or fit_probe(tiny_model_dir, LAYER, LOGISTIC_REGRESSION)
    kw.setdefault("dtype", "float32")
    kw.setdefault("device", "cpu")
    kw.setdefault("batch_size", 2)
    kw.setdefault("max_len", 64)
    return LMProbe(tiny_model_dir, LAYER, probe, **kw)


def test_truncation_drops_layers_above_probe_layer(tiny_model_dir):
    model, _ = load_truncated_decoder(tiny_model_dir, LAYER, dtype="float32",
                                      device="cpu")
    assert len(model.model.layers) == LAYER + 1 < TINY_LAYERS


def test_scores_are_probabilities(tiny_model_dir, conversations):
    p = _lm_probe(tiny_model_dir).score(conversations)
    assert p.shape == (len(conversations),)
    assert ((0 <= p) & (p <= 1)).all()
    assert len(set(np.round(p, 6))) > 1        # distinct convs -> distinct scores


def test_score_matches_probe_on_collected_activations(tiny_model_dir, conversations):
    lmp = _lm_probe(tiny_model_dir)
    X = lmp.collect(conversations)
    np.testing.assert_allclose(lmp.score(conversations),
                               lmp.probe.predict_proba(X), rtol=1e-5)


def test_truncated_matches_full_model(tiny_model_dir, conversations):
    """Blocks above the probe layer never feed activations at <= layer."""
    trunc = _lm_probe(tiny_model_dir)
    full = _lm_probe(tiny_model_dir, truncate_layers=False)
    np.testing.assert_allclose(trunc.score(conversations),
                               full.score(conversations), rtol=1e-4)


def test_multi_span_collect(tiny_model_dir, conversations):
    lmp = _lm_probe(tiny_model_dir)
    X = lmp.collect(conversations, spans=("full", "public"))
    assert X.shape == (len(conversations), 2, 16)
    assert not np.allclose(X[:, 0], X[:, 1])   # think tokens only pool into "full"
    with pytest.raises(ValueError, match="unknown spans"):
        lmp.collect(conversations, spans=("private",))


def test_layer_mismatch_rejected(tiny_model_dir):
    probe = fit_probe(tiny_model_dir, layer=3, probe_type=LOGISTIC_REGRESSION)
    with pytest.raises(ValueError, match="layer"):
        LMProbe(tiny_model_dir, LAYER, probe)


def test_from_config_and_with_probe(tiny_model_dir, conversations, tmp_path):
    lr = fit_probe(tiny_model_dir, LAYER, LOGISTIC_REGRESSION)
    dom = fit_probe(tiny_model_dir, LAYER, DIFF_OF_MEANS)
    lr_path, dom_path = tmp_path / "lr.npz", tmp_path / "dom.npz"
    lr.save(str(lr_path)), dom.save(str(dom_path))

    cfg = ExperimentConfig(model_id=tiny_model_dir, layer=LAYER,
                           probe_paths={"lr": str(lr_path), "dom": str(dom_path)},
                           batch_size=2, max_len=64, dtype="float32", device="cpu")
    with pytest.raises(ValueError, match="pass probe="):
        LMProbe.from_config(cfg)                    # ambiguous: two probes
    lmp = LMProbe.from_config(cfg, probe="lr")
    lmp2 = lmp.with_probe(str(dom_path))
    p1, p2 = lmp.score(conversations), lmp2.score(conversations)
    assert lmp2._model is lmp._model               # shared weights
    assert p1.shape == p2.shape == (len(conversations),)
