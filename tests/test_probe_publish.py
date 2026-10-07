"""CPU tests for serving/publish.py: head folding, weight remap, config surgery."""
import numpy as np
import pytest

from linear_probes.config import ProbeConfig
from linear_probes.probes import make_probe
from serving.publish import ARCHITECTURE, build_config, fold_probe_head, remap_weight_name

rng = np.random.default_rng(0)


def _probe(d=16):
    p = make_probe(ProbeConfig(model_id="m", layer=3))
    p.mu = rng.normal(size=d)
    p.sd = rng.uniform(0.5, 2.0, size=d)
    p.w = rng.normal(size=d)
    p.b = 0.7
    return p


def test_fold_affine_matches_predict_proba():
    p = _probe()
    W, b = fold_probe_head(p)
    assert W.shape == (1, 16) and b.shape == (1,)
    assert W.dtype == np.float32 and b.dtype == np.float32
    X = rng.normal(size=(32, 16))
    z = X @ W[0].astype(np.float64) + float(b[0])
    got = 1.0 / (1.0 + np.exp(-z))
    np.testing.assert_allclose(got, p.predict_proba(X), rtol=0, atol=1e-6)


def test_remap_keeps_backbone_up_to_layer():
    L = 16
    assert (remap_weight_name("model.language_model.embed_tokens.weight", L)
            == "model.embed_tokens.weight")
    assert (remap_weight_name("model.language_model.layers.0.mlp.down_proj.weight", L)
            == "model.layers.0.mlp.down_proj.weight")
    assert (remap_weight_name("model.language_model.layers.16.linear_attn.in_proj_qkv.weight", L)
            == "model.layers.16.linear_attn.in_proj_qkv.weight")
    # text-only (no VLM wrapper) layout passes through
    assert remap_weight_name("model.layers.2.self_attn.q_proj.weight", L) \
        == "model.layers.2.self_attn.q_proj.weight"


def test_remap_drops_unused_tensors():
    L = 16
    for name in ("lm_head.weight",
                 "mtp.fc.weight",
                 "mtp.layers.0.mlp.down_proj.weight",
                 "model.visual.patch_embed.proj.weight",
                 "model.language_model.norm.weight",
                 "model.language_model.layers.17.mlp.down_proj.weight",
                 "model.language_model.layers.63.input_layernorm.weight"):
        assert remap_weight_name(name, L) is None, name


def test_remap_rejects_unknown_names():
    with pytest.raises(ValueError):
        remap_weight_name("decoder.layers.0.weight", 16)
    with pytest.raises(ValueError):
        remap_weight_name("model.language_model.mystery.weight", 16)


def _src_config(n_layers=64):
    # trimmed copy of the real Qwen3.5-27B config structure
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": n_layers,
            "full_attention_interval": 4,
            "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                            for i in range(n_layers)],
            "mtp_num_hidden_layers": 1,
            "rms_norm_eps": 1e-06,
            "dtype": "bfloat16",
            "rope_parameters": {"mrope_interleaved": True,
                                "mrope_section": [11, 11, 10],
                                "rope_type": "default", "rope_theta": 10000000,
                                "partial_rotary_factor": 0.25},
        },
        "vision_config": {"depth": 27},
    }


def test_config_surgery():
    meta = {"kind": "logistic_regression", "layer": 16}
    out = build_config(_src_config(), 16, meta)
    assert out["architectures"] == [ARCHITECTURE]
    assert out["model_type"] == "qwen3_5_text"
    assert out["num_hidden_layers"] == 17
    assert len(out["layer_types"]) == 17
    assert [i for i, t in enumerate(out["layer_types"]) if t == "full_attention"] \
        == [3, 7, 11, 15]
    assert out["mtp_num_hidden_layers"] == 0
    assert out["id2label"] == {"0": "deceptive"}
    assert out["probe_meta"] == meta
    assert "vision_config" not in out
    assert "text_config" not in out
    assert "mrope_section" not in out["rope_parameters"]
    assert "mrope_interleaved" not in out["rope_parameters"]
    assert out["rope_parameters"]["partial_rotary_factor"] == 0.25


def test_config_surgery_rejects_layer_out_of_range():
    with pytest.raises(ValueError):
        build_config(_src_config(n_layers=8), 16, {})
