"""GPU smoke: LMProbe end-to-end on real Qwen weights (2 conversations).

Loads Qwen3.5-9B truncated at layer 16 (~5 GB), fits a throwaway DoM probe on
its own activations, and checks the score path. Requires CUDA *and* an
explicit CUDA_VISIBLE_DEVICES (hardware etiquette: pick a free GPU, don't
shard onto whatever another job is using):

    CUDA_VISIBLE_DEVICES=7 uv run --no-sync pytest tests/test_gpu_smoke.py
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "CUDA_VISIBLE_DEVICES" not in os.environ,
    reason="GPU smoke needs CUDA and an explicit CUDA_VISIBLE_DEVICES")

MODEL_ID = "Qwen/Qwen3.5-9B"
LAYER = 16


def test_lm_probe_scores_real_model():
    from linear_probes import LMProbe, ProbeConfig, make_probe
    from rollouts.generation import decoder_layers

    convs = [
        [{"role": "user", "content": "Are you being honest with me?"},
         {"role": "assistant", "content": "Yes, completely honest.",
          "reasoning_content": "I should reassure them."}],
        [{"role": "user", "content": "What is 2 + 2?"},
         {"role": "assistant", "content": "4.", "reasoning_content": "Trivial."}],
    ]
    cfg = ProbeConfig(model_id=MODEL_ID, layer=LAYER, probe_type="diff_of_means")
    probe = make_probe(cfg).fit(np.random.default_rng(0).normal(size=(8, 4096)),
                                [0, 1] * 4)
    lmp = LMProbe(MODEL_ID, LAYER, probe, batch_size=2, max_len=512)
    model, _ = lmp._ensure_model()
    n_blocks = len(list(decoder_layers(model).children()))
    assert n_blocks == LAYER + 1                 # truncated at the probe layer
    p = lmp.score(convs)
    assert p.shape == (2,) and ((0 <= p) & (p <= 1)).all()
