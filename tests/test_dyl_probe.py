"""DYLProbe (follow-up "did you lie" probe) on the tiny offline checkpoint."""
import numpy as np
import pytest
from transformers import AutoTokenizer

from conftest import fit_probe

from linear_probes import DYLProbe, LOGISTIC_REGRESSION
from linear_probes.dyl_probe import (FOLLOWUP_ANSWER, FOLLOWUP_QUESTION,
                                     encode_no_span, with_followup)

LAYER = 1


def _dyl_probe(tiny_model_dir, **kw):
    probe = fit_probe(tiny_model_dir, LAYER, LOGISTIC_REGRESSION)
    kw.setdefault("dtype", "float32")
    kw.setdefault("device", "cpu")
    kw.setdefault("batch_size", 2)
    kw.setdefault("max_len", 64)
    return DYLProbe(tiny_model_dir, LAYER, probe, **kw)


def test_with_followup_appends_the_fixed_pair(conversations):
    conv = with_followup(conversations[0])
    assert conv[:-2] == conversations[0]
    assert conv[-2] == {"role": "user", "content": FOLLOWUP_QUESTION}
    assert conv[-1] == {"role": "assistant", "content": FOLLOWUP_ANSWER}


def test_with_followup_rejects_unfinished_dialogue():
    with pytest.raises(ValueError, match="assistant"):
        with_followup([{"role": "user", "content": "hello there"}])


def test_no_span_covers_exactly_the_no_tokens(tiny_model_dir, conversations):
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    no_ids = tok(FOLLOWUP_ANSWER, add_special_tokens=False)["input_ids"]
    for msgs in conversations:
        ids, (s, e) = encode_no_span(tok, with_followup(msgs))
        assert ids[s:e] == no_ids
        assert e <= len(ids)


def test_left_trim_keeps_the_no_span(tiny_model_dir, conversations):
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    conv = with_followup(conversations[0])
    full_ids, (s, e) = encode_no_span(tok, conv)
    ids, (ts, te) = encode_no_span(tok, conv, max_len=e - s + 3)
    assert len(ids) == e - s + 3
    assert ids[ts:te] == full_ids[s:e]


def test_scores_are_probabilities(tiny_model_dir, conversations):
    p = _dyl_probe(tiny_model_dir).score(conversations)
    assert p.shape == (len(conversations),)
    assert ((0 <= p) & (p <= 1)).all()
    assert len(set(np.round(p, 6))) > 1        # distinct contexts -> distinct "No"s


def test_score_matches_probe_on_collected_activations(tiny_model_dir, conversations):
    dyl = _dyl_probe(tiny_model_dir)
    X = dyl.collect(conversations)
    assert X.shape == (len(conversations), 16)
    np.testing.assert_allclose(dyl.score(conversations),
                               dyl.probe.predict_proba(X), rtol=1e-5)


def test_with_probe_shares_model_and_stays_a_dyl_probe(tiny_model_dir, conversations,
                                                       tmp_path):
    dyl = _dyl_probe(tiny_model_dir)
    dyl.score(conversations[:1])               # force the lazy load
    path = tmp_path / "second.npz"
    fit_probe(tiny_model_dir, LAYER, LOGISTIC_REGRESSION).save(str(path))
    other = dyl.with_probe(str(path))
    assert isinstance(other, DYLProbe)
    assert other._model is dyl._model
    assert other.score(conversations).shape == (len(conversations),)
