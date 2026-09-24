"""Logic checks for rollouts.generation (no model, no network)."""
from rollouts.generation import CLOSE_CUE, GenerationBackend, make_backend, split_think

import pytest


def test_split_think_closed():
    reasoning, public = split_think("I am the Parsnip, blend in.\n</think>\n\nHello all!")
    assert reasoning == "I am the Parsnip, blend in."
    assert public == "Hello all!"


def test_split_think_unclosed_is_all_reasoning():
    reasoning, public = split_think("still thinking about who to accuse")
    assert public == ""
    assert "still thinking" in reasoning


def test_split_think_strips_open_tag():
    reasoning, public = split_think("<think>\nplan\n</think>\n\nSpeech.")
    assert reasoning == "plan"
    assert public == "Speech."


class _StubTok:
    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=True):
        return "PROMPT[" + conv[-1]["content"] + "]<think>\n"


class _FakeBackend(GenerationBackend):
    """First pass: conv 0 closes its think, conv 1 does not. The force-close
    continuation then supplies conv 1's public message."""

    def __init__(self):                       # no tokenizer download
        self.tok = _StubTok()
        self.n_force_closed = 0
        self.calls = []

    def _generate_texts(self, texts, max_new_tokens, seed, coefs):
        self.calls.append(list(texts))
        if len(self.calls) == 1:
            return ["ok\n</think>\n\nA speaks.", "rambling forever"]
        return ["B speaks."]                  # continuation after CLOSE_CUE


def test_generate_force_closes_unclosed_think():
    b = _FakeBackend()
    convs = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    raws = b.generate(convs, max_new_tokens=100)
    assert b.n_force_closed == 1
    assert split_think(raws[0]) == ("ok", "A speaks.")
    assert split_think(raws[1]) == (("rambling forever" + CLOSE_CUE.split("</think>")[0]).strip(),
                                    "B speaks.")
    # continuation prompt = rendered prompt + truncated output + cue
    assert b.calls[1][0] == "PROMPT[b]<think>\nrambling forever" + CLOSE_CUE


def test_make_backend_rejects_unknown():
    with pytest.raises(ValueError, match="unknown backend"):
        make_backend("triton", "some/model")
