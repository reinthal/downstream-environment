import asyncio
import json
import math

import httpx
import pytest

from twenty_questions.concepts import ConceptSet
from twenty_questions.entropy import EntropyConfig, OracleScorer, analyse, posterior_trace


def test_hard_halving_gives_one_bit_per_question():
    # 8 concepts, 3 questions that each split the consistent set in half
    prior = [1 / 8] * 8
    bits = [[1.0 if (c >> k) & 1 else 0.0 for c in range(8)] for k in range(3)]
    steps = posterior_trace(prior, [" yes", " no", " yes"], bits, secret_index=0b101)
    assert [round(s["H"], 9) for s in steps] == [3, 2, 1, 0]
    assert all(abs(s["bits"] - 1) < 1e-9 for s in steps[1:])
    assert steps[-1]["n_consistent"] == 1 and steps[-1]["rank_secret"] == 1
    assert steps[-1]["p_secret"] == pytest.approx(1.0)


def test_uninformative_question_gives_zero_bits():
    steps = posterior_trace([0.25] * 4, [" yes"], [[0.5] * 4], secret_index=None)
    assert steps[1]["bits"] == pytest.approx(0.0) and steps[1]["H"] == pytest.approx(2.0)


def test_rejects_unspaced_answer():
    with pytest.raises(ValueError):
        posterior_trace([0.5, 0.5], ["yes"], [[0.9, 0.1]], None)


def fake_vllm(p_yes):
    """Transport: /tokenize -> ids; /v1/completions -> prompt_logprobs."""
    YES, NO = 100, 200
    vocab = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            if "prompt" in body:
                return httpx.Response(200, json={"tokens": [YES if body["prompt"] == " yes" else NO]})
            text = body["messages"][0]["content"]
            key = vocab.setdefault(text, len(vocab) + 1000)
            assert body["continue_final_message"] and body["chat_template_kwargs"] == {"enable_thinking": False}
            return httpx.Response(200, json={"tokens": [1, key]})
        assert request.url.path == "/v1/completions" and body["prompt_logprobs"] == 0
        inv = {v: k for k, v in vocab.items()}
        choices = []
        for i, prompt in enumerate(body["prompt"]):
            text = inv[prompt[1]]
            q = text.split("Question: ")[1].split("\n")[0]
            c = text.split("The secret is: ")[1].split("\n")[0]
            p = p_yes(q, c) if prompt[-1] == YES else 1 - p_yes(q, c)
            choices.append({"index": i, "prompt_logprobs": [None, {}, {str(prompt[-1]): {"logprob": math.log(p)}}]})
        return httpx.Response(200, json={"choices": choices[::-1]})  # order must not matter
    return httpx.MockTransport(handler)


def test_analyse_batches_and_caches(tmp_path, monkeypatch):
    cfg = EntropyConfig(concept_set="concepts/things_n256_s0.json", batch_size=50, concurrency=3)
    secret = ConceptSet.load(cfg.concept_path()).words[0]

    def p_yes(q, c):  # "Is it an animal?"-like split: first half of the alphabet
        return 0.99 if c < "m" else 0.01

    transport = fake_vllm(p_yes)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=transport, **kw))
    games = [{"game_id": "g0", "objective": "honest", "secret": secret,
              "exchanges": [{"turn": 1, "kind": "question", "text": "Early letter?", "answer": " yes", "clue": None},
                            {"turn": 2, "kind": "final_answer", "text": "x", "answer": None, "clue": None}]}]
    res = asyncio.run(analyse(cfg, games, tmp_path / "cache.jsonl"))
    g = res["games"][0]
    assert g["n_questions"] == 1 and g["steps"][0]["H"] == pytest.approx(8.0)
    n_early = sum(1 for line in (tmp_path / "cache.jsonl").read_text().splitlines()
                  if json.loads(line)["concept"] < "m")
    assert g["steps"][1]["n_consistent"] == n_early and g["steps"][1]["secret_consistent"]
    assert 0 < g["steps"][1]["bits"] < 1.1
    assert len((tmp_path / "cache.jsonl").read_text().splitlines()) == 256
    # resume: everything cached, no requests needed
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(
        transport=httpx.MockTransport(lambda r: pytest.fail("cached pairs re-requested")), **kw))
    res2 = asyncio.run(analyse(cfg, games, tmp_path / "cache.jsonl"))
    assert res2["games"][0]["steps"] == g["steps"]
