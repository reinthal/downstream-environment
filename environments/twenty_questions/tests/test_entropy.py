import asyncio
import json
import math

from types import SimpleNamespace as NS
import httpx
import pytest

from twenty_questions.concepts import ConceptSet
from twenty_questions.entropy import EntropyConfig, analyse, posterior_trace


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


class FakeJev:
    """system_one(state={"question"}, questions={cK: Noul}) -> p_yes(question, concept)."""
    def __init__(self, p_yes):
        self.p_yes, self.requests = p_yes, []

    async def system_one(self, state, questions):
        self.requests.append(len(questions))
        answers = {}
        for k, q in questions.items():
            concept = q.instructions.split('the secret is "')[1].split('"')[0]
            answers[k] = NS(noul=self.p_yes(state["question"], concept))
        return NS(answers=answers, model="jev-1.13.0", usage=NS(input_tokens=10 * len(questions)))


def test_analyse_batches_and_caches(tmp_path):
    cfg = EntropyConfig(concept_set="concepts/things_n256_s0.json", concepts_per_request=100, concurrency=2)
    secret = ConceptSet.load(cfg.concept_path()).words[0]

    def p_yes(q, c):  # first half of the alphabet answers yes
        return 0.99 if c < "m" else 0.01

    games = [{"game_id": "g0", "objective": "competitive", "secret": secret,
              "exchanges": [{"turn": 1, "kind": "question", "text": "Early letter?", "answer": " yes", "clue": None},
                            {"turn": 2, "kind": "final_answer", "text": "x", "answer": None, "clue": None}]}]
    jev = FakeJev(p_yes)
    res = asyncio.run(analyse(cfg, games, tmp_path / "cache.jsonl", client=jev))
    assert jev.requests == [100, 100, 56]              # 256 concepts in chunks
    g = res["games"][0]
    assert g["n_questions"] == 1 and g["steps"][0]["H"] == pytest.approx(8.0)
    cached = [json.loads(line) for line in (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert len(cached) == 256
    assert g["steps"][1]["n_consistent"] == sum(r["concept"] < "m" for r in cached)
    assert g["steps"][1]["secret_consistent"] and 0 < g["steps"][1]["bits"] < 1.1
    # resume: everything cached, no requests
    res2 = asyncio.run(analyse(cfg, games, tmp_path / "cache.jsonl", client=FakeJev(lambda q, c: pytest.fail("re-requested"))))
    assert res2["games"][0]["steps"] == g["steps"]


def test_definitions_fill_the_noul(tmp_path):
    from twenty_questions.entropy import JevScorer
    cfg = EntropyConfig(concept_set="concepts/things_n256_s0.json")
    cs = ConceptSet.load(cfg.concept_path())
    q = JevScorer(cfg, cs, tmp_path / "c.jsonl").questions([cs.words[0]])["c0"]
    assert f'"{cs.words[0]}" ({cs.definitions[0]})' in q.instructions


class FakeJevKinds(FakeJev):
    async def system_one(self, state, questions):
        (kind, text), = state.items()
        self.requests.append((kind, len(questions)))
        answers = {k: NS(noul=self.p_yes(kind, text, q.instructions.split('the secret is "')[1].split('"')[0]))
                   for k, q in questions.items()}
        return NS(answers=answers, model="jev-1.13.0", usage=NS(input_tokens=1))


def test_clues_are_observations(tmp_path):
    cfg = EntropyConfig(concept_set="concepts/things_n256_s0.json", concepts_per_request=256, include_clues=True)
    words = ConceptSet.load(cfg.concept_path()).words
    secret = words[0]

    def p(kind, text, c):
        if kind == "question":
            return 0.5                                    # uninformative question
        return 0.99 if c < "m" else 0.01                  # the clue fits the first half only

    games = [{"game_id": "g", "objective": "competitive", "secret": secret, "exchanges": [
        {"turn": 1, "kind": "question", "text": "Q?", "answer": " no", "clue": "Early letter."},
        {"turn": 2, "kind": "clue_request", "text": "", "answer": None, "clue": "Early letter."},
        {"turn": 3, "kind": "final_answer", "text": "x", "answer": None, "clue": None}]}]
    jev = FakeJevKinds(p)
    res = asyncio.run(analyse(cfg, games, tmp_path / "c.jsonl", client=jev))
    g = res["games"][0]
    assert [s["kind"] for s in g["steps"]] == [None, "question", "clue", "clue"]
    assert g["steps"][1]["bits"] == pytest.approx(0.0, abs=1e-9)          # the question removes nothing
    assert g["steps"][2]["bits"] > 0.9 and g["steps"][2]["secret_consistent"]
    assert res["n_questions"] == 1 and res["n_clues"] == 2
    assert sorted(set(jev.requests)) == [("clue", 256), ("question", 256)]   # repeated clue text scored once
    rows = [json.loads(x) for x in (tmp_path / "c.jsonl").read_text().splitlines()]
    assert sum(r.get("kind") == "clue" for r in rows) == 256 and sum("kind" not in r for r in rows) == 256
    # question-only config reading the same cache: clue rows are ignored, not an error
    res_q = asyncio.run(analyse(EntropyConfig(concept_set=cfg.concept_set), games, tmp_path / "c.jsonl",
                                client=FakeJevKinds(lambda *a: pytest.fail("re-requested"))))
    assert res_q["n_clues"] == 0 and len(res_q["games"][0]["steps"]) == 2


def test_openrouter_backend_reads_first_token_logprobs(tmp_path):
    import math as m
    cfg = EntropyConfig(backend="openrouter", model="qwen/qwen3.5-9b", openrouter_provider="Venice",
                        concept_set="concepts/things_n256_s0.json", include_clues=True, concurrency=4)
    words = ConceptSet.load(cfg.concept_path()).words
    secret = words[0]
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert body["max_tokens"] == 1 and body["reasoning"] == {"enabled": False}
        assert body["provider"] == {"order": ["Venice"], "allow_fallbacks": False}
        content = body["messages"][0]["content"]
        concept = content.split('the secret is "')[1].split('"')[0]
        if concept == words[1]:                         # one pair answers something else entirely
            top = [{"token": "maybe", "logprob": -0.1}]
        else:
            p = 0.9 if concept < "m" else 0.2
            top = [{"token": "Yes", "logprob": m.log(p * 0.5)}, {"token": "yes", "logprob": m.log(p * 0.5)},
                   {"token": "no", "logprob": m.log(1 - p)}]
        return httpx.Response(200, json={"provider": "Venice", "usage": {"prompt_tokens": 60, "cost": 1e-6},
                                         "choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]})

    games = [{"game_id": "g", "objective": "competitive", "secret": secret, "exchanges": [
        {"turn": 1, "kind": "question", "text": "Early?", "answer": " yes", "clue": "Early letter."}]}]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="2 pairs had no yes/no token"):
        asyncio.run(analyse(cfg, games, tmp_path / "q.jsonl", client=client))
    rows = [json.loads(x) for x in (tmp_path / "q.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 256 and {r["kind"] for r in rows} == {"question", "clue"}
    good = next(r for r in rows if r["concept"] == words[0] and r["kind"] == "question")
    assert good["p_yes"] == pytest.approx(0.9) and good["mass"] == pytest.approx(1.0)
    assert sum(r["p_yes"] is None for r in rows) == 2         # that concept, for the question and the clue
    assert "Clue: Early letter." in next(b for b in seen if "Clue:" in b["messages"][0]["content"])["messages"][0]["content"]


def test_precomputed_backend_reads_and_requires_every_pair(tmp_path):
    cfg = EntropyConfig(backend="precomputed", model="Qwen/Qwen3.5-9B", concept_set="concepts/things_n256_s0.json")
    words = ConceptSet.load(cfg.concept_path()).words
    rows = [{"model": "Qwen/Qwen3.5-9B", "provider": "vllm", "kind": "question", "prompt": cfg.llm_question_prompt,
             "text": "Early?", "concept": c, "p_yes": 0.9 if c < "m" else 0.1, "mass": 1.0} for c in words]
    path = tmp_path / "scores.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    games = [{"game_id": "g", "objective": "competitive", "secret": words[0], "exchanges": [
        {"turn": 1, "kind": "question", "text": "Early?", "answer": " yes", "clue": None}]}]
    res = asyncio.run(analyse(cfg, games, path))
    assert res["games"][0]["steps"][1]["secret_consistent"] and res["games"][0]["steps"][1]["bits"] > 0.5
    games[0]["exchanges"].append({"turn": 2, "kind": "question", "text": "Not scored?", "answer": " no", "clue": None})
    with pytest.raises(ValueError, match="pairs missing"):
        asyncio.run(analyse(cfg, games, path))
