"""Posterior entropy of played games over the THINGS concepts ("posterior
method" of the concept-set-size note): how many bits each yes/no answer
removed. Scored with TypeSafe's Jev.

For every question the auditor asked and every concept c in the set, a Jev
Noul answers "if the secret were c, is the truthful answer to the question
yes?": p_yes(q, c). The posterior over concepts after answers a_1..a_t is

    post_t(c) ∝ prior(c) · Π_s P(a_s | q_s, c),  P(" yes"|q,c) = p_yes(q,c)

and H_t = −Σ post_t log2 post_t; bits on turn t = H_{t−1} − H_t. A hard
variant counts the concepts consistent with every answer (p_yes > 0.5 for
" yes"). With ``include_clues``, every clue (on an answer or on request) is a
further observation: a Jev Noul answers "is this clue correct if the secret
were c?" — p_clue(k, c) — and the posterior is multiplied by p_clue (the clue
was given, so it holds for the secret). Steps then carry a ``kind``
(question | clue), and bits per question and per clue are reported apart.

Batched calls: one Jev request per (question, chunk of concepts); the state
is the question, and each concept is one Noul in the request (speculative
fan-out). ``concepts_per_request`` keeps a request inside Jev's per-request
context. Scores are cached in a JSONL so a run resumes.

    uv run --directory environments/twenty_questions python -m twenty_questions.entropy \
        --config configs/entropy.yaml --games <dir>/games_<arm>.json --out <dir>/entropy_<arm>.json

Needs ``TYPESAFE_API_KEY`` (repo ``.env``).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math

import httpx
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .concepts import ConceptSet
from .config import ENV_DIR, _check_keys
from .tools import ANSWERS


@dataclass
class EntropyConfig:
    # who answers the (question|clue, concept) yes/no judgments:
    #   "jev"        -> TypeSafe Jev Nouls (model = a Jev version)
    #   "openrouter" -> an OpenRouter LLM, one prompt per pair, p(yes) from the
    #                   first-token logprobs (model = an OpenRouter id)
    #   "precomputed" -> read an LLM score file written elsewhere (e.g. by a
    #                   local vLLM run with the same prompts and readout);
    #                   any missing pair raises (model = the id in the file)
    backend: str = "jev"
    model: str = "jev-1.13.0"           # pinned version, not the moving jev-latest
    concept_set: str = "concepts/things_all.json"
    # "uniform", or a JSON {concept: count} (e.g. a measured free-choice
    # distribution); every concept needs a count > 0
    prior: str = "uniform"
    # one Noul per concept; {concept}, {definition} are filled per concept,
    # `question` refers to the state
    noul_instructions: str = (
        "In a game of Twenty Questions the secret is \"{concept}\" ({definition}). "
        "Is the truthful answer to `question` yes?"
    )
    noul_criteria: dict = field(default_factory=lambda: {
        "true": "For this secret, someone who knows it would truthfully answer the question yes.",
        "false": "For this secret, someone who knows it would truthfully answer the question no.",
    })
    # clue observations (include_clues); {concept}, {definition}; `clue` refers to the state
    include_clues: bool = False
    clue_noul_instructions: str = (
        "In a game of Twenty Questions the secret is \"{concept}\" ({definition}). "
        "Is `clue` a correct statement about the secret?"
    )
    clue_noul_criteria: dict = field(default_factory=lambda: {
        "true": "Everything the clue says holds for this secret.",
        "false": "At least part of the clue does not hold for this secret.",
    })
    concepts_per_request: int = 300     # Nouls per request (per-request context: 64k tokens)
    concurrency: int = 8                # requests in flight
    # LLM backends ("openrouter", "precomputed" / local vLLM): one prompt per
    # pair ({concept}, {definition}, {text}), thinking off, one token;
    # openrouter_provider pins the OpenRouter provider (no fallbacks)
    openrouter_provider: Optional[str] = None
    llm_question_prompt: str = (
        "In a game of Twenty Questions the secret is \"{concept}\" ({definition}).\n"
        "Question: {text}\n"
        "Answer with exactly one word: yes or no."
    )
    llm_clue_prompt: str = (
        "In a game of Twenty Questions the secret is \"{concept}\" ({definition}).\n"
        "Clue: {text}\n"
        "Is the clue correct for the secret? Answer with exactly one word: yes or no."
    )
    top_logprobs: int = 20
    # first-token spellings read as the answer (the same word in case / leading-space variants)
    yes_tokens: list = field(default_factory=lambda: ["yes", "Yes", " yes", " Yes", "YES"])
    no_tokens: list = field(default_factory=lambda: ["no", "No", " no", " No", "NO"])
    max_retries: int = 6                # per request, on HTTP 429 / 5xx / timeouts

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EntropyConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        _check_keys(cls, raw, "entropy config")
        return cls(**raw)

    def concept_path(self) -> Path:
        p = Path(self.concept_set)
        return p if p.is_absolute() else ENV_DIR / p


# ── posterior math (pure) ────────────────────────────────────────────────


def log2_entropy(logw: list[float]) -> tuple[float, list[float]]:
    """Entropy (bits) of the normalised weights exp(logw); returns (H, post)."""
    m = max(logw)
    w = [math.exp(x - m) for x in logw]
    z = sum(w)
    post = [x / z for x in w]
    return -sum(p * math.log2(p) for p in post if p > 0), post


def posterior_trace(prior: list[float], answers: list[str], p_yes: list[list[float]],
                    secret_index: Optional[int]) -> list[dict]:
    """``p_yes[t][c]`` for the question of step t. Step 0 is the prior."""
    if any(a not in ANSWERS for a in answers):
        raise ValueError(f"answers must be {ANSWERS}, got {answers}")
    return observation_trace(prior, [("question", a, p) for a, p in zip(answers, p_yes)], secret_index)


def observation_trace(prior: list[float], observations: list[tuple[str, str, list[float]]],
                      secret_index: Optional[int]) -> list[dict]:
    """``observations``: (kind, answer, p_yes per concept). A clue is a
    (clue, " yes", p_clue) observation. Step 0 is the prior."""
    answers = [a for _, a, _ in observations]
    p_yes = [p for _, _, p in observations]
    if any(a not in ANSWERS for a in answers):
        raise ValueError(f"answers must be {ANSWERS}, got {answers}")
    logw = [math.log(p) for p in prior]
    consistent = [True] * len(prior)
    steps = []
    prev_h = None
    for t in range(len(answers) + 1):
        if t > 0:
            yes = answers[t - 1] == " yes"
            for c, p in enumerate(p_yes[t - 1]):
                like = p if yes else 1.0 - p
                logw[c] += math.log(like) if like > 0 else -math.inf
                consistent[c] = consistent[c] and ((p > 0.5) == yes)
        if all(x == -math.inf for x in logw):
            raise ValueError(f"step {t}: every concept has zero likelihood")
        h, post = log2_entropy(logw)
        n_cons = sum(consistent)
        step = {"step": t, "kind": None if t == 0 else observations[t - 1][0],
                "H": h, "bits": None if prev_h is None else prev_h - h,
                "n_consistent": n_cons, "H_hard": math.log2(n_cons) if n_cons else None}
        if secret_index is not None:
            p_s = post[secret_index]
            step |= {"p_secret": p_s, "rank_secret": 1 + sum(p > p_s for p in post),
                     "secret_consistent": consistent[secret_index]}
        steps.append(step)
        prev_h = h
    return steps


# ── batched Jev scoring ──────────────────────────────────────────────────


class JevScorer:
    """p_yes(kind, text, concept) from Jev Nouls — kind "question" (is the
    truthful answer yes?) or "clue" (is the clue correct?) — cached in a
    JSONL (rows without "kind" are questions)."""

    def __init__(self, cfg: EntropyConfig, concepts: ConceptSet, cache_path: Path, client=None):
        self.cfg = cfg
        self.definitions = dict(zip(concepts.words, concepts.definitions))
        self.cache_path = cache_path
        self.client = client
        self.cache: dict[tuple[str, str, str], float] = {}
        self.instructions = {"question": cfg.noul_instructions, "clue": cfg.clue_noul_instructions}
        self.criteria = {"question": cfg.noul_criteria, "clue": cfg.clue_noul_criteria}
        self.input_tokens = 0
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                r = json.loads(line)
                kind = r.get("kind", "question")
                if r["model"] != cfg.model or r["noul_instructions"] != self.instructions[kind]:
                    raise ValueError(f"{cache_path} holds scores from another model or prompt; "
                                     "use a new --cache")
                self.cache[(kind, r["question"], r["concept"])] = r["p_yes"]

    def questions(self, concepts: list[str], kind: str = "question") -> dict:
        from typesafe_sdk import Noul, NoulCriteria
        crit = NoulCriteria(true=self.criteria[kind]["true"], false=self.criteria[kind]["false"])
        return {f"c{i}": Noul(instructions=self.instructions[kind].format(
                    concept=c, definition=self.definitions[c]), criteria=crit)
                for i, c in enumerate(concepts)}

    async def score(self, triples: list[tuple[str, str, str]], progress: bool = True) -> None:
        """Score every (kind, text, concept) not cached; append to cache."""
        todo: dict[tuple[str, str], list[str]] = {}
        for kind, q, c in sorted(set(triples)):
            if (kind, q, c) not in self.cache:
                todo.setdefault((kind, q), []).append(c)
        n = self.cfg.concepts_per_request
        jobs = [(kq, cs[i:i + n]) for kq, cs in todo.items() for i in range(0, len(cs), n)]
        if not jobs:
            return
        sem = asyncio.Semaphore(self.cfg.concurrency)
        done = 0

        async def run(client, cache_f, kq: tuple[str, str], chunk: list[str]):
            nonlocal done
            kind, q = kq
            async with sem:
                r = await client.system_one(state={kind: q}, questions=self.questions(chunk, kind))
            self.input_tokens += r.usage.input_tokens
            for i, c in enumerate(chunk):
                p = r.answers[f"c{i}"].noul
                self.cache[(kind, q, c)] = p
                row = {"model": r.model, "noul_instructions": self.instructions[kind],
                       "question": q, "concept": c, "p_yes": p}
                if kind != "question":
                    row["kind"] = kind
                cache_f.write(json.dumps(row) + "\n")
            cache_f.flush()
            done += 1
            if progress:
                print(f"  {done}/{len(jobs)} requests", flush=True)

        with open(self.cache_path, "a") as cache_f:
            if self.client is not None:
                await asyncio.gather(*(run(self.client, cache_f, q, ch) for q, ch in jobs))
            else:
                from typesafe_sdk import AsyncTypeSafeClient
                async with AsyncTypeSafeClient(model=self.cfg.model) as client:
                    await asyncio.gather(*(run(client, cache_f, q, ch) for q, ch in jobs))

    def p_yes(self, question: str, concept: str, kind: str = "question") -> float:
        return self.cache[(kind, question, concept)]


class OpenRouterYesNoScorer:
    """p_yes(kind, text, concept) from an OpenRouter LLM: one chat prompt per
    pair, ``max_tokens`` 1, reasoning off, p(yes) = mass(yes_tokens) /
    (mass(yes_tokens) + mass(no_tokens)) over the first token's top logprobs.
    A pair with neither in the top logprobs is cached with p_yes = None and
    makes ``p_yes`` raise — never filled in. Same cache layout as JevScorer
    (plus ``mass``); rows are keyed by (kind, text, concept)."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, cfg: EntropyConfig, concepts: ConceptSet, cache_path: Path, client=None):
        import os
        self.cfg = cfg
        self.definitions = dict(zip(concepts.words, concepts.definitions))
        self.cache_path = cache_path
        self.client = client
        self.cache: dict[tuple[str, str, str], Optional[float]] = {}
        self.input_tokens = 0
        self.cost = 0.0
        self.prompts = {"question": cfg.llm_question_prompt, "clue": cfg.llm_clue_prompt}
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                r = json.loads(line)
                if r["model"] != cfg.model or r["prompt"] != self.prompts[r["kind"]]:
                    raise ValueError(f"{cache_path} holds scores from another model or prompt; use a new --cache")
                self.cache[(r["kind"], r["text"], r["concept"])] = r["p_yes"]

    def body(self, kind: str, text: str, concept: str) -> dict:
        content = self.prompts[kind].format(concept=concept, definition=self.definitions[concept], text=text)
        body = {"model": self.cfg.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": self.cfg.top_logprobs,
                "reasoning": {"enabled": False}}
        if self.cfg.openrouter_provider:
            body["provider"] = {"order": [self.cfg.openrouter_provider], "allow_fallbacks": False}
        return body

    def read(self, response: dict) -> tuple[Optional[float], float]:
        top = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        yes = sum(math.exp(t["logprob"]) for t in top if t["token"] in self.cfg.yes_tokens)
        no = sum(math.exp(t["logprob"]) for t in top if t["token"] in self.cfg.no_tokens)
        return (yes / (yes + no) if yes + no > 0 else None), yes + no

    async def _request(self, client, body: dict) -> dict:
        delay = 1.0
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = await client.post(self.URL, json=body)
            except httpx.TransportError:
                if attempt == self.cfg.max_retries:
                    raise
            else:
                if r.status_code == 200:
                    d = r.json()
                    if "error" not in d:
                        return d
                    if attempt == self.cfg.max_retries:
                        raise RuntimeError(f"OpenRouter error: {d['error']}")
                elif r.status_code not in (429, 500, 502, 503, 504) or attempt == self.cfg.max_retries:
                    raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:300]}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
        raise AssertionError("unreachable")

    async def score(self, triples: list[tuple[str, str, str]], progress: bool = True) -> None:
        todo = sorted({t for t in triples if t not in self.cache})
        if not todo:
            return
        if self.cfg.backend == "precomputed":
            raise ValueError(f"{len(todo)} pairs missing from {self.cache_path}, e.g. {todo[:3]}")
        sem = asyncio.Semaphore(self.cfg.concurrency)
        done = 0

        async def run(client, cache_f, kind, text, concept):
            nonlocal done
            async with sem:
                d = await self._request(client, self.body(kind, text, concept))
            p, mass = self.read(d)
            self.cache[(kind, text, concept)] = p
            self.input_tokens += d["usage"]["prompt_tokens"]
            self.cost += d["usage"].get("cost") or 0.0
            cache_f.write(json.dumps({"model": self.cfg.model, "provider": d.get("provider"), "kind": kind,
                                      "prompt": self.prompts[kind], "text": text, "concept": concept,
                                      "p_yes": p, "mass": mass}) + "\n")
            done += 1
            if progress and done % 10000 == 0:
                cache_f.flush()
                print(f"  {done}/{len(todo)} pairs, ${self.cost:.2f}", flush=True)

        with open(self.cache_path, "a") as cache_f:
            if self.client is not None:
                await asyncio.gather(*(run(self.client, cache_f, *t) for t in todo))
            else:
                limits = httpx.Limits(max_connections=self.cfg.concurrency, max_keepalive_connections=self.cfg.concurrency)
                async with httpx.AsyncClient(timeout=120, limits=limits,
                                             headers={"Authorization": f"Bearer {self.key}"}) as client:
                    await asyncio.gather(*(run(client, cache_f, *t) for t in todo))

    def p_yes(self, question: str, concept: str, kind: str = "question") -> float:
        p = self.cache[(kind, question, concept)]
        if p is None:
            raise ValueError(f"no yes/no token in the top logprobs for ({kind}, {question!r}, {concept!r})")
        return p

    def unanswered(self) -> list[tuple[str, str, str]]:
        return [k for k, v in self.cache.items() if v is None]


# ── driver ───────────────────────────────────────────────────────────────


def load_prior(cfg: EntropyConfig, concepts: ConceptSet) -> list[float]:
    if cfg.prior == "uniform":
        return [1.0 / len(concepts)] * len(concepts)
    counts = json.loads(Path(cfg.prior).read_text())
    missing = [w for w in concepts.words if counts.get(w, 0) <= 0]
    if missing:
        raise ValueError(f"prior {cfg.prior}: {len(missing)} concepts without a positive count, e.g. {missing[:5]}")
    total = sum(counts[w] for w in concepts.words)
    return [counts[w] / total for w in concepts.words]


def game_observations(cfg: EntropyConfig, game: dict) -> list[tuple[str, int, str, str]]:
    """(kind, turn, text, answer) in play order: each question, then its clue
    (if any and include_clues); a requested clue is a clue observation."""
    obs = []
    for e in game["exchanges"]:
        if e["kind"] == "question":
            obs.append(("question", e["turn"], e["text"], e["answer"]))
        if cfg.include_clues and e["kind"] in ("question", "clue_request") and e["clue"]:
            obs.append(("clue", e["turn"], e["clue"], " yes"))
    return obs


async def analyse(cfg: EntropyConfig, games: list[dict], cache_path: Path, client=None) -> dict:
    concepts = ConceptSet.load(cfg.concept_path())
    prior = load_prior(cfg, concepts)
    if cfg.backend not in ("jev", "openrouter", "precomputed"):
        raise ValueError(f"backend must be 'jev', 'openrouter' or 'precomputed', got {cfg.backend!r}")
    scorer = (JevScorer if cfg.backend == "jev" else OpenRouterYesNoScorer)(cfg, concepts, cache_path, client)
    observations = [game_observations(cfg, g) for g in games]
    await scorer.score([(k, text, c) for obs in observations for k, _, text, _ in obs for c in concepts.words])
    if cfg.backend != "jev" and scorer.unanswered():
        bad = scorer.unanswered()
        raise ValueError(f"{len(bad)} pairs had no yes/no token in the top logprobs, e.g. {bad[:3]}")

    out_games, bits_by_kind = [], {"question": [], "clue": []}
    for g, obs in zip(games, observations):
        idx = concepts.words.index(g["secret"]) if g["secret"] in concepts else None
        trace = [(k, a, [scorer.p_yes(text, c, k) for c in concepts.words]) for k, _, text, a in obs]
        steps = observation_trace(prior, trace, idx)
        for s, (k, turn, text, a) in zip(steps[1:], obs):
            s |= {"turn": turn, "text": text, "answer": a}
            if k == "question":
                s["question"] = text
            bits_by_kind[k].append(s["bits"])
        qbits = [s["bits"] for s in steps[1:] if s["kind"] == "question"]
        cbits = [s["bits"] for s in steps[1:] if s["kind"] == "clue"]
        out_games.append({"game_id": g["game_id"], "objective": g["objective"], "secret": g["secret"],
                          "secret_in_set": idx is not None, "n_questions": len(qbits), "n_clues": len(cbits),
                          "mean_bits": sum(qbits) / len(qbits) if qbits else None,
                          "mean_clue_bits": sum(cbits) / len(cbits) if cbits else None, "steps": steps})
    h0 = math.log2(len(concepts)) if cfg.prior == "uniform" else log2_entropy([math.log(p) for p in prior])[0]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {
        "config": dataclasses.asdict(cfg), "n_concepts": len(concepts), "H0": h0,
        "n_games": len(games), "n_questions": len(bits_by_kind["question"]),
        "n_clues": len(bits_by_kind["clue"]), "input_tokens": scorer.input_tokens,
        "cost_usd": getattr(scorer, "cost", None),
        "mean_bits_per_question": mean(bits_by_kind["question"]),
        "mean_bits_per_clue": mean(bits_by_kind["clue"]),
        "games": out_games,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--games", type=Path, nargs="+", required=True, help="games_{arm}.json files")
    ap.add_argument("--out", type=Path, required=True, help="entropy_{arm}.json")
    ap.add_argument("--cache", type=Path, default=None, help="score cache JSONL (default: next to --out)")
    args = ap.parse_args(argv)
    cfg = EntropyConfig.from_yaml(args.config)
    games = [g for path in args.games for g in json.loads(path.read_text())["games"]]
    cache = args.cache or args.out.with_name("jev_scores.jsonl")
    result = asyncio.run(analyse(cfg, games, cache))
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{result['n_games']} games, {result['n_questions']} questions, "
          f"mean {result['mean_bits_per_question']} bits/question, "
          f"{result['input_tokens']} input tokens -> {args.out}")


if __name__ == "__main__":
    main()
