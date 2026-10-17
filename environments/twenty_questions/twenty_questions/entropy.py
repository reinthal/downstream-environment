"""Posterior entropy of played games over the THINGS concepts ("posterior
method" of the concept-set-size note): how many bits each yes/no answer
removed. Scored with TypeSafe's Jev.

For every question the auditor asked and every concept c in the set, a Jev
Noul answers "if the secret were c, is the truthful answer to the question
yes?": p_yes(q, c). The posterior over concepts after answers a_1..a_t is

    post_t(c) ∝ prior(c) · Π_s P(a_s | q_s, c),  P(" yes"|q,c) = p_yes(q,c)

and H_t = −Σ post_t log2 post_t; bits on turn t = H_{t−1} − H_t. A hard
variant counts the concepts consistent with every answer (p_yes > 0.5 for
" yes"). Only the yes/no channel is measured — clues are ignored.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .concepts import ConceptSet
from .config import ENV_DIR, _check_keys
from .tools import ANSWERS


@dataclass
class EntropyConfig:
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
    concepts_per_request: int = 300     # Nouls per request (per-request context: 64k tokens)
    concurrency: int = 8                # requests in flight

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
        step = {"step": t, "H": h, "bits": None if prev_h is None else prev_h - h,
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
    """p_yes(question, concept) from Jev Nouls, cached in a JSONL."""

    def __init__(self, cfg: EntropyConfig, concepts: ConceptSet, cache_path: Path, client=None):
        self.cfg = cfg
        self.definitions = dict(zip(concepts.words, concepts.definitions))
        self.cache_path = cache_path
        self.client = client
        self.cache: dict[tuple[str, str], float] = {}
        self.input_tokens = 0
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                r = json.loads(line)
                if r["model"] != cfg.model or r["noul_instructions"] != cfg.noul_instructions:
                    raise ValueError(f"{cache_path} holds scores from another model or prompt; "
                                     "use a new --cache")
                self.cache[(r["question"], r["concept"])] = r["p_yes"]

    def questions(self, concepts: list[str]) -> dict:
        from typesafe_sdk import Noul, NoulCriteria
        crit = NoulCriteria(true=self.cfg.noul_criteria["true"], false=self.cfg.noul_criteria["false"])
        return {f"c{i}": Noul(instructions=self.cfg.noul_instructions.format(
                    concept=c, definition=self.definitions[c]), criteria=crit)
                for i, c in enumerate(concepts)}

    async def score(self, pairs: list[tuple[str, str]], progress: bool = True) -> None:
        """Score every (question, concept) pair not cached; append to cache."""
        todo: dict[str, list[str]] = {}
        for q, c in sorted(set(pairs)):
            if (q, c) not in self.cache:
                todo.setdefault(q, []).append(c)
        n = self.cfg.concepts_per_request
        jobs = [(q, cs[i:i + n]) for q, cs in todo.items() for i in range(0, len(cs), n)]
        if not jobs:
            return
        sem = asyncio.Semaphore(self.cfg.concurrency)
        done = 0

        async def run(client, cache_f, q: str, chunk: list[str]):
            nonlocal done
            async with sem:
                r = await client.system_one(state={"question": q}, questions=self.questions(chunk))
            self.input_tokens += r.usage.input_tokens
            for i, c in enumerate(chunk):
                p = r.answers[f"c{i}"].noul
                self.cache[(q, c)] = p
                cache_f.write(json.dumps({"model": r.model, "noul_instructions": self.cfg.noul_instructions,
                                          "question": q, "concept": c, "p_yes": p}) + "\n")
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

    def p_yes(self, question: str, concept: str) -> float:
        return self.cache[(question, concept)]


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


async def analyse(cfg: EntropyConfig, games: list[dict], cache_path: Path, client=None) -> dict:
    concepts = ConceptSet.load(cfg.concept_path())
    prior = load_prior(cfg, concepts)
    scorer = JevScorer(cfg, concepts, cache_path, client)
    questions = [[e for e in g["exchanges"] if e["kind"] == "question"] for g in games]
    await scorer.score([(e["text"], c) for qs in questions for e in qs for c in concepts.words])

    out_games, all_bits = [], []
    for g, qs in zip(games, questions):
        idx = concepts.words.index(g["secret"]) if g["secret"] in concepts else None
        p_yes = [[scorer.p_yes(e["text"], c) for c in concepts.words] for e in qs]
        steps = posterior_trace(prior, [e["answer"] for e in qs], p_yes, idx)
        for s, e in zip(steps[1:], qs):
            s |= {"turn": e["turn"], "question": e["text"], "answer": e["answer"]}
        bits = [s["bits"] for s in steps[1:]]
        all_bits += bits
        out_games.append({"game_id": g["game_id"], "objective": g["objective"], "secret": g["secret"],
                          "secret_in_set": idx is not None, "n_questions": len(qs),
                          "mean_bits": sum(bits) / len(bits) if bits else None, "steps": steps})
    h0 = math.log2(len(concepts)) if cfg.prior == "uniform" else log2_entropy([math.log(p) for p in prior])[0]
    return {
        "config": dataclasses.asdict(cfg), "n_concepts": len(concepts), "H0": h0,
        "n_games": len(games), "n_questions": len(all_bits), "input_tokens": scorer.input_tokens,
        "mean_bits_per_question": sum(all_bits) / len(all_bits) if all_bits else None,
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
