"""Posterior entropy of played games over a concept set ("posterior method"
of the concept-set-size note): how many bits each yes/no answer removed.

For every question a game asked and every concept c in the set, an oracle
prompt ("the secret is c — answer the question") is scored on the local vLLM
server: the logprobs of the single tokens " yes" and " no" as the next token,
p_yes(q, c) = sigmoid(lp_yes − lp_no). The posterior over concepts after
answers a_1..a_t is

    post_t(c) ∝ prior(c) · Π_s P(a_s | q_s, c),  P(" yes"|q,c) = p_yes(q,c)

and H_t = −Σ post_t log2 post_t; bits on turn t = H_{t−1} − H_t. A hard
variant counts the concepts consistent with every answer (p_yes > 0.5 for
" yes"). Only the yes/no channel is measured — clues are ignored.

Batched calls: each (q, c) pair renders through the server's chat template
(``/tokenize``, thinking off, assistant turn continued from
``answer_prefix``), the " yes" / " no" token id is appended, and the
completions endpoint scores many token-id prompts per request with
``prompt_logprobs``. Prefix caching makes the second variant ~free. Scores
are cached in a JSONL so a run resumes.

    uv run --directory environments/twenty_questions python -m twenty_questions.entropy \
        --config configs/entropy.yaml --games <dir>/games_<arm>.json --out <dir>/entropy_<arm>.json
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import yaml

from .concepts import ConceptSet
from .config import ENV_DIR, _check_keys
from .tools import ANSWERS


@dataclass
class EntropyConfig:
    base_url: str = "http://localhost:8001/v1"
    model: str = "Qwen/Qwen3.8-27B"
    api_key_env: str = "VLLM_API_KEY"
    concept_set: str = "concepts/things_n512_s0.json"
    # "uniform", or a JSON {concept: count} (e.g. a measured free-choice
    # distribution); every concept needs a count > 0
    prior: str = "uniform"
    oracle_system: str = ""
    oracle_prompt: str = (
        "You are answering one question in a game of Twenty Questions.\n"
        "Question: {question}\n"
        "The secret is: {concept}\n"
        "Answer the question truthfully about the secret, with yes or no."
    )
    answer_prefix: str = "Answer:"
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})
    batch_size: int = 128          # token-id prompts per completions request
    concurrency: int = 8           # requests in flight (tokenize: 4x this)
    timeout_s: float = 600.0

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


# ── batched vLLM scoring ─────────────────────────────────────────────────


class OracleScorer:
    def __init__(self, cfg: EntropyConfig, cache_path: Path):
        self.cfg = cfg
        self.cache_path = cache_path
        self.cache: dict[tuple[str, str], tuple[float, float]] = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                r = json.loads(line)
                if r["model"] != cfg.model:
                    raise ValueError(f"{cache_path} holds scores of {r['model']}, config model is {cfg.model}")
                self.cache[(r["question"], r["concept"])] = (r["lp_yes"], r["lp_no"])
        key = os.environ.get(cfg.api_key_env)
        self.client = httpx.AsyncClient(
            base_url=cfg.base_url.removesuffix("/v1"), timeout=cfg.timeout_s,
            headers={"Authorization": f"Bearer {key}"} if key else {})

    async def _post(self, path: str, body: dict) -> dict:
        resp = await self.client.post(path, json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"{path} -> {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    async def answer_token_ids(self) -> tuple[int, int]:
        ids = []
        for a in ANSWERS:
            r = await self._post("/tokenize", {"model": self.cfg.model, "prompt": a, "add_special_tokens": False})
            if len(r["tokens"]) != 1:
                raise ValueError(f"{a!r} is {len(r['tokens'])} tokens for {self.cfg.model}, need exactly 1")
            ids.append(r["tokens"][0])
        return ids[0], ids[1]

    def messages(self, question: str, concept: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.cfg.oracle_system}] if self.cfg.oracle_system else []
        return msgs + [
            {"role": "user", "content": self.cfg.oracle_prompt.format(question=question, concept=concept)},
            {"role": "assistant", "content": self.cfg.answer_prefix},
        ]

    async def _tokenize(self, question: str, concept: str) -> list[int]:
        r = await self._post("/tokenize", {
            "model": self.cfg.model, "messages": self.messages(question, concept),
            "add_generation_prompt": False, "continue_final_message": True,
            "add_special_tokens": False, "chat_template_kwargs": self.cfg.chat_template_kwargs})
        return r["tokens"]

    async def _score_batch(self, prompts: list[list[int]]) -> list[float]:
        r = await self._post("/v1/completions", {
            "model": self.cfg.model, "prompt": prompts, "max_tokens": 1, "temperature": 0.0,
            "prompt_logprobs": 0})
        choices = sorted(r["choices"], key=lambda c: c["index"])
        if len(choices) != len(prompts):
            raise RuntimeError(f"{len(choices)} choices for {len(prompts)} prompts")
        out = []
        for prompt, ch in zip(prompts, choices):
            last = ch["prompt_logprobs"][-1]
            out.append(last[str(prompt[-1])]["logprob"])
        return out

    async def score(self, pairs: list[tuple[str, str]], progress: bool = True) -> None:
        """Score every (question, concept) pair not cached; append to cache."""
        todo = sorted({p for p in pairs if p not in self.cache})
        if not todo:
            return
        yes_id, no_id = await self.answer_token_ids()
        tok_sem = asyncio.Semaphore(4 * self.cfg.concurrency)

        async def tok(pair):
            async with tok_sem:
                return await self._tokenize(*pair)

        req_sem = asyncio.Semaphore(self.cfg.concurrency)
        chunk = max(1, self.cfg.batch_size // 2)
        done = 0
        with open(self.cache_path, "a") as cache_f:
            async def run(batch: list[tuple[str, str]]):
                nonlocal done
                bases = await asyncio.gather(*(tok(p) for p in batch))
                prompts = [b + [i] for b in bases for i in (yes_id, no_id)]
                async with req_sem:
                    lps = await self._score_batch(prompts)
                for k, pair in enumerate(batch):
                    lp_yes, lp_no = lps[2 * k], lps[2 * k + 1]
                    self.cache[pair] = (lp_yes, lp_no)
                    cache_f.write(json.dumps({"model": self.cfg.model, "question": pair[0], "concept": pair[1],
                                              "lp_yes": lp_yes, "lp_no": lp_no}) + "\n")
                cache_f.flush()
                done += len(batch)
                if progress:
                    print(f"  scored {done}/{len(todo)} pairs", flush=True)

            await asyncio.gather(*(run(todo[i:i + chunk]) for i in range(0, len(todo), chunk)))

    def p_yes(self, question: str, concept: str) -> float:
        lp_yes, lp_no = self.cache[(question, concept)]
        return 1.0 / (1.0 + math.exp(lp_no - lp_yes))

    def mass(self, question: str, concept: str) -> float:
        """p(" yes") + p(" no"): how much of the next-token mass the two answers hold."""
        lp_yes, lp_no = self.cache[(question, concept)]
        return math.exp(lp_yes) + math.exp(lp_no)


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


async def analyse(cfg: EntropyConfig, games: list[dict], cache_path: Path) -> dict:
    concepts = ConceptSet.load(cfg.concept_path())
    prior = load_prior(cfg, concepts)
    scorer = OracleScorer(cfg, cache_path)
    try:
        questions = [[e for e in g["exchanges"] if e["kind"] == "question"] for g in games]
        pairs = [(e["text"], c) for qs in questions for e in qs for c in concepts.words]
        await scorer.score(pairs)
    finally:
        await scorer.client.aclose()

    out_games, all_bits, masses = [], [], []
    for g, qs in zip(games, questions):
        idx = concepts.words.index(g["secret"]) if g["secret"] in concepts else None
        p_yes = [[scorer.p_yes(e["text"], c) for c in concepts.words] for e in qs]
        masses += [scorer.mass(e["text"], c) for e in qs for c in concepts.words]
        steps = posterior_trace(prior, [e["answer"] for e in qs], p_yes, idx)
        for s, e in zip(steps[1:], qs):
            s |= {"turn": e["turn"], "question": e["text"], "answer": e["answer"]}
        bits = [s["bits"] for s in steps[1:]]
        all_bits += bits
        out_games.append({"game_id": g["game_id"], "objective": g["objective"], "secret": g["secret"],
                          "secret_in_set": idx is not None, "n_questions": len(qs),
                          "mean_bits": sum(bits) / len(bits) if bits else None, "steps": steps})
    masses.sort()
    return {
        "config": dataclasses.asdict(cfg), "n_concepts": len(concepts), "H0": math.log2(len(concepts)) if cfg.prior == "uniform" else log2_entropy([math.log(p) for p in prior])[0],
        "n_games": len(games), "n_questions": len(all_bits),
        "mean_bits_per_question": sum(all_bits) / len(all_bits) if all_bits else None,
        "answer_mass_quantiles": {q: masses[int(q * (len(masses) - 1))] for q in (0.0, 0.01, 0.1, 0.5)} if masses else None,
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
    cache = args.cache or args.out.with_name("oracle_scores.jsonl")
    result = asyncio.run(analyse(cfg, games, cache))
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{result['n_games']} games, {result['n_questions']} questions, "
          f"mean {result['mean_bits_per_question']} bits/question -> {args.out}")


if __name__ == "__main__":
    main()
