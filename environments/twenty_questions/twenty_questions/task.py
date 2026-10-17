"""Twenty Questions as an Inspect task: one sample per game (seed), the
engine as the solver, the judged outcome as the score. Every model call is
logged, so ``inspect view`` shows each player's (and the judge's) turns; the
full game record is in the sample store under ``"game"``.

    inspect eval twenty_questions/task.py -T config=configs/default.yaml -T n_games=4
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver

from .backends import InspectBackend
from .config import GameConfig, PlayerConfig, SuspectConfig
from .engine import TwentyQuestionsGame


def _generate_config(p: PlayerConfig, max_tokens: Optional[int] = None) -> GenerateConfig:
    logprobs = isinstance(p, SuspectConfig) and p.logprobs
    return GenerateConfig(
        max_tokens=max_tokens or p.max_tokens, temperature=p.temperature,
        reasoning_effort=p.reasoning_effort, reasoning_history="none",
        logprobs=True if logprobs else None, top_logprobs=p.top_logprobs if logprobs else None,
    )


def make_backends(cfg: GameConfig) -> dict[str, InspectBackend]:
    s, a = cfg.suspects[0], cfg.auditor
    judge_cfg = _generate_config(a, cfg.judge.max_tokens)
    judge_args = a.model_args if cfg.judge.model is None else {}
    return {
        s.name: InspectBackend(cfg.model_for(s), _generate_config(s), s.model_args),
        a.name: InspectBackend(cfg.model_for(a), _generate_config(a), a.model_args),
        "judge": InspectBackend(cfg.judge_model(), judge_cfg, judge_args),
    }


@solver
def play_game(cfg: GameConfig):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        seed = state.metadata["seed"]
        game = TwentyQuestionsGame(cfg, seed, make_backends(cfg), game_id=str(state.sample_id))
        result = await game.play()
        state.store.set("game", result.summary())
        state.store.set("turns", [t.to_dict() for t in result.turns])
        state.messages = list(game.auditor.messages)
        state.output.completion = f"winner={result.winner} end_turn={result.end_turn} secret={result.secret}"
        return state
    return solve


@scorer(metrics={"auditor_won": [mean()], "end_turn": [mean()]})
def game_outcome():
    async def score(state: TaskState, target: Target) -> Score:
        g = state.store.get("game")
        return Score(value={"auditor_won": float(g["winner"] == "auditor"), "end_turn": float(g["end_turn"])},
                     answer=g["exchanges"][-1]["text"], explanation=f"secret={g['secret']} reason={g['end_reason']}")
    return score


def make_task(cfg: GameConfig, n_games: int, base_seed: int = 0, name: str = "twenty_questions") -> Task:
    return Task(
        name=name,
        dataset=[Sample(id=f"game_{base_seed + i}", input="", metadata={"seed": base_seed + i})
                 for i in range(n_games)],
        solver=play_game(cfg),
        scorer=game_outcome(),
        metadata={"game_config": cfg.to_dict()},
    )


@task
def twenty_questions(config: str = "configs/default.yaml", n_games: int = 4, base_seed: int = 0) -> Task:
    return make_task(GameConfig.from_yaml(Path(config)), n_games, base_seed)
