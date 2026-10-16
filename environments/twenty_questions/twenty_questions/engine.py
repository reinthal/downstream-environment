"""The Twenty Questions engine: a seeded game between one suspect and one
auditor, judged after every turn.

Flow (``TwentyQuestionsGame.play``):

1. **Setup.** Both players get the public rules as a user message (identical
   text; the bounded concept list is appended for the suspect, and for the
   auditor only if ``auditor.show_concepts``, default off). Hidden dynamics — the suspect's
   objective — are system prompts.
2. **Commit** (turn 0). ``free_form``: the suspect picks anything;
   ``bounded_free_form``: it picks from the concept set; ``bounded_random``:
   the engine draws a concept uniformly (seeded) and injects the
   ``commit_secret`` call with empty reasoning after the same prompt.
3. **Turns 1..max_questions.** The auditor calls ``ask_question`` or
   ``final_answer`` (last turn: ``final_answer`` only). A question goes to
   the suspect, who calls ``answer(yes|no, clue?)`` (stored " yes"/" no"); the auditor gets
   the answer as its tool result.
4. **Judge** after every turn: the auditor's conversation is forked, the
   secret revealed, and the judge (the auditor's model unless configured)
   rules ``judge(correct)`` (bounded modes: the judge prompt carries the
   concept list, ``judge.show_concepts``). Correct -> the auditor wins on that
   turn. A wrong ``final_answer`` ends the game (``final_answer_mode: end``)
   or the auditor is told and plays on (``continue``). The fork is discarded: the
   auditor's live conversation never sees the secret.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Optional

from inspect_ai.model import ChatMessageTool

from . import tools as T
from .backends import Backend
from .concepts import ConceptSet
from .config import GameConfig
from .players import Action, Player
from .records import Turn


@dataclass
class Exchange:
    turn: int
    kind: str                      # question | final_answer
    text: str                      # the question or the guess
    answer: Optional[str]          # " yes" | " no" (questions)
    clue: Optional[str]
    judged_correct: bool


@dataclass
class GameResult:
    game_id: str
    seed: int
    secret_mode: str
    clue_policy: str
    objective: str
    max_questions: int
    n_concepts: Optional[int]
    secret: str
    winner: str                    # auditor | suspect
    end_turn: int                  # turn of the last action
    end_reason: str                # judged_correct | final_answer_wrong | turns_exhausted
    models: dict
    exchanges: list[Exchange] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)

    def summary(self) -> dict:
        """Everything but the per-turn records (games_{arm}.json rows)."""
        d = asdict(self)
        d.pop("turns")
        return d


def render_rules(cfg: GameConfig, concepts: Optional[ConceptSet], *, show_concepts: bool) -> str:
    p = cfg.prompts
    if concepts is None:
        space, listing = p.secret_space_free, ""
    else:
        visibility = p.visibility_both if cfg.auditor.show_concepts else p.visibility_suspect
        space = p.secret_space_bounded.format(n=len(concepts), visibility=visibility)
        listing = concept_listing(cfg, concepts) if show_concepts else ""
    clue_rule = {"helpful": p.clue_rule_helpful, "misleading": p.clue_rule_misleading,
                 "either": p.clue_rule_either}[cfg.clue_policy]
    clue_examples = {"helpful": p.clue_examples_helpful, "misleading": p.clue_examples_misleading,
                     "either": p.clue_examples_either}[cfg.clue_policy]
    final_rule = {"end": p.final_answer_rule_end, "continue": p.final_answer_rule_continue}[cfg.final_answer_mode]
    rules = p.rules.format(
        suspect=cfg.suspects[0].name, auditor=cfg.auditor.name, secret_space=space,
        max_questions=cfg.max_questions, clue_rule=clue_rule, clue_examples=clue_examples,
        final_answer_rule=final_rule,
    )
    return f"{rules}\n\n{listing}" if listing else rules


def concept_listing(cfg: GameConfig, concepts: ConceptSet) -> str:
    return cfg.prompts.concept_list.format(n=len(concepts), concepts=concepts.as_prompt_list())


class TwentyQuestionsGame:
    def __init__(self, cfg: GameConfig, seed: int, backends: dict[str, Backend], game_id: Optional[str] = None):
        """``backends``: one per seat name, plus ``"judge"``."""
        self.cfg = cfg
        self.seed = seed
        self.game_id = game_id or f"game_{seed}"
        self.rng = random.Random(seed)
        self.concepts = ConceptSet.load(cfg.concept_path()) if cfg.bounded else None
        s_cfg, a_cfg = cfg.suspects[0], cfg.auditor
        missing = {s_cfg.name, a_cfg.name, "judge"} - set(backends)
        if missing:
            raise ValueError(f"no backend for {sorted(missing)}")
        self.backends = backends
        self.suspect = Player(s_cfg.name, "suspect", backends[s_cfg.name], self.game_id, seed, cfg.max_attempts)
        self.auditor = Player(a_cfg.name, "auditor", backends[a_cfg.name], self.game_id, seed, cfg.max_attempts)
        self.turns: list[Turn] = []
        self.exchanges: list[Exchange] = []
        self.secret: Optional[str] = None

    def _record(self, action: Action) -> Action:
        self.turns.append(action.turn)
        return action

    # ── phases ───────────────────────────────────────────────────────

    async def _setup(self) -> None:
        cfg, p = self.cfg, self.cfg.prompts
        s_cfg = cfg.suspects[0]
        system = {"honest": p.system_honest, "adversarial": p.system_adversarial}[s_cfg.objective]
        self.suspect.system(system.format(max_questions=cfg.max_questions))
        self.suspect.user("\n\n".join([
            render_rules(cfg, self.concepts, show_concepts=True),
            p.role_suspect.format(name=s_cfg.name),
            p.commit_instruction,
        ]))
        commit = T.commit_secret(self.concepts.words if self.concepts else None)
        if cfg.secret_mode == "bounded_random":
            drawn = self.rng.choice(self.concepts.words)
            action = self.suspect.inject(commit, {"secret": drawn}, turn=0, phase="commit")
        else:
            action = await self.suspect.act([commit], turn=0, phase="commit")
        self.secret = action.arguments["secret"]
        self.suspect.tool_result(action, p.commit_ack)
        self._record(action)

        self.auditor.system(p.system_auditor.format(max_questions=cfg.max_questions))
        start = p.last_turn_notice if cfg.max_questions == 1 else p.auditor_start.format(max_questions=cfg.max_questions)
        self.auditor.user("\n\n".join([
            render_rules(cfg, self.concepts, show_concepts=cfg.auditor.show_concepts),
            p.role_auditor.format(name=cfg.auditor.name),
            start,
        ]))

    async def _judge(self, turn: int, action_name: str, content: str, pending: Optional[Action] = None) -> bool:
        """``pending``: an auditor call whose result depends on this ruling
        (final_answer); the fork answers it with the neutral ack."""
        judge = self.auditor.fork("judge", self.backends["judge"])
        if pending is not None:
            judge.messages.append(ChatMessageTool(content=self.cfg.prompts.final_answer_ack,
                                                  tool_call_id=pending.call_id, function=pending.tool))
        listing = concept_listing(self.cfg, self.concepts) if self.concepts and self.cfg.judge.show_concepts else ""
        template = {"question": self.cfg.prompts.judge_prompt_question,
                    "final answer": self.cfg.prompts.judge_prompt_final_answer}[action_name]
        text = template.format(secret=self.secret, content=content)
        judge.user(f"{text}\n\n{listing}" if listing else text)
        action = self._record(await judge.act([T.judge()], turn=turn, phase="judge"))
        return action.arguments["correct"]

    async def _turn(self, turn: int) -> Optional[str]:
        """Play one turn; return the end reason if the game is over."""
        cfg, p = self.cfg, self.cfg.prompts
        last = turn == cfg.max_questions
        offered = [T.final_answer()] if last else [T.ask_question(), T.final_answer()]
        move = self._record(await self.auditor.act(offered, turn=turn, phase="question"))

        if move.tool == "final_answer":
            move.turn.phase = "final_answer"
            guess = move.arguments["answer"]
            correct = await self._judge(turn, "final answer", guess, pending=move)
            self.exchanges.append(Exchange(turn, "final_answer", guess, None, None, correct))
            if correct or cfg.final_answer_mode == "end" or last:
                self.auditor.tool_result(move, p.final_answer_ack)
                return "judged_correct" if correct else "final_answer_wrong"
            result = p.final_answer_wrong.format(next_turn=turn + 1, max_questions=cfg.max_questions)
            if turn + 1 == cfg.max_questions:
                result += "\n" + p.last_turn_notice
            self.auditor.tool_result(move, result)
            return None

        question = move.arguments["question"]
        self.suspect.user(p.question_to_suspect.format(turn=turn, max_questions=cfg.max_questions, question=question))
        reply = self._record(await self.suspect.act([T.answer()], turn=turn, phase="answer"))
        self.suspect.tool_result(reply, p.answer_ack)
        ans, clue = reply.arguments["answer"], reply.arguments["clue"]
        result = p.answer_to_auditor.format(answer=ans.strip(), clue=clue or p.no_clue,
                                            next_turn=turn + 1, max_questions=cfg.max_questions)
        if turn + 1 == cfg.max_questions:
            result += "\n" + p.last_turn_notice
        self.auditor.tool_result(move, result)
        correct = await self._judge(turn, "question", f"{question} (answer:{ans})")
        self.exchanges.append(Exchange(turn, "question", question, ans, clue, correct))
        return "judged_correct" if correct else None

    async def play(self) -> GameResult:
        await self._setup()
        reason, turn = None, 0
        for turn in range(1, self.cfg.max_questions + 1):
            reason = await self._turn(turn)
            if reason is not None:
                break
        if reason is None:
            raise AssertionError("the last turn offers only final_answer, which always ends the game")
        if reason == "final_answer_wrong" and self.cfg.final_answer_mode == "continue":
            reason = "turns_exhausted"
        s_cfg = self.cfg.suspects[0]
        return GameResult(
            game_id=self.game_id, seed=self.seed, secret_mode=self.cfg.secret_mode,
            clue_policy=self.cfg.clue_policy, objective=s_cfg.objective,
            max_questions=self.cfg.max_questions,
            n_concepts=len(self.concepts) if self.concepts else None, secret=self.secret,
            winner="auditor" if reason == "judged_correct" else "suspect", end_turn=turn,
            end_reason=reason,
            models={"suspect": self.suspect.backend.model_name, "auditor": self.auditor.backend.model_name,
                    "judge": self.backends["judge"].model_name},
            exchanges=self.exchanges, turns=self.turns,
        )
