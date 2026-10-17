"""The Twenty Questions engine: a seeded game between one suspect and one
auditor, judged after every turn.

Flow (``TwentyQuestionsGame.play``):

1. **Setup.** Both players get the public rules as a user message (identical
   text; the bounded concept list is appended for the suspect, and for the
   auditor only if ``auditor.show_concepts``, default off). The rules state
   both players' goals (the suspect's ``objective``: competitive or
   cooperative); system prompts are left for dynamics the other player must
   not see (empty by default). The models see the seat names (Player 1 /
   Player 2), never "suspect" or "auditor".
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
    kind: str                      # question | clue_request | final_answer
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


def fill(cfg: GameConfig, template: str, **fields) -> str:
    """Format a prompt template; {player1}, {player2} and {max_questions}
    are available in every template."""
    return template.format(player1=cfg.suspects[0].name, player2=cfg.auditor.name,
                           max_questions=cfg.max_questions, **fields)


def render_rules(cfg: GameConfig, concepts: Optional[ConceptSet], *, show_concepts: bool) -> str:
    p = cfg.prompts
    if concepts is None:
        space, listing = fill(cfg, p.secret_space_free), ""
    else:
        visibility = fill(cfg, p.visibility_both if cfg.auditor.show_concepts else p.visibility_suspect)
        space = fill(cfg, p.secret_space_bounded, n=len(concepts), visibility=visibility)
        listing = concept_listing(cfg, concepts) if show_concepts else ""
    pick = lambda options, key: fill(cfg, options[key])
    drop = "\x00"   # a rules line that is switched off
    if cfg.clues:
        answer_rule = fill(cfg, p.clue_answer_rule, clue_rule=pick(
            {"helpful": p.clue_rule_helpful, "misleading": p.clue_rule_misleading,
             "either": p.clue_rule_either}, cfg.clue_policy))
        clue_examples = pick({"helpful": p.clue_examples_helpful, "misleading": p.clue_examples_misleading,
                              "either": p.clue_examples_either}, cfg.clue_policy)
        counter_example = fill(cfg, p.clue_counter_example)
    else:
        answer_rule = clue_examples = counter_example = drop
    if cfg.clue_requests:
        request_rule = fill(cfg, p.clue_request_rule_turn if cfg.clue_request_uses_turn else p.clue_request_rule_free)
        request_example = fill(cfg, p.clue_request_example)
    else:
        request_rule = request_example = drop
    rules = fill(
        cfg, p.rules, secret_space=space,
        clue_answer_rule=answer_rule, clue_examples=clue_examples, clue_counter_example=counter_example,
        final_answer_rule=pick({"end": p.final_answer_rule_end, "continue": p.final_answer_rule_continue},
                               cfg.final_answer_mode),
        goals=pick({"competitive": p.goals_competitive, "cooperative": p.goals_cooperative},
                   cfg.suspects[0].objective),
        clue_request_rule=request_rule, clue_request_example=request_example,
    )
    rules = "\n".join(line for line in rules.split("\n") if line != drop)
    return f"{rules}\n\n{listing}" if listing else rules


def concept_listing(cfg: GameConfig, concepts: ConceptSet) -> str:
    return fill(cfg, cfg.prompts.concept_list, n=len(concepts), concepts=concepts.as_prompt_list())


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
        system = {"cooperative": p.system_cooperative, "competitive": p.system_competitive}[s_cfg.objective]
        self.suspect.system(fill(cfg, system))
        self.suspect.user("\n\n".join([
            render_rules(cfg, self.concepts, show_concepts=True),
            fill(cfg, p.role_suspect),
            fill(cfg, p.commit_instruction),
        ]))
        commit = T.commit_secret(self.concepts.words if self.concepts else None)
        if cfg.secret_mode == "bounded_random":
            drawn = self.rng.choice(self.concepts.words)
            action = self.suspect.inject(commit, {"secret": drawn}, turn=0, phase="commit")
        else:
            action = await self.suspect.act([commit], turn=0, phase="commit")
        self.secret = action.arguments["secret"]
        self.suspect.tool_result(action, fill(cfg, p.commit_ack))
        self._record(action)

        self.auditor.system(fill(cfg, p.system_auditor))
        start = fill(cfg, p.last_turn_notice if cfg.max_questions == 1 else p.auditor_start)
        self.auditor.user("\n\n".join([
            render_rules(cfg, self.concepts, show_concepts=cfg.auditor.show_concepts),
            fill(cfg, p.role_auditor),
            start,
        ]))

    async def _judge(self, turn: int, action_name: str, content: str, pending: Optional[Action] = None) -> bool:
        """``pending``: an auditor call whose result depends on this ruling
        (final_answer); the fork answers it with the neutral ack."""
        judge = self.auditor.fork("judge", self.backends["judge"])
        if pending is not None:
            judge.messages.append(ChatMessageTool(content=fill(self.cfg, self.cfg.prompts.final_answer_ack),
                                                  tool_call_id=pending.call_id, function=pending.tool))
        listing = concept_listing(self.cfg, self.concepts) if self.concepts and self.cfg.judge.show_concepts else ""
        template = {"question": self.cfg.prompts.judge_prompt_question,
                    "final answer": self.cfg.prompts.judge_prompt_final_answer}[action_name]
        text = fill(self.cfg, template, secret=self.secret, content=content)
        judge.user(f"{text}\n\n{listing}" if listing else text)
        action = self._record(await judge.act([T.judge()], turn=turn, phase="judge"))
        return action.arguments["correct"]

    async def _turn(self, turn: int) -> Optional[str]:
        """Play one turn; return the end reason if the game is over."""
        cfg, p = self.cfg, self.cfg.prompts
        last = turn == cfg.max_questions
        offered = [T.final_answer()] if last else [T.ask_question(), T.final_answer()]
        if cfg.clue_requests and not last:
            offered.insert(1, T.ask_for_clue())
        move = self._record(await self.auditor.act(offered, turn=turn, phase="question"))

        if move.tool == "ask_for_clue":
            # clues never identify the secret, so no judge here
            move.turn.phase = "clue_request"
            self.suspect.user(fill(cfg, p.clue_request_to_suspect, turn=turn))
            reply = self._record(await self.suspect.act([T.give_clue()], turn=turn, phase="clue"))
            self.suspect.tool_result(reply, fill(cfg, p.clue_ack))
            clue = reply.arguments["clue"]
            self.exchanges.append(Exchange(turn, "clue_request", "", None, clue, False))
            if cfg.clue_request_uses_turn:
                result = fill(cfg, p.clue_to_auditor_turn, clue=clue, next_turn=turn + 1)
                if turn + 1 == cfg.max_questions:
                    result += "\n" + fill(cfg, p.last_turn_notice)
                self.auditor.tool_result(move, result)
                return None
            self.auditor.tool_result(move, fill(cfg, p.clue_to_auditor_free, clue=clue, turn=turn))
            offered = [t for t in offered if t.name != "ask_for_clue"]   # once per turn
            move = self._record(await self.auditor.act(offered, turn=turn, phase="question"))

        if move.tool == "final_answer":
            move.turn.phase = "final_answer"
            guess = move.arguments["answer"]
            correct = await self._judge(turn, "final answer", guess, pending=move)
            self.exchanges.append(Exchange(turn, "final_answer", guess, None, None, correct))
            if correct or cfg.final_answer_mode == "end" or last:
                self.auditor.tool_result(move, fill(cfg, p.final_answer_ack))
                return "judged_correct" if correct else "final_answer_wrong"
            result = fill(cfg, p.final_answer_wrong, next_turn=turn + 1)
            if turn + 1 == cfg.max_questions:
                result += "\n" + fill(cfg, p.last_turn_notice)
            self.auditor.tool_result(move, result)
            return None

        question = move.arguments["question"]
        self.suspect.user(fill(cfg, p.question_to_suspect, turn=turn, question=question))
        reply = self._record(await self.suspect.act([T.answer(cfg.clues)], turn=turn, phase="answer"))
        self.suspect.tool_result(reply, fill(cfg, p.answer_ack))
        ans, clue = reply.arguments["answer"], reply.arguments["clue"]
        if cfg.clues:
            result = fill(cfg, p.answer_to_auditor, answer=ans.strip(), clue=clue or fill(cfg, p.no_clue),
                          next_turn=turn + 1)
        else:
            result = fill(cfg, p.answer_to_auditor_no_clues, answer=ans.strip(), next_turn=turn + 1)
        if turn + 1 == cfg.max_questions:
            result += "\n" + fill(cfg, p.last_turn_notice)
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
