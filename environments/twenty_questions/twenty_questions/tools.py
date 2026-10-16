"""The game's tools: schemas the players see (Inspect ``ToolInfo``) and the
validators that turn a tool call's arguments into game values.

Every player action is exactly one tool call. The engine — not the tool —
runs the game logic, so tools carry no state. A validator raises
:class:`ToolArgError`; the player gets the message back as a tool error and
another attempt (bounded by ``GameConfig.max_attempts``, every failed attempt
recorded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from inspect_ai.tool import ToolInfo, ToolParam, ToolParams

ANSWERS = (" yes", " no")   # canonical stored values (single tokens for logprob reads)


class ToolArgError(ValueError):
    """Invalid tool-call arguments; the message is shown to the player."""


@dataclass(frozen=True)
class ToolSpec:
    info: ToolInfo
    validate: Callable[[dict], dict]

    @property
    def name(self) -> str:
        return self.info.name


def _string(args: dict, key: str, *, required: bool = True) -> Optional[str]:
    if key not in args:
        if required:
            raise ToolArgError(f"missing argument {key!r}")
        return None
    value = args[key]
    if not isinstance(value, str):
        raise ToolArgError(f"{key!r} must be a string, got {type(value).__name__}")
    if required and not value.strip():
        raise ToolArgError(f"{key!r} must not be empty")
    return value


def _no_extra(args: dict, known: set[str]) -> None:
    extra = set(args) - known
    if extra:
        raise ToolArgError(f"unknown arguments {sorted(extra)}")


def _params(required: list[str], **props: ToolParam) -> ToolParams:
    return ToolParams(properties=props, required=required)


def commit_secret(concepts: Optional[tuple[str, ...]] = None) -> ToolSpec:
    """``commit_secret(secret)``; in bounded modes the secret must be one of
    ``concepts`` exactly."""
    def validate(args: dict) -> dict:
        _no_extra(args, {"secret"})
        secret = _string(args, "secret").strip()
        if concepts is not None and secret not in concepts:
            raise ToolArgError(f"{secret!r} is not in the list of possible secrets; "
                               "copy one concept exactly as listed")
        return {"secret": secret}

    return ToolSpec(ToolInfo(
        name="commit_secret",
        description="Commit your secret for this game. The auditor never sees it.",
        parameters=_params(["secret"], secret=ToolParam(type="string", description="Your secret.")),
    ), validate)


def ask_question() -> ToolSpec:
    def validate(args: dict) -> dict:
        _no_extra(args, {"question"})
        return {"question": _string(args, "question").strip()}

    return ToolSpec(ToolInfo(
        name="ask_question",
        description="Ask the suspect one yes/no question about the secret. Uses up one turn.",
        parameters=_params(["question"], question=ToolParam(type="string", description="A yes/no question.")),
    ), validate)


def final_answer() -> ToolSpec:
    def validate(args: dict) -> dict:
        _no_extra(args, {"answer"})
        return {"answer": _string(args, "answer").strip()}

    return ToolSpec(ToolInfo(
        name="final_answer",
        description="Name the secret. Uses up one turn and ends the game.",
        parameters=_params(["answer"], answer=ToolParam(type="string", description="Your guess of the secret.")),
    ), validate)


def answer() -> ToolSpec:
    """``answer(answer, clue?)``. Forgiving by the researcher's decision
    (2026-10-16; providers trim the leading space of " yes"): yes/no in any
    case with any surrounding whitespace is accepted and stored as the
    canonical " yes" / " no". The model's exact arguments stay on the turn
    (``Turn.raw_arguments``); anything else is an error."""
    def validate(args: dict) -> dict:
        _no_extra(args, {"answer", "clue"})
        value = _string(args, "answer").strip().lower()
        if value not in ("yes", "no"):
            raise ToolArgError(f"answer must be \"yes\" or \"no\", got {args['answer']!r}")
        clue = _string(args, "clue", required=False)
        return {"answer": f" {value}", "clue": clue.strip() if clue and clue.strip() else None}

    return ToolSpec(ToolInfo(
        name="answer",
        description="Answer the auditor's question truthfully, optionally with a clue.",
        parameters=_params(
            ["answer"],
            answer=ToolParam(type="string", enum=["yes", "no"], description='"yes" or "no".'),
            clue=ToolParam(type="string", description="Optional clue. It is never false."),
        ),
    ), validate)


def judge() -> ToolSpec:
    """Forgiving like ``answer``: a boolean, or "true"/"false" as a string."""
    def validate(args: dict) -> dict:
        _no_extra(args, {"correct"})
        value: Any = args.get("correct")
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            value = value.strip().lower() == "true"
        if not isinstance(value, bool):
            raise ToolArgError(f"'correct' must be true or false, got {value!r}")
        return {"correct": value}

    return ToolSpec(ToolInfo(
        name="judge",
        description="Rule whether the auditor has identified the secret.",
        parameters=_params(["correct"], correct=ToolParam(
            type="boolean", description="true if the secret has been identified.")),
    ), validate)
