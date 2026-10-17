"""Probe interface: which conversation a probe reads, and how scores attach
back to turns.

A :class:`ProbeView` is one player action exactly as the model produced it:
the OpenAI-format messages it saw, the tools it was offered, and its own
assistant message (reasoning + the tool call). Render it with the model's
chat template (``tokenizer.apply_chat_template(view.messages(), tools=view.tools)``)
and read the probe on ``view.span`` — e.g. the suspect's answer + clue.

Probes themselves (``LMProbe``, fibs ensembles, ...) live in the repo-root
``.venv``; this module has no torch dependency. Anything with a
``score(views) -> list[float]`` method is a :class:`Probe`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


@dataclass(frozen=True)
class ProbeView:
    game_id: str
    turn: int
    phase: str
    player: str
    role: str
    input_messages: list[dict]
    tools: list[dict]
    reasoning: str | None
    tool: str
    arguments: dict      # the model's raw arguments: what it actually produced
    injected: bool

    def assistant_message(self) -> dict:
        return {"role": "assistant", "content": "", "reasoning_content": self.reasoning,
                "tool_calls": [{"type": "function", "function": {"name": self.tool, "arguments": self.arguments}}]}

    def messages(self) -> list[dict]:
        return [*self.input_messages, self.assistant_message()]


class Probe(Protocol):
    def score(self, views: list[ProbeView]) -> list[float]: ...


def read_views(transcripts: Path | str, *, roles: Iterable[str] = ("suspect",),
               phases: Iterable[str] = ("answer", "clue")) -> list[ProbeView]:
    """Views of the selected turns of a transcripts_{arm}.jsonl, file order.
    Default: every suspect answer and requested clue."""
    roles, phases = set(roles), set(phases)
    views = []
    for line in Path(transcripts).read_text().splitlines():
        r = json.loads(line)
        if r["role"] in roles and r["phase"] in phases:
            views.append(ProbeView(r["game_id"], r["turn"], r["phase"], r["player"], r["role"],
                                   r["input_messages"], r["tools"], r["reasoning"], r["tool"],
                                   r["raw_arguments"], r["injected"]))
    return views


def score_rows(probe: Probe, views: list[ProbeView], name: str) -> list[dict]:
    """``{game_id, turn, phase, player, <name>}`` rows for a scores_{arm}.csv."""
    scores = probe.score(views)
    if len(scores) != len(views):
        raise ValueError(f"probe returned {len(scores)} scores for {len(views)} views")
    return [{"game_id": v.game_id, "turn": v.turn, "phase": v.phase, "player": v.player, name: s}
            for v, s in zip(views, scores)]
