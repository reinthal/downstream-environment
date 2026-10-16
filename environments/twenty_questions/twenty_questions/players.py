"""Players: a seat with a backend and its own conversation.

:meth:`Player.act` generates until the model makes exactly one valid call to
one of the offered tools, appending the assistant message to the
conversation. A reply with no (or several) tool calls is resampled from the
same context; a call to an unoffered tool or with invalid arguments gets a
tool error back and the model tries again. After ``max_attempts`` failures
the game raises :class:`GameError` — nothing is guessed on the model's
behalf. :meth:`Player.inject` writes an action into the conversation without
sampling (bounded_random commit).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Optional

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall, ToolCallError

from .backends import Backend
from .records import Turn, reasoning_of, to_openai, tool_schema
from .tools import ToolArgError, ToolSpec


class GameError(RuntimeError):
    """A player could not produce a valid action; the game is invalid."""


_PARSE_ERROR_PREFIX = "Error parsing the following tool call arguments:\n\n"
_PARSE_ERROR_SUFFIX = "\n\nError details: "


def duplicated_arguments(parse_error: str) -> Optional[tuple[dict, str]]:
    """Researcher-approved repair (2026-10-16; GLM-5.3 Flash on OpenRouter
    sometimes sends ``{"answer":"yes"}{"answer":"yes"}``): if the raw
    arguments in Inspect's parse error are one JSON object repeated two or
    more times and nothing else, return ``(object, raw text)``; otherwise
    None (still an error). Depends on Inspect's ``tool_parse_error_message``
    format, pinned by a test against ``parse_tool_call``."""
    if not parse_error.startswith(_PARSE_ERROR_PREFIX) or _PARSE_ERROR_SUFFIX not in parse_error:
        return None
    raw = parse_error[len(_PARSE_ERROR_PREFIX):].rsplit(_PARSE_ERROR_SUFFIX, 1)[0]
    decoder, objects, i = json.JSONDecoder(), [], 0
    while i < len(raw):
        while i < len(raw) and raw[i].isspace():
            i += 1
        if i == len(raw):
            break
        try:
            obj, i = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            return None
        objects.append(obj)
    if len(objects) < 2 or not isinstance(objects[0], dict) or any(o != objects[0] for o in objects):
        return None
    return objects[0], raw


@dataclass
class Action:
    tool: str
    arguments: dict          # validated
    call_id: str
    turn: Turn


@dataclass
class Player:
    name: str
    role: str
    backend: Backend
    game_id: str
    seed: int
    max_attempts: int
    messages: list[ChatMessage] = field(default_factory=list)

    def system(self, text: str) -> None:
        if self.messages:
            raise RuntimeError("system message must come first")
        if text:
            self.messages.append(ChatMessageSystem(content=text))

    def user(self, text: str) -> None:
        self.messages.append(ChatMessageUser(content=text))

    def tool_result(self, action: Action, text: str) -> None:
        self.messages.append(ChatMessageTool(content=text, tool_call_id=action.call_id, function=action.tool))
        action.turn.tool_result = text

    def _view(self) -> list[dict]:
        return [to_openai(m) for m in self.messages]

    async def act(self, tools: list[ToolSpec], *, turn: int, phase: str, role: Optional[str] = None) -> Action:
        specs = {t.name: t for t in tools}
        infos = [t.info for t in tools]
        failed: list[dict] = []
        for _ in range(self.max_attempts):
            view = self._view()
            out = await self.backend.generate(list(self.messages), infos)
            msg = out.message
            calls = msg.tool_calls or []
            attempt = {"text": msg.text, "reasoning": reasoning_of(msg), "stop_reason": out.stop_reason,
                       "tool_calls": [{"name": c.function, "arguments": dict(c.arguments)} for c in calls]}
            if len(calls) != 1:
                failed.append({**attempt, "error": f"{len(calls)} tool calls (need exactly one); resampled"})
                continue
            call = calls[0]
            error = None
            arguments, repaired_from = dict(call.arguments), None
            repair = duplicated_arguments(call.parse_error) if call.parse_error else None
            if repair is not None:
                arguments, repaired_from = repair
            if call.parse_error and repair is None:
                error = f"could not parse the tool call arguments: {call.parse_error}"
            elif call.function not in specs:
                error = f"tool {call.function!r} is not available now; use one of {sorted(specs)}"
            else:
                try:
                    args = specs[call.function].validate(dict(arguments))
                except ToolArgError as exc:
                    error = str(exc)
            if error is not None:
                failed.append({**attempt, "error": error})
                self.messages.append(msg)
                self.messages.append(ChatMessageTool(
                    content="", tool_call_id=call.id, function=call.function,
                    error=ToolCallError(type="parsing", message=error)))
                continue
            if repaired_from is not None:
                # the conversation carries the repaired call (one object)
                msg = msg.model_copy(update={"tool_calls": [
                    dataclasses.replace(call, arguments=arguments, parse_error=None)]})
            self.messages.append(msg)
            record = Turn(
                game_id=self.game_id, seed=self.seed, turn=turn, phase=phase, player=self.name,
                role=role or self.role, model=self.backend.model_name, tool=call.function,
                arguments=args, raw_arguments=dict(arguments), raw_arguments_text=repaired_from,
                reasoning=reasoning_of(msg), text=msg.text, tool_result=None,
                input_messages=view, tools=[tool_schema(i) for i in infos],
                stop_reason=out.stop_reason, failed_attempts=failed,
                logprobs=out.choices[0].logprobs.model_dump()["content"] if out.choices[0].logprobs else None,
                usage=out.usage.model_dump() if out.usage else None,
            )
            return Action(call.function, args, call.id, record)
        raise GameError(f"{self.name} ({phase}, turn {turn}): no valid action in "
                        f"{self.max_attempts} attempts; last: {failed[-1]}")

    def inject(self, tool: ToolSpec, arguments: dict, *, turn: int, phase: str) -> Action:
        """Append an assistant tool call the model did not sample, with an
        empty reasoning block, as if the model had made it."""
        args = tool.validate(arguments)
        view = self._view()
        call = ToolCall(id=f"injected_{phase}_{turn}", function=tool.name, arguments=args)
        self.messages.append(ChatMessageAssistant(
            content=[ContentReasoning(reasoning=""), ContentText(text="")],
            tool_calls=[call], source="generate", model=self.backend.model_name))
        record = Turn(
            game_id=self.game_id, seed=self.seed, turn=turn, phase=phase, player=self.name,
            role=self.role, model=self.backend.model_name, tool=tool.name, arguments=args,
            raw_arguments=dict(arguments),
            reasoning="", text="", tool_result=None, input_messages=view,
            tools=[tool_schema(tool.info)], injected=True,
        )
        return Action(tool.name, args, call.id, record)

    def fork(self, role: str, backend: Backend) -> "Player":
        """A copy of this conversation on another seat (the judge)."""
        return Player(self.name, role, backend, self.game_id, self.seed, self.max_attempts, list(self.messages))
