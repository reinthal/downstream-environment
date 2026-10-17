"""Turn records: one per player action, in OpenAI chat format, so a probe
can re-render exactly what the model saw (``input_messages`` + ``tools``)
followed by what it produced (``reasoning`` + the tool call).

``input_messages`` drop the reasoning of earlier assistant turns: players
generate with ``reasoning_history="none"``, so earlier reasoning was never
sent (the Qwen chat template drops it as well). This turn's own reasoning is
``reasoning``; ``""`` on an injected turn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
)
from inspect_ai.tool import ToolInfo


def reasoning_of(message: ChatMessageAssistant) -> Optional[str]:
    """The message's reasoning text; None when it carries no reasoning block."""
    if isinstance(message.content, str):
        return None
    blocks = [c for c in message.content if isinstance(c, ContentReasoning)]
    if not blocks:
        return None
    if any(b.redacted for b in blocks):
        raise ValueError(f"redacted reasoning from {message.model}: the provider must expose it verbatim")
    return "\n\n".join(b.reasoning for b in blocks)


def to_openai(message: ChatMessage, *, keep_reasoning: bool = False) -> dict:
    if isinstance(message, (ChatMessageSystem, ChatMessageUser)):
        return {"role": message.role, "content": message.text}
    if isinstance(message, ChatMessageAssistant):
        out: dict[str, Any] = {"role": "assistant", "content": message.text}
        if keep_reasoning:
            out["reasoning_content"] = reasoning_of(message)
        if message.tool_calls:
            out["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function, "arguments": dict(tc.arguments)}}
                for tc in message.tool_calls
            ]
        return out
    if isinstance(message, ChatMessageTool):
        content = message.error.message if message.error else message.text
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": content}
    raise TypeError(f"unexpected message type {type(message).__name__}")


def tool_schema(info: ToolInfo) -> dict:
    return {"type": "function", "function": info.model_dump(exclude_none=True)}


@dataclass
class Turn:
    """One successful player action (failed attempts nested inside)."""

    game_id: str
    seed: int
    turn: int                      # 0 = setup (commit), then 1..max_questions
    phase: str                     # commit | question | answer | final_answer | judge
    player: str                    # seat name; the judge is the auditor's seat
    role: str                      # suspect | auditor | judge
    model: str
    tool: str
    arguments: dict                # validated (canonical) values
    raw_arguments: dict            # exactly what the model sent (parsed)
    reasoning: Optional[str]
    text: str
    tool_result: Optional[str]
    input_messages: list[dict]
    tools: list[dict]
    injected: bool = False         # bounded_random commit: written by the engine
    # set only when the arguments were repaired (one JSON object sent
    # repeatedly): the raw text the model produced
    raw_arguments_text: Optional[str] = None
    stop_reason: Optional[str] = None
    failed_attempts: list[dict] = field(default_factory=list)
    logprobs: Optional[list[dict]] = None
    usage: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)
