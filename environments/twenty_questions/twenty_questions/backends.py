"""Compute backends: what a player calls to generate one assistant message.

- :class:`InspectBackend` — any Inspect model string: ``openrouter/...``,
  the local Qwen3.8-27B server (``openai-api/vllm/Qwen/Qwen3.8-27B`` with
  ``VLLM_BASE_URL``/``VLLM_API_KEY``), ``mockllm/model``. Inside an Inspect
  eval every call is logged, so ``inspect view`` shows each player's turn.
- :class:`ScriptedBackend` — a Python policy; deterministic engine tests.
"""

from __future__ import annotations

import itertools
from typing import Callable, Protocol

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolCall, ToolInfo


class Backend(Protocol):
    model_name: str

    async def generate(self, messages: list[ChatMessage], tools: list[ToolInfo]) -> ModelOutput: ...


class InspectBackend:
    def __init__(self, model_name: str, config: GenerateConfig, model_args: dict | None = None):
        self.model_name = model_name
        self.config = config
        self.model_args = dict(model_args or {})

    async def generate(self, messages: list[ChatMessage], tools: list[ToolInfo]) -> ModelOutput:
        model = get_model(self.model_name, config=self.config, **self.model_args)
        return await model.generate(input=messages, tools=tools, tool_choice="auto")


_call_ids = itertools.count()


def tool_call_output(function: str, arguments: dict, *, reasoning: str = "", text: str = "",
                     model: str = "scripted") -> ModelOutput:
    """A ModelOutput holding one tool call (scripted players, tests)."""
    content = [ContentReasoning(reasoning=reasoning), ContentText(text=text)]
    message = ChatMessageAssistant(
        content=content, model=model, source="generate",
        tool_calls=[ToolCall(id=f"call_{next(_call_ids)}", function=function, arguments=arguments)],
    )
    return ModelOutput.from_message(message, stop_reason="tool_calls")


class ScriptedBackend:
    """``policy(messages, tool_names) -> ModelOutput`` (see tool_call_output)."""

    def __init__(self, policy: Callable[[list[ChatMessage], list[str]], ModelOutput],
                 model_name: str = "scripted"):
        self.policy = policy
        self.model_name = model_name

    async def generate(self, messages: list[ChatMessage], tools: list[ToolInfo]) -> ModelOutput:
        return self.policy(messages, [t.name for t in tools])
