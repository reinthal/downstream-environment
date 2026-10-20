"""`openrouterstream/<model>`: Inspect's OpenRouter provider with streaming on by default.

Long Qwen3.8 generations on DeepInfra exceeded OpenRouter's ~300 s upstream timeout when sent
as one non-streamed request ("Provider timed out after 307152ms"); streamed, the connection
carries tokens as they are generated (researcher, 2026-10-20: stream, provider
"deepinfra/bf16"). BashArena's policies build their model with get_model(model_name) and no
model args, so streaming has to come from the provider name. Import this module before
models are resolved (run.py does).
"""

from typing import Any

from inspect_ai.model import GenerateConfig, ModelAPI, modelapi


@modelapi(name="openrouterstream")
def openrouterstream() -> type[ModelAPI]:
    from inspect_ai.model._providers.openrouter import OpenRouterAPI

    class OpenRouterStreamAPI(OpenRouterAPI):
        def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            api_key: str | None = None,
            config: GenerateConfig = GenerateConfig(),
            **model_args: Any,
        ) -> None:
            model_args.setdefault("stream", True)
            super().__init__(model_name, base_url, api_key, config, **model_args)

    return OpenRouterStreamAPI
