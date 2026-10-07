"""vLLM plugin: probe read-out models (see serving/README.md).

Installed as a ``vllm.general_plugins`` entry point so that :func:`register`
runs in every vLLM process — the model registry is per-process, and the
engine core + workers each import it separately.
"""


def register():
    from transformers import AutoConfig

    from vllm import ModelRegistry
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig

    # .venv-vllm's transformers predates qwen3_5; published probe checkpoints
    # are text-only configs with model_type "qwen3_5_text", which vLLM's own
    # config registry does not map either.
    AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)

    # Lazy string form: avoids importing torch/CUDA in the client process.
    ModelRegistry.register_model(
        "Qwen3_5ProbeForSequenceClassification",
        "vllm_probe_plugin.model:Qwen3_5ProbeForSequenceClassification",
    )
