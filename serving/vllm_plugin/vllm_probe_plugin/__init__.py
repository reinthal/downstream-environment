"""vLLM plugin: probe read-out models (see serving/README.md).

Installed as a ``vllm.general_plugins`` entry point so that :func:`register`
runs in every vLLM process — the model registry is per-process, and the
engine core + workers each import it separately.
"""


def register():
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from vllm import ModelRegistry

    # Published probe checkpoints are text-only configs with model_type
    # "qwen3_5_text". Older transformers (e.g. .venv-vllm's 4.57) predate
    # qwen3_5 — fill the gap with vLLM's config class. Never shadow a native
    # transformers implementation (newer serving images have one).
    if "qwen3_5_text" not in CONFIG_MAPPING:
        from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig

        AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig)

    # Lazy string form: avoids importing torch/CUDA in the client process.
    ModelRegistry.register_model(
        "Qwen3_5ProbeForSequenceClassification",
        "vllm_probe_plugin.model:Qwen3_5ProbeForSequenceClassification",
    )
