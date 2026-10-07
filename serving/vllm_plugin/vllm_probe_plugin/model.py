"""Qwen3.5 probe read-out model for vLLM.

A truncated Qwen3.5 decoder (the published checkpoint carries only blocks
``0..layer``) with one fp32 ``score`` Linear on top: the classify task
mean-pools the RAW residual stream over all prompt tokens and returns
``sigmoid(score(x))`` — exactly ``LinearProbe.predict_proba`` on the
activations the probe was trained on (see serving/publish.py for the
weight-folding math).

Two deliberate deviations from the stock seq-cls adapter:

* the final RMSNorm is replaced by :class:`ResidualAddNorm` — vLLM fuses the
  last residual add into the final norm, so a plain Identity would drop the
  residual, and a real norm would rescale per token (the probe lives in raw
  pre-norm activation space);
* the backbone is wrapped directly (no ``Qwen3_5ForCausalLMBase``), so no
  vocab-sized ``lm_head``/logits processor is ever materialized.
"""
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.seqwise import MeanPool, pooler_for_classify
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLMBase,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors


class ResidualAddNorm(nn.Module):
    """Stands in for the decoder's final RMSNorm: performs the deferred
    residual add and nothing else, so pooled hidden states equal the raw
    block-``layer`` residual stream the probe was trained on (the HF-side
    analog is load_truncated_decoder's Identity)."""

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return x
        s = x + residual
        return s, s


class Qwen3_5ProbeForSequenceClassification(nn.Module, HasInnerState, IsHybrid):
    is_pooling_model = True
    default_seq_pooling_type = "MEAN"
    packed_modules_mapping = Qwen3_5ForCausalLMBase.packed_modules_mapping

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 does not support 'all' prefix caching, "
                "use '--mamba-cache-mode=align' instead")
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vllm_config = vllm_config

        self.model = Qwen3_5Model(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "model"))
        self.model.norm = ResidualAddNorm()   # the probe reads PRE-norm activations

        self.score = ReplicatedLinear(
            config.hidden_size,
            config.num_labels,
            bias=True,
            params_dtype=torch.float32,
            return_bias=False,
            prefix=maybe_prefix(prefix, "score"),
        )
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        # MEAN over all prompt tokens is the probe's contract, not a tunable.
        self.pooler = DispatchPooler({
            "classify": pooler_for_classify(pooler_config, pooling=MeanPool(),
                                            classifier=self.score),
        })
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)

    # GDN cache sizing — identical to the stock Qwen3.5 model; the methods
    # only read vllm_config, so borrowing the bound classmethods is safe.
    get_mamba_state_dtype_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config)
    get_mamba_state_shape_from_config = (
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config)
    get_mamba_state_copy_func = Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func
