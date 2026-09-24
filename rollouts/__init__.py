"""Reusable LLM-agent rollout infrastructure.

Generation backends for game/agent rollouts: vLLM for bulk generation, HF
transformers when steering hooks or shared-model activation reads are needed.
See rollouts/generation.py and AGENTS.md.
"""
from .generation import (CLOSE_CUE, GEN_KWARGS, GenerationBackend, HFBackend,
                         VLLMBackend, decoder_layers, load_hf_model,
                         make_backend, split_think)

__all__ = ["CLOSE_CUE", "GEN_KWARGS", "GenerationBackend", "HFBackend",
           "VLLMBackend", "decoder_layers", "load_hf_model", "make_backend",
           "split_think"]
