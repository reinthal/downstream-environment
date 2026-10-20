"""The vllm-lens post-hook of modal_probe_lens.py, in a module of its own.

vllm-lens cloudpickles hooks by reference, so vLLM's engine-core process imports the hook's
module; this one imports nothing, unlike the Modal app module.
"""


def capture_last(ctx, h):
    """The residual stream at this chunk's last position, + a token count, per layer."""
    ctx.saved[f"L{ctx.layer_idx}"] = h[-1].float().cpu()
    ctx.saved[f"n{ctx.layer_idx}"] = ctx.saved.get(f"n{ctx.layer_idx}", 0) + ctx.seq_len
    return None
