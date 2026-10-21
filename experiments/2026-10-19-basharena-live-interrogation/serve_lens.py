"""`vllm serve` for U with vllm-lens, plus a probe route on the same engine (runs on Modal).

U's chat completions and the fibs probe reads share one vLLM engine (researcher, 2026-10-21: "the
probe should leverage the same vllm-lens runtime ... so that the current trajectory is likely
cached in KV already"). POST /v1/probe/read takes the probe_server.py contract ({"messages",
"tools", "reads": [{"question", "answer"}]} or {"messages", "tools", "self": true}) and returns
{"scores", "n_tokens", "cached_tokens", "seconds"}.

Why a route of our own: vllm-lens forces skip_reading_prefix_cache on any hooked request (its
hooks may need every position), which would re-prefill the whole trajectory on every read. The
probe reads only the last position, which vLLM always computes, so this route replays vllm-lens's
server-side hook flow (install_hooks, set_hook_data, generate, get_hook_results, clear_hook_data)
through the unpatched AsyncLLM.generate with the prefix cache left on.

Rendering is the fibs read's (acts.render_completion, thinking off, earlier reasoning dropped):
the probe's training distribution. It shares KV blocks with earlier probe reads of the same
trajectory, not with U's own generation requests (rendered with thinking on).
"""

import asyncio
import sys
import time
import uuid
from types import SimpleNamespace

sys.path[:0] = ["/root"]

LAYERS = [19, 27, 35, 43, 51, 58]
TEMPLATE_KWARGS = {"enable_thinking": False, "preserve_thinking": False}
_state = SimpleNamespace(probe=None, hooks_installed=False, lock=asyncio.Lock())


def _render(tok, job, read_window):
    import torch
    from acts import render_completion, window_masks
    from probe_inference.score import build_read_mask

    if job.get("self"):
        assert job["messages"][-1]["role"] == "assistant", job["messages"][-1]["role"]
        full = tok.apply_chat_template(job["messages"], tools=job["tools"], tokenize=False,
                                       **TEMPLATE_KWARGS)
        if not full.endswith("<|im_end|>\n"):
            raise ValueError(f"unexpected template ending {full[-40:]!r}")
        ids = tok(full[:-1], add_special_tokens=False)["input_ids"]
        return [(ids, len(ids) - 2)]
    out = []
    for rd in job["reads"]:
        r = render_completion(tok, [*job["messages"], {"role": "user", "content": rd["question"]}],
                              rd["answer"], job["tools"], TEMPLATE_KWARGS)
        pm, cm = window_masks(r)
        mask = build_read_mask(read_window, pm, cm, torch.tensor(0))
        (pos,) = (mask.nonzero().flatten() + r.followup_start).tolist()
        out.append((r.ids, pos))
    return out


async def _read_one(engine, ids, pos):
    """The probe at ids[pos] (the prompt is cut after it), prefix cache allowed."""
    import cloudpickle
    import torch
    import vllm_lens._activations_plugin as lens
    from lens_hook import capture_last
    from vllm import SamplingParams, TokensPrompt
    from vllm_lens import Hook

    rid = f"probe-{uuid.uuid4().hex}"
    await engine.collective_rpc("set_hook_data", args=(
        rid, cloudpickle.dumps([Hook(fn=capture_last, layer_indices=LAYERS)])))
    try:
        final = None
        async for out in lens._original_generate(
                engine, TokensPrompt(prompt_token_ids=ids[: pos + 1]),
                SamplingParams(max_tokens=1, temperature=0.0), rid):
            final = out
        saved = lens._merge_hook_results(
            await engine.collective_rpc("get_hook_results", args=(rid,)))["0"]
    finally:
        await engine.collective_rpc("clear_hook_data", args=(rid,))
    n = saved[f"n{LAYERS[0]}"]
    if not 1 <= n <= pos + 1:
        raise RuntimeError(f"hook saw {n} tokens for a {pos + 1}-token prompt")
    acts = {L: saved[f"L{L}"][None, :] for L in LAYERS}
    p = float(_state.probe.score(acts, torch.tensor([True])))
    return p, len(ids), (final.num_cached_tokens or 0)


def probe_router():
    from fastapi import APIRouter, Request

    router = APIRouter(prefix="/v1/probe", tags=["ais407-probe"])

    @router.post("/read")
    async def read(raw: Request) -> dict:
        body = await raw.json()
        engine = raw.app.state.engine_client
        async with _state.lock:
            if not _state.hooks_installed:
                await engine.collective_rpc("install_hooks")
                _state.hooks_installed = True
        t0 = time.time()
        tok = engine.get_tokenizer()  # synchronous in vLLM 0.30
        reads = _render(tok, body, _state.probe.read_window)
        got = await asyncio.gather(*(_read_one(engine, ids, pos) for ids, pos in reads))
        return {"scores": [g[0] for g in got], "n_tokens": [g[1] for g in got],
                "cached_tokens": [g[2] for g in got], "seconds": round(time.time() - t0, 2)}

    return router


def main():
    from probe_inference import load_probe

    _state.probe = load_probe("/root/probe", device="cpu")
    assert list(_state.probe.layers) == LAYERS, _state.probe.layers
    import vllm.entrypoints.serve as serve_mod

    original = serve_mod.register_vllm_serve_api_routers

    def register(app):
        original(app)
        app.include_router(probe_router())

    serve_mod.register_vllm_serve_api_routers = register
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm", "serve", *sys.argv[1:]]
    vllm_main()


if __name__ == "__main__":
    main()
