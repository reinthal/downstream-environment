"""fibs probe reads on Modal: Qwen3.8-27B (bf16) under vLLM + vllm-lens, the linear ensemble as a hook.

Same contract as probe_server.py (the bf16 service on gpaulo-ord-0), so PROBE_READ_URL can point here:
POST /read {"messages", "tools", "reads": [{"question", "answer"}]} -> {"scores", "n_tokens",
"seconds"}; a job {"messages", "tools", "self": true} is the self read instead (U's own step,
no follow-up: the token before the step's <|im_end|>, one score). POST /read_batch {"jobs": [<read bodies>]} -> {"results": [...]} runs every read of
every job in one vLLM batch. Each read is rendered with U's chat template exactly as
probe_server.py renders it (fibs acts.render_completion), then sent truncated after the probe's
read token (the token before the answer's <|im_end|>): in a causal model that token's
activations depend only on what precedes it. A vllm-lens hook keeps the residual stream
(hidden + residual, as vllm-lens hands it to hooks) at the last position of each forward chunk
at layers 19-58; with max_tokens=1 the last chunk ends at the read token (checked: the hook's
token count must equal the prompt length). The linear ensemble then scores it on the CPU.

Same weights and revision the probe was trained on (bf16, 1d4bf0f), on one H200 (141 GB: ~55 GB
of weights + KV cache for 262k-token reads). An FP8 version (Qwen3.8-27B-FP8, H100) matched the
bf16 reads at r 0.996 (probe_lens_validation.json) and was replaced to drop that deviation.
validate_probe_lens.py compares this deployment with the bf16 HF service on Phase A steps.

Known: vllm-lens 1.3.0 logs "pre-hook error on layer N, skipping" for every layer on Qwen3.8
(its pre-hook wrapper expects forward(positions, hidden_states, residual)); post-hooks, the only
kind used here, are unaffected (every read is checked against the prompt length). Reported:
https://github.com/UKGovernmentBEIS/vllm-lens/issues/52

Deploy (cadenza workspace):  MODAL_PROFILE=cadenza modal deploy modal_probe_lens.py
"""

from pathlib import Path

import modal

# The add_local_* sources below are read only where the app is deployed from; inside the
# container this module sits at /root/modal_probe_lens.py and has no repo around it.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PROBE_DIR = "/root/probe"
LAYERS = [19, 27, 35, 43, 51, 58]
TEMPLATE_KWARGS = {"enable_thinking": False, "preserve_thinking": False}

image = (
    # CUDA devel base: vLLM JIT-compiles kernels at startup (FlashInfer sampling, DeepGEMM) and
    # needs nvcc; the vllm==0.30.0 PyPI wheel is CUDA-13-built.
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.30.0", "vllm-lens==1.3.0", "fastapi[standard]", "huggingface_hub")
    .env({"HF_HOME": "/hf", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
    .add_local_dir(REPO / "vendor" / "caught-in-the-act-probes" / "src" / "probe_inference",
                   "/root/probe_inference")
    .add_local_dir(REPO / "experiments" / "2026-10-13-fibs-basharena" / "probes_main" / "main"
                   / "linear", PROBE_DIR)
    .add_local_file(REPO / "experiments" / "2026-10-13-fibs-basharena" / "acts.py",
                    "/root/acts.py")
    .add_local_file(REPO / "experiments" / "2026-10-19-basharena-live-interrogation"
                    / "lens_hook.py", "/root/lens_hook.py")
)
hf_cache = modal.Volume.from_name("ais407-hf-cache", create_if_missing=True)
app = modal.App("ais407-probe-lens", image=image)


@app.cls(gpu="H200", volumes={"/hf": hf_cache}, timeout=3600, scaledown_window=300,
         max_containers=6)
class ProbeLens:
    @modal.enter()
    def load(self):
        import torch
        from huggingface_hub import snapshot_download
        from probe_inference import load_probe
        from vllm import LLM

        path = snapshot_download(MODEL, revision=REVISION)
        hf_cache.commit()
        self.llm = LLM(model=path, tokenizer=path, max_model_len=262144,
                       gpu_memory_utilization=0.90, enable_prefix_caching=False,
                       limit_mm_per_prompt={"image": 0, "video": 0})
        self.tok = self.llm.get_tokenizer()
        self.probe = load_probe(PROBE_DIR, device="cpu")
        assert list(self.probe.layers) == LAYERS, self.probe.layers
        self.torch = torch

    def _render(self, job: dict) -> list:
        from types import SimpleNamespace

        from acts import render_completion, window_masks
        from probe_inference.score import build_read_mask

        if job.get("self"):
            # Self read (no follow-up): the conversation through U's step, read at the token
            # before that assistant turn's <|im_end|> -- U's own step, on-policy.
            assert job["messages"][-1]["role"] == "assistant", job["messages"][-1]["role"]
            full = self.tok.apply_chat_template(job["messages"], tools=job["tools"],
                                                tokenize=False, **TEMPLATE_KWARGS)
            if not full.endswith("<|im_end|>\n"):
                raise ValueError(f"unexpected template ending {full[-40:]!r}")
            ids = self.tok(full[:-1], add_special_tokens=False)["input_ids"]
            if self.tok.convert_ids_to_tokens(ids[-1]) != "<|im_end|>":
                raise ValueError("the rendered step does not end at <|im_end|>")
            return [(SimpleNamespace(ids=ids, n_tokens=len(ids)), len(ids) - 2)]

        out = []
        for rd in job["reads"]:
            r = render_completion(
                self.tok, [*job["messages"], {"role": "user", "content": rd["question"]}],
                rd["answer"], job["tools"], TEMPLATE_KWARGS)
            pm, cm = window_masks(r)
            mask = build_read_mask(self.probe.read_window, pm, cm, self.torch.tensor(0))
            (pos,) = (mask.nonzero().flatten() + r.followup_start).tolist()
            out.append((r, pos))
        return out

    def _score(self, jobs: list[dict]) -> list[dict]:
        import time

        from lens_hook import capture_last
        from vllm import SamplingParams, TokensPrompt
        from vllm_lens import Hook

        t0 = time.time()
        rendered = [self._render(j) for j in jobs]
        flat = [x for rs in rendered for x in rs]
        prompts = [TokensPrompt(prompt_token_ids=r.ids[: pos + 1]) for r, pos in flat]
        params = [SamplingParams(max_tokens=1, temperature=0.0,
                                 extra_args={"apply_hooks": [Hook(fn=capture_last,
                                                                  layer_indices=LAYERS)]})
                  for _ in flat]
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        scores = []
        for (r, pos), o in zip(flat, outs):
            saved = o.hook_results["0"]
            for L in LAYERS:
                if saved[f"n{L}"] != pos + 1:
                    raise RuntimeError(f"layer {L}: hook saw {saved[f'n{L}']} tokens, "
                                       f"prompt has {pos + 1}")
            acts = {L: saved[f"L{L}"][None, :] for L in LAYERS}
            scores.append(float(self.probe.score(acts, self.torch.tensor([True]))))
        seconds = round(time.time() - t0, 2)
        results, k = [], 0
        for rs in rendered:
            results.append({"scores": scores[k:k + len(rs)],
                            "n_tokens": [r.n_tokens for r, _ in rs], "seconds": seconds})
            k += len(rs)
        return results

    @modal.method()
    def score_jobs(self, jobs: list[dict]) -> list[dict]:
        """Bulk reads through the Modal SDK (score_probe.py): no web-request window."""
        return self._score(jobs)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def read(self, body: dict) -> dict:
        return self._score([body])[0]

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def read_batch(self, body: dict) -> dict:
        return {"results": self._score(body["jobs"])}
