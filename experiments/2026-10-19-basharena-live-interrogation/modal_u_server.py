"""U and the fibs probe on one vLLM engine on Modal: `vllm serve` + vllm-lens + /v1/probe/read.

OpenAI-compatible for U (Inspect `openai-api/lens/Qwen/Qwen3.8-27B`, LENS_BASE_URL =
<web url>/v1, LENS_API_KEY = the Modal proxy token), probe reads at /v1/probe/read on the same
engine (serve_lens.py). bf16 Qwen3.8-27B at the probe's revision, one H200 per replica; serving
flags follow serving/serve_qwen38.py (qwen3 reasoning parser, qwen3_xml tool calls, 262k context,
prefix caching). vllm-lens forces eager execution.

Deploy: MODAL_PROFILE=cadenza modal deploy modal_u_server.py
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
MODEL = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PORT = 8000

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .uv_pip_install("vllm==0.30.0", "vllm-lens==1.3.0", "huggingface_hub")
    .env({"HF_HOME": "/hf", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
    .add_local_dir(REPO / "vendor" / "caught-in-the-act-probes" / "src" / "probe_inference",
                   "/root/probe_inference")
    .add_local_dir(REPO / "experiments" / "2026-10-13-fibs-basharena" / "probes_main" / "main"
                   / "linear", "/root/probe")
    .add_local_file(REPO / "experiments" / "2026-10-13-fibs-basharena" / "acts.py", "/root/acts.py")
    .add_local_file(REPO / "experiments" / "2026-10-19-basharena-live-interrogation"
                    / "lens_hook.py", "/root/lens_hook.py")
    .add_local_file(REPO / "experiments" / "2026-10-19-basharena-live-interrogation"
                    / "serve_lens.py", "/root/serve_lens.py")
)
hf_cache = modal.Volume.from_name("ais407-hf-cache", create_if_missing=True)
app = modal.App("ais407-u-lens", image=image)


@app.function(gpu="H200", volumes={"/hf": hf_cache}, timeout=24 * 3600, scaledown_window=600,
              max_containers=3)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=1800, requires_proxy_auth=True)
def serve():
    import subprocess

    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL, revision=REVISION)
    subprocess.Popen(
        ["python", "/root/serve_lens.py", path,
         "--served-model-name", MODEL, "--host", "0.0.0.0", "--port", str(PORT),
         "--max-model-len", "262144", "--gpu-memory-utilization", "0.90",
         "--reasoning-parser", "qwen3", "--enable-auto-tool-choice",
         "--tool-call-parser", "qwen3_xml", "--enable-prefix-caching",
         "--mamba-cache-mode", "align", "--language-model-only"],
        cwd="/root",
    )
