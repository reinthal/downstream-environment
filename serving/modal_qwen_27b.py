"""Serve Qwen/Qwen3.5-27B (and Qwen3.8-27B as ``Judge``), bf16, pinned revisions, on Modal with vLLM.

Same engine build and the same flags as the local server
(``serving/serve_qwen38.py``): reasoning split into ``reasoning_content``
(``--reasoning-parser qwen3``), tool calls (``qwen3_xml``), 262k context,
prefix caching, MTP speculative decoding. Modal's stock endpoint for this
model is the FP8 checkpoint; this app serves the full-precision one the
probes were trained on.

Differences from the box, all forced by the hardware:
  - one H200 (141 GB) per replica instead of two A40s with tensor parallel 2;
  - replicas are Modal containers behind Modal's proxy (autoscaled between
    0 and ``MAX_CONTAINERS``) instead of vLLM data-parallel workers.

    uvx modal run serving/modal_qwen_27b.py::download     # weights -> Volume (once)
    uvx modal deploy serving/modal_qwen_27b.py            # prints the URL

The deploy is declarative: the app scales to zero when idle, so leave it
deployed and change it by editing this file and deploying again.

The endpoint requires a Modal proxy token (``modal workspace proxy-tokens
create``); OpenAI clients send it as the API key: ``wk-<id>.ws-<secret>``.
The environment is the frozen ``requirements/venv-serve.lock.txt``.
"""
import json
import subprocess
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3.5-27B"
REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
# second server of the app (class Judge): the box's default Qwen3.8-27B
JUDGE_MODEL = "Qwen/Qwen3.8-27B"
JUDGE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
GPU = "H200"
MAX_CONTAINERS = 4          # replicas (one H200 each); cost scales with it
TARGET_CONCURRENCY = 64     # in-flight requests per replica the autoscaler aims for
MAX_NUM_SEQS = 128          # vLLM's own cap per replica (requests beyond it queue)
MAX_MODEL_LEN = 262144
MAX_NUM_BATCHED_TOKENS = 8192
GPU_MEMORY_UTILIZATION = 0.90
MTP_TOKENS = 3
PORT = 8000
MINUTES = 60

LOCK = Path(__file__).resolve().parents[1] / "requirements" / "venv-serve.lock.txt"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")  # torch.compile's C++ backend
    .add_local_file(LOCK, "/root/venv-serve.lock.txt", copy=True)
    .run_commands(
        "uv pip install --system --compile-bytecode -r /root/venv-serve.lock.txt "
        "--extra-index-url https://download.pytorch.org/whl/cu129 "
        "--extra-index-url https://download.pytorch.org/whl/cpu "
        "--index-strategy unsafe-best-match"
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # as on the box (no nvcc in the image either)
    })
)

weights = modal.Volume.from_name("qwen35-27b-weights", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen35-27b-vllm-cache", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": weights, "/root/.cache/vllm": vllm_cache}

app = modal.App("qwen-27b")


@app.function(image=image, volumes=VOLUMES, timeout=60 * MINUTES)
def download():
    from huggingface_hub import snapshot_download

    for model, revision in ((MODEL, REVISION), (JUDGE_MODEL, JUDGE_REVISION)):
        print("weights at", snapshot_download(model, revision=revision))
    weights.commit()


def vllm_argv(model: str, revision: str) -> list[str]:
    return [
        "vllm", "serve", model,
        "--revision", revision,
        "--served-model-name", model,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
        "--language-model-only",
        "--enable-prefix-caching",
        "--mamba-cache-mode", "align",
        "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": MTP_TOKENS}),
        "--async-scheduling",
    ]


SERVER = dict(

    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    env={"HF_HUB_OFFLINE": "1"},  # weights come from the Volume; a missing file must fail
    port=PORT,
    startup_timeout=30 * MINUTES,
    scaledown_window=10 * MINUTES,
    exit_grace_period=20 * MINUTES,  # let in-flight long generations finish on scale-down
    target_concurrency=TARGET_CONCURRENCY,
    max_containers=MAX_CONTAINERS,
)


def _start(model: str, revision: str) -> subprocess.Popen:
    argv = vllm_argv(model, revision)
    print(*argv, flush=True)
    return subprocess.Popen(argv)


@app.server(**SERVER)
class Server:
    """Qwen3.5-27B: the game players."""

    @modal.enter()
    def start(self):
        self.process = _start(MODEL, REVISION)

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.server(**SERVER)
class Judge:
    """Qwen3.8-27B: the oracle judge."""

    @modal.enter()
    def start(self):
        self.process = _start(JUDGE_MODEL, JUDGE_REVISION)

    @modal.exit()
    def stop(self):
        self.process.terminate()
