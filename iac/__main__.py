"""RunPod serverless vLLM endpoint serving Qwen/Qwen3.5-27B.

Deploy:  set -a; source ../.env; set +a; pulumi up
The endpoint exposes an OpenAI-compatible API at
https://api.runpod.ai/v2/<endpoint_id>/openai/v1 (see stack outputs).
"""

import pulumi

import runpod_rest

MODEL_NAME = "Qwen/Qwen3.5-27B"
# runpod/worker-v1-vllm, pinned for reproducibility.
WORKER_IMAGE = "runpod/worker-v1-vllm:v2.28.0"  # bundles vLLM v0.30.0 (native qwen3_5 kernels)
MAX_MODEL_LEN = 32768
# 48 GB cards: in far better supply on RunPod than 80 GB A100/H100 (those
# sat `throttled` for the whole 2026-10-08 smoke test). Shared by every
# endpoint in this stack.
GPU_TYPE_IDS = [
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA L40",
    "NVIDIA L40S",
]
# bf16 27B weights are ~51 GiB — more than one 48 GB card, so the generation
# endpoint runs 2 GPUs per worker with tensor parallelism.
GEN_GPU_COUNT = 2
# Weights are downloaded onto the container disk on cold start.
CONTAINER_DISK_GB = 100

template = runpod_rest.Template(
    "qwen35-27b-vllm-template",
    body={
        "name": "qwen3.5-27b-vllm",
        "imageName": WORKER_IMAGE,
        "isServerless": True,
        "containerDiskInGb": CONTAINER_DISK_GB,
        "env": {
            "MODEL_NAME": MODEL_NAME,
            "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
            # RunPod secret reference, resolved at container start (never in state).
            "HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}",
            # CUDA-graph capture OOMed at startup on 1x80 GB (51 GiB weights;
            # worker logs 2026-10-08) and the worker crash-loops; run eager.
            "ENFORCE_EAGER": "true",
            "TENSOR_PARALLEL_SIZE": str(GEN_GPU_COUNT),
        },
        "readme": f"Serverless vLLM worker serving {MODEL_NAME} (managed by Pulumi, iac/).",
    },
)

endpoint = runpod_rest.Endpoint(
    "qwen35-27b-vllm",
    body={
        "name": "qwen35-27b-vllm",
        "templateId": template.templateId,
        "computeType": "GPU",
        "gpuTypeIds": GPU_TYPE_IDS,
        "gpuCount": GEN_GPU_COUNT,
        "workersMin": 0,  # scale to zero: no cost while idle
        "workersMax": 2,
        "idleTimeout": 5,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "flashboot": True,
    },
    opts=pulumi.ResourceOptions(depends_on=[template]),
)

pulumi.export("endpoint_id", endpoint.endpointId)
pulumi.export("template_id", template.templateId)
pulumi.export(
    "openai_base_url",
    endpoint.endpointId.apply(
        lambda eid: f"https://api.runpod.ai/v2/{eid}/openai/v1"
    ),
)


# ── probe classifiers ────────────────────────────────────────────────────────
# Truncated-decoder + probe-head models (serving/README.md). Served with the
# modern worker (vllm serve subprocess + generic route proxy): job input
#   {"route": "/classify", "method": "POST", "body": {"input": ["<text>"]}}
# on /runsync returns p(deceptive) per string. RUNNER=pooling becomes
# `--runner pooling`. The out-of-tree architecture comes from the repo's
# vllm plugin, pip-installed at container start (entrypoint override);
# --no-deps so pip can never replace the image's vLLM. Kernel note: this
# worker's vLLM (v0.30.0) has NATIVE qwen3_5 support with vendored Triton
# GDN kernels — no causal-conv1d/flash-linear-attention needed; those only
# matter on the HF-transformers fallback path, which old images (vLLM 0.8.x,
# e.g. v2.5.0stable) would hit for Qwen3.5.
PROBE_WORKER_IMAGE = "runpod/worker-v1-vllm:v2.28.0"  # bundles vLLM v0.30.0
# Tarball URL, not git+https (no git binary in the vllm-openai release
# image); python3 -m pip, not bare pip (the image bootstraps pip via
# ensurepip — the console script is not guaranteed on PATH).
PROBE_PLUGIN_INSTALL = (
    "python3 -m pip install --no-deps "
    "'https://github.com/reinthal/downstream-environment/archive/main.tar.gz"
    "#subdirectory=serving/vllm_plugin'"
)
# ~16 GB bf16 weights + GDN/KV cache at 8k context: one card of GPU_TYPE_IDS.


def probe_endpoint(slug: str, hf_repo: str) -> runpod_rest.Endpoint:
    # slug stays the Pulumi logical name (renaming it would replace the
    # endpoint and change its id); the RunPod-side name carries the model.
    name = f"qwen35-27b-{slug}"
    tmpl = runpod_rest.Template(
        f"{slug}-template",
        body={
            "name": name,
            "imageName": PROBE_WORKER_IMAGE,
            "isServerless": True,
            "containerDiskInGb": 80,
            "dockerEntrypoint": ["bash", "-c"],
            "dockerStartCmd": [f"{PROBE_PLUGIN_INSTALL} && exec python3 /src/main.py"],
            "env": {
                "MODEL_NAME": hf_repo,
                "MAX_MODEL_LEN": "8192",
                "RUNNER": "pooling",
                "GPU_MEMORY_UTILIZATION": "0.95",
                "HF_TOKEN": "{{ RUNPOD_SECRET_HF_TOKEN }}",
            },
            "readme": f"Serverless vLLM probe classifier serving {hf_repo} "
                      "(managed by Pulumi, iac/). POST /classify via the generic "
                      "route proxy; see the model card for the client contract.",
        },
    )
    ep = runpod_rest.Endpoint(
        slug,
        body={
            "name": name,
            "templateId": tmpl.templateId,
            "computeType": "GPU",
            "gpuTypeIds": GPU_TYPE_IDS,
            "gpuCount": 1,
            "workersMin": 0,
            "workersMax": 1,  # account quota is 5 workers total; the gen endpoint holds 2
            "idleTimeout": 5,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
            "flashboot": True,
        },
        opts=pulumi.ResourceOptions(depends_on=[tmpl]),
    )
    key = slug.replace("-", "_")
    pulumi.export(f"{key}_endpoint_id", ep.endpointId)
    pulumi.export(
        f"{key}_runsync_url",
        ep.endpointId.apply(lambda eid: f"https://api.runpod.ai/v2/{eid}/runsync"),
    )
    return ep


probe_endpoint("probe-l16-logistic", "reinthal/qwen3.5-27b-deception-probe-l16-logistic-regression")
probe_endpoint("probe-dyl-l18-logistic", "reinthal/qwen3.5-27b-deception-probe-dyl-l18-logistic-regression")
