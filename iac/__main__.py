"""RunPod serverless vLLM endpoint serving Qwen/Qwen3.5-27B.

Deploy:  set -a; source ../.env; set +a; pulumi up
The endpoint exposes an OpenAI-compatible API at
https://api.runpod.ai/v2/<endpoint_id>/openai/v1 (see stack outputs).
"""

import pulumi

import runpod_rest

MODEL_NAME = "Qwen/Qwen3.5-27B"
# runpod/worker-v1-vllm, pinned for reproducibility.
WORKER_IMAGE = "runpod/worker-v1-vllm:v2.5.0stable-cuda12.1.0"
MAX_MODEL_LEN = 32768
# bf16 27B weights are ~55 GB: 80 GB cards only. Order = rental priority
# (A100 tiers are cheaper than H100).
GPU_TYPE_IDS = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
]
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
        "gpuCount": 1,
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
