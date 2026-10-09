# iac — RunPod serverless infra (Pulumi)

Pulumi program managing a RunPod **serverless vLLM endpoint** serving
`Qwen/Qwen3.5-27B` with an OpenAI-compatible API. Scale-to-zero
(`workersMin=0`): costs nothing while idle.

## Layout

- `__main__.py` — the stack: one template + one endpoint; all parameters
  (model, image, GPU tiers, workers, context length) are constants at the top.
- `runpod_rest.py` — Pulumi dynamic providers over RunPod's REST API
  (`rest.runpod.io/v1`). The official `runpodinfra` provider is NOT used: its
  `saveEndpoint` GraphQL mutation declares `$scalerValue: Int` where RunPod's
  API requires `Float`, so every endpoint create fails (v1.9.99, 2026-10).
  This is our own diagnosis (original research, not from any doc) — repro and
  details in the issue we filed:
  <https://github.com/runpod/pulumi-runpod-native/issues/35>. Revisit if
  upstream fixes it.
- Deps are uv-managed (`pyproject.toml` / `uv.lock`, own venv — separate from
  the repo's two ML envs). `Pulumi.yaml` sets `toolchain: uv`, so `pulumi`
  drives uv itself.

## Usage

```sh
export PATH=$HOME/.pulumi/bin:$PATH   # pulumi CLI (installed via get.pulumi.com)
export PULUMI_CONFIG_PASSPHRASE=""    # local file backend, no secrets in config
set -a; source ../.env; set +a        # RUNPOD_API_KEY
cd iac
pulumi preview   # diff
pulumi up        # apply
pulumi destroy   # tear down endpoint + template
pulumi stack output openai_base_url
```

State: **repo-local file backend, git-tracked** — `Pulumi.yaml` pins
`backend: url: file://./state`, so the state JSON lives at
`iac/state/.pulumi/stacks/runpod-iac/dev.json` and travels with the repo (no
`pulumi login` needed; `PULUMI_CONFIG_PASSPHRASE=""`). This is deliberate:
single-operator stack, and the state contains no secrets — the API key is
read from the environment at operation time and never written to state.
Discipline it requires: **commit after every `pulumi up`**, and never apply
from a second clone without pulling first (git is the lock). Backup/lock
churn (`*.bak`, `locks/`, `history/`) is gitignored. The pre-migration copy
remains under `~/.pulumi` (stale as of 2026-10-08).

## Current deployment (2026-10-08)

- endpoint `jd42hxujb3bagp` → `https://api.runpod.ai/v2/jd42hxujb3bagp/openai/v1`
- template `7us1hklk1i`, image `runpod/worker-v1-vllm:v2.28.0` (vLLM 0.30.0,
  native qwen3_5 kernels; bumped 2026-10-08 from `v2.5.0stable`, whose
  vLLM 0.8.x never served a request)
- 1x 80 GB GPU per worker (A100 preferred, then H100), max 2 workers,
  FlashBoot on, `MAX_MODEL_LEN=32768`

### Qwen3.8-27B generation endpoint, 4x A40 (2026-10-09)

- endpoint `9pb82nk3f4pon8` (`qwen38-27b-vllm`) →
  `https://api.runpod.ai/v2/9pb82nk3f4pon8/openai/v1`, model id `Qwen/Qwen3.8-27B`
  (revision pinned), text-only, tool calling (`qwen3_xml`) + reasoning (`qwen3`)
  parsers, prefix caching, 262,144-token context, up to 64 requests in flight.
- 4x A40 per worker (tensor parallel 4), CA-MTL-1 only, max 1 worker,
  `idleTimeout` 600 s. **~$4.9/h while a worker is up** (4 x $1.22 serverless).
- network volume `3c1zxz41vt` (`qwen38-27b-model-cache`, 100 GB, CA-MTL-1) holds
  the weights in the worker's HF cache layout (`huggingface-cache/hub`).
- Measured 2026-10-09: cold start ~3.5 min in the worker; 5 of 5 cold requests
  answered 3.7-5.5 min after submit; 204 / 556 / 780 generated tok/s total at 8 / 32 / 64
  concurrent requests (~25 tok/s for a single stream).

The first request after scale-to-zero must be async (`/run` + `/status`): the
sync `/openai/...` route sits behind a ~100 s edge timeout and 524s on a cold
worker. Once warm, use the OpenAI route.

**Pre-loading the volume** (needed once per volume, e.g. after `pulumi destroy`
or a new model revision). The worker cannot download 55.6 GB itself: RunPod
stops a serverless container that has not reported ready ~9.5 min after start.
Run a throwaway pod on the volume and delete it afterwards:

```sh
runpodctl pod create --name qwen38-cache-loader --gpu-id "NVIDIA A40" \
  --image python:3.12-slim --data-center-ids CA-MTL-1 \
  --network-volume-id <volume id> --volume-mount-path /workspace --ssh=false \
  --docker-args "bash -c 'pip install -q huggingface_hub hf_transfer hf_xet; \
    HF_HUB_CACHE=/workspace/huggingface-cache/hub HF_HUB_ENABLE_HF_TRANSFER=1 \
    hf download Qwen/Qwen3.8-27B --revision <sha>; echo LOADER_DONE; sleep 3600'"
runpodctl pod logs <pod id>      # wait for LOADER_DONE (~1 min for 52 GB)
runpodctl pod delete <pod id>
```

Debugging a worker that never becomes ready: `runpodctl serverless logs
<endpoint id> --follow` (v2.10+). `py-spy` is refused in the container (no
ptrace); a `sitecustomize.py` on `PYTHONPATH` calling
`faulthandler.dump_traceback_later(200, repeat=True)` prints every process's
stacks into the same logs.

### Probe classifier endpoints (2026-10-07)

Serverless probe read-outs (`serving/README.md`), image
`runpod/worker-v1-vllm:v2.28.0` (vLLM 0.30.0 — native qwen3_5 Triton GDN
kernels, no pytorch-reference fallback), `RUNNER=pooling`, repo plugin
pip-installed at container start, max 1 worker each (account quota 5):

- `qwen35-27b-probe-l16-logistic` `4jg7t3qd7xasel` —
  `reinthal/qwen3.5-27b-deception-probe-l16-logistic-regression` (MEAN pooling)
- `qwen35-27b-probe-dyl-l18-logistic` `c97e4gem6s1icu` —
  `reinthal/qwen3.5-27b-deception-probe-dyl-l18-logistic-regression` (LAST/"No" token)

```sh
curl -s https://api.runpod.ai/v2/<endpoint_id>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"route": "/classify", "method": "POST",
       "body": {"input": ["<rendered text, see model card>"]}}}'
```

Smoke test (cold start pulls ~55 GB of weights — first request takes minutes
and bills GPU time):

```sh
curl https://api.runpod.ai/v2/jd42hxujb3bagp/openai/v1/chat/completions \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.5-27B", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 20}'
```

## Gotchas (cost us a debugging round each)

- **`runpod/worker-v1-vllm:v2.28.0` prints `HF_TOKEN` in plain text** in its
  launch line (`--hf-token ...`) in the worker logs on every start. Fixed
  upstream in v2.29.0, which had no Docker Hub tag on 2026-10-09 (the Hub
  builds it from source). Redact when sharing logs; rotate the token if logs
  were shared.
- A serverless container that has not started the RunPod handler ~9.5 min after
  start is stopped and recreated ("stop container" in the system log). The vLLM
  worker only starts the handler once vLLM is healthy, so anything slow or hung
  at startup shows up as a crash loop with jobs stuck `IN_QUEUE`.
- Multi-GPU (tensor parallel) on serverless: NCCL's default P2P transports hung
  in vLLM's communicator warm-up, always (CUMEM) or intermittently (IPC). The
  Qwen3.8 template forces shared memory (`NCCL_P2P_DISABLE=1`,
  `NCCL_CUMEM_ENABLE=0`) and disables vLLM's custom all-reduce.
- Multimodal checkpoints: vLLM profiles the vision encoder at startup (minutes
  on this model). `--language-model-only` skips it.
- The v2.28.0 worker maps env vars to `vllm serve` flags from a fixed table; a
  flag missing from it (e.g. `--language-model-only`) is silently ignored as an
  env var and must go through `VLLM_EXTRA_ARGS`.
- New workers pick up a changed template on their next container start, not
  immediately.

- Pulumi's property protocol delivers every number as a float; RunPod's Go
  API rejects `100.0` for `int32` fields — `runpod_rest._canon` converts
  integral floats back before sending.
- RunPod template names are unique account-wide; the Template provider uses
  delete-before-replace for that reason.
- Endpoint updates PATCH in place, so the endpoint id (and base URL) survives
  parameter changes; template changes replace the template.
