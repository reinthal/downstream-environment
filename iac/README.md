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

State: local file backend (`pulumi login --local`, lives under `~/.pulumi`),
stack `dev`. The API key is read from the environment at operation time and is
never written to state.

## Current deployment (2026-10-07)

- endpoint `jd42hxujb3bagp` → `https://api.runpod.ai/v2/jd42hxujb3bagp/openai/v1`
- template `7us1hklk1i`, image `runpod/worker-v1-vllm:v2.5.0stable-cuda12.1.0`
- 1x 80 GB GPU per worker (A100 preferred, then H100), max 2 workers,
  FlashBoot on, `MAX_MODEL_LEN=32768`

Smoke test (cold start pulls ~55 GB of weights — first request takes minutes
and bills GPU time):

```sh
curl https://api.runpod.ai/v2/jd42hxujb3bagp/openai/v1/chat/completions \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.5-27B", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 20}'
```

## Gotchas (cost us a debugging round each)

- Pulumi's property protocol delivers every number as a float; RunPod's Go
  API rejects `100.0` for `int32` fields — `runpod_rest._canon` converts
  integral floats back before sending.
- RunPod template names are unique account-wide; the Template provider uses
  delete-before-replace for that reason.
- Endpoint updates PATCH in place, so the endpoint id (and base URL) survives
  parameter changes; template changes replace the template.
