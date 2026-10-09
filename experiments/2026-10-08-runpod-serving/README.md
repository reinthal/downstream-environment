# 2026-10-08 — RunPod serving smoke test (engineering)

Does the `iac/` RunPod deployment serve what we think it serves?

- **Generation**: `Qwen/Qwen3.5-27B` behind the OpenAI-compatible API
  (`/openai/v1/chat/completions`), one 20-token "ping".
- **Probes**: the L16 LR span probe and the L18 DYL probe, each called with
  its model card's "remote /classify" snippet (render client-side with the
  repo's tokenizer, POST `{URL}/classify`). On RunPod, `URL` is
  `https://api.runpod.ai/v2/<endpoint_id>/openai` plus the
  `Authorization: Bearer $RUNPOD_API_KEY` header. Also called through the
  `/runsync` route envelope from `iac/README.md` (batch of 2).

`run.py` reads the deployed endpoint/template config and live health from
RunPod's REST API, runs the requests, and writes `results.json`. No GPU or
ML env needed:

```sh
set -a; source .env; set +a
uv run --no-project --with requests --with transformers --with jinja2 \
    python experiments/2026-10-08-runpod-serving/run.py
```

## Deliberate limitations

- Smoke test only: one example per probe (the model-card dialogue). Score
  parity with the HF probe path was measured in `experiments/2026-10-07*`.
- A generation timeout is recorded as a result, not diagnosed — worker logs
  are not reachable through the REST API.
- The model cards' local `vllm serve` / offline `LLM(...)` snippets are not
  run here (no GPU on this host); only the remote path is tested.
