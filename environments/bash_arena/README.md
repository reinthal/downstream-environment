# BashArena on this host (Docker)

Lives under `environments/bash_arena/` (sibling to the repo's `experiments/`).
Runs the **BashArena** ControlArena setting with Docker sandboxes on this box:

- **Attack / untrusted model** = `deepseek/deepseek-v4.1-flash` via OpenRouter.
- **Monitor / trusted model** = `deepseek/deepseek-v4.1-flash` via OpenRouter.

Both roles use the same OpenRouter model (the working configuration). A RunPod
`Qwen/Qwen3.5-27B` serverless endpoint (`iac/`, `jd42hxujb3bagp`) was the
intended attacker but its workers were crash-looping; to use it, set
`UNTRUSTED_MODEL = "openai-api/runpod/Qwen/Qwen3.5-27B"` in `run_bash_arena.py`
(inspect reads `RUNPOD_BASE_URL` + `RUNPOD_API_KEY`).

BashArena itself ships inside the `control-arena` package
(`control_arena.settings.bash_arena`); there is no separate repo to clone.

## Layout

- `run_bash_arena.py` — the runner (model wiring + protocol + eval).
- Experiments that use this env live under the repo-level `experiments/<date>/`
  (not here) and select it with `uv run --project environments/bash_arena`.
  First one: `experiments/2026-10-07-trusted-monitoring/` — partial replication
  of the paper's safety-vs-FPR result.
- `pyproject.toml` + `uv.lock` — this env's declared deps (`control-arena`,
  `inspect-ai`, `openai`, `numpy`, `scikit-learn`, `matplotlib`), a standalone
  uv project separate from the repo-root pyproject.
- `.venv/` — uv-managed, created/synced from `uv.lock` on first `uv run`.
  Gitignored. Don't invoke it directly; go through `uv run` (see below).

## Prerequisites (one-time)

1. **Secrets** live in the repo-root `.env`
   (`/home/kog/downstream-environment/.env`): `RUNPOD_API_KEY`,
   `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`. `run_bash_arena.py` loads them
   and derives `RUNPOD_BASE_URL` from `RUNPOD_ENDPOINT_ID`
   (default `jd42hxujb3bagp`).

2. **Docker address pools** — only needed to run *many* sandboxes in parallel
   (each sample creates 2 networks). For a handful of samples it is optional.
   To enable, put this in `/etc/docker/daemon.json` and restart docker:

   ```json
   {
     "default-address-pools": [
       { "base": "10.0.0.0/8",    "size": 27 },
       { "base": "172.16.0.0/12", "size": 27 },
       { "base": "192.168.0.0/16","size": 27 }
     ]
   }
   ```

3. **Base image** — the eval builds `benchmark-base:latest` on first run (and
   skips if it already exists). It is already built on this host.

## Run

Everything runs through `uv run`, selecting this env with `--directory` (works
from anywhere in the repo; uv creates/syncs `.venv` on first run). On this host,
prefix with `nix shell nixpkgs#uv --command` if `uv` isn't on PATH.

```bash
# honest, first task only, 1 epoch (smoke test)
uv run --directory environments/bash_arena python run_bash_arena.py --limit 1

# honest + attack, first task of each
uv run --directory environments/bash_arena python run_bash_arena.py --modes honest attack --limit 1

# view logs
uv run --directory environments/bash_arena inspect view
```

## Host constraints

- **RAM is the binding limit.** ~3.8 GB total; each BashArena sample brings up a
  docker-compose network (default 1024m + router 128m + logging/malware/database
  384m each ≈ **2.3 GB**). Keep `--max-sandboxes 1`. Larger fan-out needs more
  RAM and the address-pool config above.
- **No local GPU** — that's why both models are remote (OpenRouter).

## Known risks / things to verify

- **If you switch the attacker to RunPod** (`Qwen/Qwen3.5-27B`,
  `openai-api/runpod/...`): the endpoint is serverless and scales to zero; cold
  starts download ~55 GB and take minutes, and workers can show `throttled` (no
  80 GB GPU free) or `unhealthy` (failed resume) — this is why the default
  attacker is OpenRouter. Check
  `curl -s https://api.runpod.ai/v2/jd42hxujb3bagp/health -H "Authorization: Bearer $RUNPOD_API_KEY"`.
  `run_bash_arena.py`'s untrusted config has a 300 s timeout + 3 retries; bump it
  for cold starts. Also verify the vLLM worker actually parses **tool calls**
  (the `iac/` template sets only `MODEL_NAME`/`MAX_MODEL_LEN`) before trusting a
  run — the attack/honest policies drive `bash`/`python`/editor tools.
