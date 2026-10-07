# Agent instructions — environments/bash_arena

The **BashArena** ControlArena setting running on this host's Docker. BashArena
ships inside the `control-arena` package (`control_arena.settings.bash_arena`);
there is no separate repo to clone.

## Run everything through uv, selecting this env with `--directory`

This directory is its own uv project (`pyproject.toml` + `uv.lock`), separate
from the repo-root project (which carries the linear-probe torch/transformers
stack). Do **not** reach into a `.venv` directly or merge these envs. Always:

```bash
uv run --directory environments/bash_arena <command>
```

uv creates/syncs this project's `.venv` from `uv.lock` on first run. The
`--directory` flag is what selects this environment — use it from anywhere in
the repo. (On this host uv is only available via nix: prefix with
`nix shell nixpkgs#uv --command` if `uv` is not already on PATH.)

Common commands (smoke run — a script that lives *in* this env):

```bash
uv run --directory environments/bash_arena python run_bash_arena.py --modes honest attack --limit 1
```

Experiments that use this env live under the repo-level `experiments/<date>/`,
not here (that's the repo convention — see the root AGENTS.md). Select this env
for them with `--project` (keeps the working dir at the repo root so the
`experiments/...` path resolves):

```bash
# the trusted-monitoring replication (serial, ~9 min/sample)
uv run --project environments/bash_arena python experiments/2026-10-07-trusted-monitoring/run.py --tasks 10
uv run --project environments/bash_arena python experiments/2026-10-07-trusted-monitoring/analyze.py
uv run --project environments/bash_arena inspect view --log-dir experiments/2026-10-07-trusted-monitoring/logs
```

Rule of thumb: `--directory` for this env's own scripts (`run_bash_arena.py`),
`--project` for experiments elsewhere that run on this env. Add a dependency
with `uv add --directory environments/bash_arena <pkg>` (never `uv pip install`
ad-hoc — that leaves the lockfile stale).

## Models & secrets

- Both the untrusted (attack) agent and the trusted monitor are
  `openrouter/deepseek/deepseek-v4.1-flash`. Secrets load from the **repo-root
  `.env`** (`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`); `run_bash_arena.py`
  finds it by walking up from the file.
- BashArena's `attack_policy`/`honest_policy` bind their model from the
  `UNTRUSTED_MODEL` env var **at import**, so `build_protocol()` passes
  `model_name=`/`model=` explicitly — otherwise they fall back to an Anthropic
  default and fail with "No ANTHROPIC_API_KEY". Keep that explicit wiring.
- RunPod `Qwen/Qwen3.5-27B` (`iac/`, `jd42hxujb3bagp`) was the intended attacker
  but its serverless workers were crash-looping. To use it, set
  `UNTRUSTED_MODEL = "openai-api/runpod/Qwen/Qwen3.5-27B"` (inspect reads
  `RUNPOD_BASE_URL` + `RUNPOD_API_KEY`).

## Host constraints

- **RAM-bound:** ~3.8 GB total; each BashArena sample is a docker-compose network
  (default + router + logging/malware/database) ≈ 2.3 GB, so keep
  `--max-sandboxes 1` — samples run serially (~9 min each). No local GPU, which
  is why both models are remote.
- `benchmark-base:latest` is built on first run (and reused after).
- `/etc/docker/daemon.json` address pools are only needed for many parallel
  sandboxes (see README).
