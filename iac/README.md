# iac — Modal resources (Pulumi)

Pulumi program managing the Modal **Volumes** and **Secret** that
`experiments/2026-10-13-fibs-basharena/modal_app.py` mounts, in the
`cadenza` workspace. (Until 2026-10-13 this program managed RunPod
serverless endpoints; that was torn down on 2026-10-11 and removed from the
program on 2026-10-13 — the recipe is in git history, commit `93b0359`.)

## Layout

- `__main__.py` — the stack: three volumes (`qwen35-27b-weights`,
  `fibs-basharena`, `fibs-basharena-triton`) and the secret
  `fibs-basharena-hf` (`HF_TOKEN`). Names must match
  `experiments/2026-10-13-fibs-basharena/common.py`.
- `modal_res.py` — Pulumi dynamic providers over the Modal Python SDK (there
  is no official Pulumi provider for Modal). Volume creation is idempotent (a
  hand-made volume of the same name is adopted); deleting a Volume resource
  deletes its data. The secret's value is read from the environment variable
  named by `value_env` at operation time and never stored in state.
- Deps are uv-managed (`pyproject.toml` / `uv.lock`, own venv — separate from
  the repo's ML envs). `Pulumi.yaml` sets `toolchain: uv`.

## Usage

```sh
export PULUMI_CONFIG_PASSPHRASE=""        # local file backend, no secrets in config
export MODAL_PROFILE=cadenza              # the workspace; the provider refuses any other
set -a; source ../.env; set +a            # HF_TOKEN
cd iac
nix shell nixpkgs#pulumi nixpkgs#pulumiPackages.pulumi-python nixpkgs#uv --command pulumi preview
nix shell nixpkgs#pulumi nixpkgs#pulumiPackages.pulumi-python nixpkgs#uv --command pulumi up
```

Auth: the Modal token of profile `cadenza` in `~/.modal.toml`
(`modal token new --profile cadenza`).

State: **repo-local file backend, git-tracked** — `Pulumi.yaml` pins
`backend: url: file://./state`, so the state JSON lives at
`iac/state/.pulumi/stacks/modal-iac/dev.json` and travels with the repo (no
`pulumi login` needed). Single-operator stack; the state holds resource
names and volume ids only. Discipline: **commit after every `pulumi up`**,
and never apply from a second clone without pulling first (git is the
lock). Backup/lock churn (`*.bak`, `locks/`, `history/`) is gitignored.
