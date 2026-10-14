# iac — Modal resources (Pulumi)

Pulumi program managing the Modal **Volumes** and **Secret** that
`experiments/2026-10-13-fibs-basharena/modal_app.py` mounts, in the
`cadenza` workspace (active) and the `reinthal` workspace. (Until 2026-10-13 this program managed RunPod
serverless endpoints; that was torn down on 2026-10-11 and removed from the
program on 2026-10-13 — the recipe is in git history, commit `93b0359`.)

## Layout

- `__main__.py` — the stack: three volumes (`qwen35-27b-weights`,
  `fibs-basharena`, `fibs-basharena-triton`) and the secret
  `fibs-basharena-hf` (`HF_TOKEN`), plus the secret `fibs-basharena-b2`
  (`B2_KEY_ID`, `B2_APPLICATION_KEY`, `B2_BUCKET`, `B2_ENDPOINT`) used by
  `scripts/push_b2.py`. Names must match
  `experiments/2026-10-13-fibs-basharena/common.py`.
- `modal_res.py` — Pulumi dynamic providers over the Modal Python SDK (there
  is no official Pulumi provider for Modal). Volume creation is idempotent (a
  hand-made volume of the same name is adopted); deleting a Volume resource
  deletes its data. A secret's `values` map each key to the environment
  variable it is read from at operation time; values are never stored in state.
- Deps are uv-managed (`pyproject.toml` / `uv.lock`, own venv — separate from
  the repo's ML envs). `Pulumi.yaml` sets `toolchain: uv`.

## Usage

One stack per Modal workspace (`cadenza`, `reinthal`); the stack config
`modal-iac:profile` names the Modal profile and `MODAL_PROFILE` must match
at apply time (the provider refuses otherwise). The experiment moved from
`cadenza` to `reinthal` on 2026-10-13 when cadenza's credits ran out, and
back to `cadenza` on 2026-10-14; both stacks hold the same resources.

```sh
export PULUMI_CONFIG_PASSPHRASE=""        # local file backend, no secrets in config
set -a; source ../.env; set +a            # HF_TOKEN, B2_*
cd iac
P="nix shell nixpkgs#pulumi nixpkgs#pulumiPackages.pulumi-python nixpkgs#uv --command pulumi"
$P stack select cadenza
MODAL_PROFILE=cadenza $P preview
MODAL_PROFILE=cadenza $P up
```

Auth: the Modal token of the stack's profile (`reinthal` or `cadenza`) in `~/.modal.toml`
(`modal token new --profile <name>`).

State: **repo-local file backend, git-tracked** — `Pulumi.yaml` pins
`backend: url: file://./state`, so the state JSON lives at
`iac/state/.pulumi/stacks/modal-iac/<stack>.json` and travels with the repo (no
`pulumi login` needed). Single-operator stack; the state holds resource
names and volume ids only. Discipline: **commit after every `pulumi up`**,
and never apply from a second clone without pulling first (git is the
lock). Backup/lock churn (`*.bak`, `locks/`, `history/`) is gitignored.
