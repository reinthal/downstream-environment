# Agent instructions for this repo

Deception-probe research: linear probes on residual-stream activations,
evaluated on downstream tasks (social-deduction games with an outside
auditor). Read `research-log/log.md` for what has been run and why.

## Reuse, don't rebuild

- **Game design**: `game/Carrot-Parsnip/Carrot_Parsnip.py` is the game engine
  (seeded role assignment, players, log). New social-deduction experiments
  must reuse it — do not write a new engine. The discussion-only rollout
  harness (speaking order, private-think turns, transcripts JSONL, outside
  auditor) lives in `experiments/2026-09-22/run_games.py` / `run_auditor.py`;
  copy or import that pattern for new experiments.
- **Generation**: `rollouts/generation.py` — always use `make_backend(...)`.
  - `VLLMBackend` (`--backend vllm`, the default) for ALL bulk generation.
    ~10x faster than HF `generate`. Requires the vLLM env (below).
  - Steered generation: `rollouts/steering.py::LMSteering` (vLLM, see below).
    `HFBackend --backend hf` + `attach_steering` remains as the slow fallback
    and for sharing weights/env with activation collection.
  - Both backends render the chat template with thinking enabled and
    force-close unclosed `<think>` blocks (`backend.n_force_closed`).
  - Callers of `VLLMBackend` need an `if __name__ == "__main__":` guard —
    vLLM spawns worker processes that re-import the main module.
- **Steering on vLLM 0.21: works** (spiked 2026-09-24 on Qwen3.5-9B TP=2:
  coef 0 bit-reproduces the unhooked baseline, steered runs differ, resetting
  the coef restores the baseline). No public hook API, so `LMSteering` sends
  a forward hook to every worker via `LLM.apply_model`; the hook adds
  `coef * vec` to decoder block `layer`'s output, exactly like the HF
  `_Steerer`. Constraints, baked into `LMSteering` — keep them if you touch it:
  - engine runs `enforce_eager=True`: under torch.compile/CUDA-graphs the
    hook's coefficient is captured once and later changes are silently
    ignored (still ~2-4x faster than HF generate for bulk decode);
  - worker-bound callables cross TWO boundaries — client→engine core
    (cloudpickle, gated behind `VLLM_ALLOW_INSECURE_SERIALIZATION=1`, set by
    `LMSteering`) and engine core→workers (PLAIN pickle) — so they must be
    module-level functions or `functools.partial`s of them, never closures;
  - the coefficient is engine-global: `generate` runs one engine pass per
    distinct coef value (fine for the usual ±coef arms).
- **Transcript viewer**: `rollouts/viewer/index.html` — static, reusable;
  reads `transcripts_/games_/scores_/auditor_*` straight from any data dir.
  `python3 -m http.server 8000 --bind 127.0.0.1` at repo root, then open
  `http://localhost:8000/rollouts/viewer/?data=/experiments/<date>`
  (add `&arms=a,b` if the server has no directory listing). New experiments
  keep the same file names/columns and get the viewer for free.
- **Probes**: `linear_probes/` (config, backends, probes, train). Activation
  collection for scoring lives in `linear_probes/activations.py`
  (`encode_with_spans`, `collect_activations`, `load_truncated_decoder`;
  `experiments/2026-09-22/common.py` re-exports them) — note it runs the bare
  decoder because full-model forwards CUDA-launch-fail at long sequences.
- **Experiment primitives** — build experiments from these three; see
  `experiments/_template/` for the canonical skeleton:
  - `ExperimentConfig` (`linear_probes/config.py`) — dataclass inheriting
    `ProbeConfig`; the ONE place an experiment's hyperparameters live. Each
    `experiments/<date>/` carries a `config.json` that reproduces the run
    (`cfg.save(out_dir)` on run start); subclass it to add fields.
  - `LMProbe` (`linear_probes/lm_probe.py`, `.venv`) — model truncated at the
    probe layer + a trained probe; `score(convs, span="full"|"public")`
    returns `p(deceptive)` per conversation. `with_probe(...)` shares one
    model across read-outs. Truncation neutralizes the decoder's final norm
    so activations equal the full model's (verified in tests).
  - `LMSteering` (`rollouts/steering.py`, `.venv-vllm`) — vLLM generation
    with a diff-of-means direction added to the residual stream at the probe
    layer; same `generate(convs, ..., coefs=[...])` contract as the other
    backends. DoM-only: takes a `DiffOfMeansProbe`/its npz (fitted probes
    store `raw_diff` since 2026-09-24) or a `steer_vec` npz; rejects LR
    probes and bare arrays.

## Two virtualenvs (do not merge them)

| env | run with | use for |
|---|---|---|
| `.venv` | `uv run --no-sync python ...` | probes, activations, HF generation, analysis (torch 2.14+cu126, transformers 5.17); primitives: `LMProbe`, `ExperimentConfig` |
| `.venv-vllm` | `PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python ...` | vLLM generation only (vllm 0.21.0+cu129, torch 2.11+cu126); primitives: `LMSteering`, `ExperimentConfig` |

`ExperimentConfig` imports in both envs (no heavy deps); `LMProbe` and
`LMSteering` stay import-isolated per env — keep it that way (lazy imports,
as `rollouts/generation.py` does).

The PATH prefix matters: vLLM's spawned workers JIT-compile with `ninja`,
which lives in `.venv-vllm/bin` and is not found when the venv python is
invoked directly without it (`FileNotFoundError: 'ninja'`).

The box's NVIDIA driver is 535 (CUDA 12.x): **no cu13-built wheels run here**.
vLLM must stay at 0.21.0 with the `+cu129` wheel from the GitHub release
(the PyPI wheel of the same version is CUDA-13-built and fails with
`libcudart.so.13` missing) unless the driver is upgraded. Do not let a vLLM
upgrade replace torch in `.venv` — pyproject pins the cu126 index for a
reason.

## Hardware etiquette & gotchas

- 8x A40 (46 GB); GPU 4 is often occupied by another user's job — check
  `nvidia-smi` and set `CUDA_VISIBLE_DEVICES` around it.
- HF generation: batch 25 is the safe ceiling at long contexts on 4 GPUs
  (batch 50 OOM-thrashes). Qwen3.5-27B needs `flash-linear-attention`
  installed in `.venv` (reference kernels are ~5x slower).
- Qwen3.5 thinking runs long: budget ~1800 tokens/turn and rely on the
  backend's force-close; expect most long-context turns to hit the budget.

## Conventions

- New experiments go in `experiments/<date>/` with a README stating design
  and deliberate limitations; log every run in `research-log/log.md`
  (Claude fills Context/Experiment/Parameters/Artifacts; Expected/Actual
  Outcome are the researcher's).
- Every experiment folder carries a `config.json` (an `ExperimentConfig`,
  possibly subclassed) holding ALL of the run's hyperparameters — scripts
  read it, never hardcode; `cfg.save(out_dir)` on run start. Skeleton:
  `experiments/_template/`.
- **Every newly fitted probe gets a symlink in `probes/`** named
  `<model>_layer<L>_<probe_type>.npz` (e.g.
  `qwen3.5-27b_layer16_logistic_regression.npz`), pointing at the `.npz` in
  its experiment folder. Downstream experiments load probes via `probes/`,
  never by reaching into another experiment's directory.
- Sampled data (`transcripts_/rollouts_/judge_/auditor_*.jsonl`) is
  git-lfs-tracked via `.gitattributes`; regenerable activation `.npz` stays
  untracked (gitignored). Keep new data files to these name patterns so LFS
  picks them up.
- `CLAUDE.md` is a symlink to this file.
