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
  - `HFBackend` (`--backend hf`) ONLY when you need residual-stream steering
    (`attach_steering`) or must share weights/env with activation collection.
  - Both backends render the chat template with thinking enabled and
    force-close unclosed `<think>` blocks (`backend.n_force_closed`).
  - Callers of `VLLMBackend` need an `if __name__ == "__main__":` guard —
    vLLM spawns worker processes that re-import the main module.
- **Transcript viewer**: `rollouts/viewer/index.html` — static, reusable;
  reads `transcripts_/games_/scores_/auditor_*` straight from any data dir.
  `python3 -m http.server 8000 --bind 127.0.0.1` at repo root, then open
  `http://localhost:8000/rollouts/viewer/?data=/experiments/<date>`
  (add `&arms=a,b` if the server has no directory listing). New experiments
  keep the same file names/columns and get the viewer for free.
- **Probes**: `linear_probes/` (config, backends, probes, train). Activation
  collection for scoring lives in `experiments/2026-09-22/common.py`
  (`encode_with_spans`, `collect_activations`) — note it runs the bare
  decoder because full-model forwards CUDA-launch-fail at long sequences.

## Two virtualenvs (do not merge them)

| env | run with | use for |
|---|---|---|
| `.venv` | `uv run --no-sync python ...` | probes, activations, HF generation, analysis (torch 2.14+cu126, transformers 5.17) |
| `.venv-vllm` | `PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python ...` | vLLM generation only (vllm 0.21.0+cu129, torch 2.11+cu126) |

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
- `CLAUDE.md` is a symlink to this file.
