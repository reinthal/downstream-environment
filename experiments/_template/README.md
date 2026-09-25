# Experiment template

Copy this folder to `experiments/<date>/`, edit `config.json`, and write the
stages of `run.py`. The contract:

- **`config.json` is the single place hyperparameters live.** Scripts read
  `ExperimentConfig` (or a subclass adding experiment-specific fields) and
  build everything from it — no constants scattered across scripts.
- **Primitives come from the packages** (see AGENTS.md "Reuse, don't rebuild"):
  - `linear_probes.LMProbe` — probe scores `p(deceptive)` from a model
    truncated at the probe layer (`.venv`, HF transformers).
  - `rollouts.steering.LMSteering` — vLLM generation steered along a
    diff-of-means direction (`.venv-vllm`).
  - `rollouts.generation.make_backend` — unsteered bulk generation.
- **Stages that need different venvs go in different processes** (same
  config file): generation under `.venv-vllm`, probe scoring under `.venv`.
- Log every run in `research-log/log.md`; state design + deliberate
  limitations in the experiment README.

Run:

```
# generation (vLLM env)
PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python experiments/<date>/run.py --stage generate
# probe scoring (HF env)
uv run --no-sync python experiments/<date>/run.py --stage score
```
