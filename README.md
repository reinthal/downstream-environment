# downstream-environment

Minimal linear-probe training on residual-stream activations. The forward pass
runs either **locally** (transformers) or **remotely on NDIF** (nnsight) —
the trainer doesn't care, the backend is injected at runtime.

Probes follow Goldowsky-Dill et al. (2025), *Detecting Strategic Deception
Using Linear Probes* ([arXiv:2502.03407](https://arxiv.org/abs/2502.03407)):

- `DiffOfMeansProbe` — direction = mean(deceptive) − mean(honest)
- `LogisticRegressionProbe` — L2-regularized logistic regression on
  standardized activations (λ = `reg_coeff`)

## Install

Deps are managed with uv (`uv.lock` is committed):

```bash
uv sync --extra ndif        # or --extra local; plain `uv sync` for probes-only (no model deps)
uv run pytest               # synthetic probe tests, no model or network needed
```

For NDIF, export `NDIF_API_KEY` (and `HF_TOKEN` for gated repos) before running.

## Use

```python
from linear_probes import ProbeConfig, make_backend, train_probe, evaluate

config = ProbeConfig(
    model_id="google/gemma-3-27b-it",
    lora_id=None,                       # optional adapter
    layer=20,                           # decoder block whose output is probed
    probe_type="logistic_regression",   # or "diff_of_means"
)

# runtime decides where the forward pass happens: "local" or "ndif"
backend = make_backend("ndif", config)

# conversations: chat `messages` lists; labels: 1 = deceptive
probe = train_probe(config, backend, train_conversations, train_labels)
probe.save("probe.npz")

metrics = evaluate(probe, backend.collect(test_conversations), test_labels)
print(metrics)   # accuracy / balanced_accuracy / auroc
```

Activations are read at `config.layer` (that block's output, i.e. the residual
stream) and mean-pooled over the final assistant turn's tokens. A saved probe
reloads with `LinearProbe.load("probe.npz")` — no sklearn needed at inference.

## Run experiments

Experiments live in `experiments/<date>/` and are logged in
`research-log/log.md`. The current pipeline (Carrot-Parsnip auditor win rate,
full commands in `experiments/2026-09-22/README.md`):

```bash
uv run --no-sync python experiments/2026-09-22/train_probes.py          # fit probes (.venv)
PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python \
    experiments/2026-09-22/run_games.py --arm base                      # rollouts (vLLM env)
uv run --no-sync python experiments/2026-09-22/score_messages.py --arm base
PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python \
    experiments/2026-09-22/run_auditor.py --arm base
uv run --no-sync python experiments/2026-09-22/analyze.py               # results.json + figure
```

Set `CUDA_VISIBLE_DEVICES` per step; see `AGENTS.md` for the two-venv setup
and hardware notes.

## Serve results

Transcripts, probe scores, and auditor verdicts are browsable with the
reusable viewer (`rollouts/viewer/`):

```bash
python3 -m http.server 8000 --bind 127.0.0.1     # from the repo root
# remote box? tunnel first:  ssh -L 8000:127.0.0.1:8000 <box>
```

Then open <http://localhost:8000/rollouts/viewer/?data=/experiments/2026-09-22>.

## Layout

```
linear_probes/
  config.py    ProbeConfig dataclass (model params, layer, probe type, hyperparams)
  backends.py  ActivationBackend interface + LocalBackend / NDIFBackend + make_backend
  probes.py    LinearProbe base + DiffOfMeansProbe / LogisticRegressionProbe + make_probe
  train.py     train_probe (collect via injected backend, fit) + evaluate
rollouts/      generation backends (vLLM / HF + steering) + transcript viewer
game/          Carrot-Parsnip game engine (reused by experiments)
experiments/   dated experiment scripts, data, and results
research-log/  one log entry per experiment
```
