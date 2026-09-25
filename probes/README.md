# probes/ — fitted probes (symlinks)

Canonical entry points to every fitted probe, named
`<model>_layer<L>_<probe_type>.npz`. Each link points into the experiment
that trained it. Load with `linear_probes.probes.LinearProbe.load(path)`
(steer vectors are raw `.npz` arrays, not LinearProbes). New probes MUST be
added here (see AGENTS.md conventions).

| link | from experiment | probe |
|---|---|---|
| `llama-3.3-70b_layer22_logistic_regression.npz` | `2026-09-20` | instructed-pairs LR probe (repe_honesty, reg_coeff 10, standardized), Llama-3.3-70B layer 22 |
| `llama-3.3-70b_layer22_diff_of_means.npz` | `2026-09-20` | diff-of-means probe on the same activations |
| `qwen3.5-27b_layer16_logistic_regression.npz` | `2026-09-22` | instructed-pairs LR probe, Qwen3.5-27B layer 16 (of 64) |
| `qwen3.5-27b_layer16_diff_of_means.npz` | `2026-09-22` | diff-of-means probe on the same activations |
| `qwen3.5-27b_layer16_steer_diff_of_means.npz` | `2026-09-22` | RAW-space diff-of-means steering vector (deceptive − honest mean, |v| ≈ 1.8) for residual-stream steering at layer 16 |

Training recipes: `experiments/2026-09-20/run_sandbagging.py`,
`experiments/2026-09-22/train_probes.py`; evaluations across settings:
`research-log/log.md`.
