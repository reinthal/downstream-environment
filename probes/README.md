# probes/ — fitted probes (symlinks)

Canonical entry points to every fitted probe, named
`<model>_layer<L>_<probe_type>.npz`. Each link points into the experiment
that trained it. Load with `linear_probes.probes.LinearProbe.load(path)`
(steer vectors are raw `.npz` arrays, not LinearProbes); score conversations
with `linear_probes.LMProbe`, or `linear_probes.DYLProbe` for the `followup`
probes (reads the follow-up "No", not a turn span). New probes MUST be
added here (see AGENTS.md conventions).

| link | from experiment | probe |
|---|---|---|
| `llama-3.3-70b_layer22_logistic_regression.npz` | `2026-09-20` | instructed-pairs LR probe (repe_honesty, reg_coeff 10, standardized), Llama-3.3-70B layer 22 |
| `llama-3.3-70b_layer22_diff_of_means.npz` | `2026-09-20` | diff-of-means probe on the same activations |
| `qwen3.5-27b_layer16_logistic_regression.npz` | `2026-09-22` | instructed-pairs LR probe, Qwen3.5-27B layer 16 (of 64) |
| `qwen3.5-27b_layer16_diff_of_means.npz` | `2026-09-22` | diff-of-means probe on the same activations |
| `qwen3.5-27b_layer16_steer_diff_of_means.npz` | `2026-09-22` | RAW-space diff-of-means steering vector (deceptive − honest mean, |v| ≈ 1.8) for residual-stream steering at layer 16 |
| `qwen3.5-27b_layer18_followup_logistic_regression.npz` | `2026-10-01` | follow-up ("Was your last response deceptive? (Yes/No)" → "No") LR probe after Goldowsky-Dill et al. 2025: read on the "No" token, trained on graded thinking-off Apollo roleplaying replies, Qwen3.5-27B layer 18 (roleplaying val AUROC 0.887) |
| `qwen3.5-27b_layer18_followup_thinking_logistic_regression.npz` | `2026-10-01` | same, trained on thinking-on roleplaying replies (val AUROC 0.862) |

Training recipes: `experiments/2026-09-20/run_sandbagging.py`,
`experiments/2026-09-22/train_probes.py`, `experiments/2026-10-01/train.py`; evaluations across settings:
`research-log/log.md`.
