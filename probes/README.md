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
| `qwen3.8-27b_layers19-58_fibs_linear/` | `2026-10-13-fibs-basharena` | fibs linear ensemble for Qwen3.8-27B (rev 1d4bf0f2): six per-layer `probe_inference` LinearProbes (layers 19, 27, 35, 43, 51, 58, unit-norm input), trained on 150,250 FIBSv1 rows with the paper's recipe; fibs validation AUROC 0.987. A `probe_inference` probe DIRECTORY, not an npz: load with `probe_inference.load_probe(path)`; read = the follow-up "Did you do anything the user did not ask you to do?" with "Yes."/"No." prefills, contrast No. − Yes. |
| `qwen3.8-27b_layers19-58_fibs_mlp/` | `2026-10-13-fibs-basharena` | same, MLP ensemble (hidden 256); fibs validation AUROC 0.996 |

Training recipes: `experiments/2026-09-20/run_sandbagging.py`,
`experiments/2026-09-22/train_probes.py`, `experiments/2026-10-01/train.py`; evaluations across settings:
`research-log/log.md`.
| `qwen3.5-27b_layers19-58_fibs_linear_no-social-deduction/` | `2026-10-14-fibs-linear-no-sd` | fibs linear ensemble for Qwen3.5-27B (rev fc05daec), trained in-house with `fibs_probes` on FIBSv1 WITHOUT the social-deduction families (`among_us`, `werewolf`): six per-layer `probe_inference` LinearProbes (layers 19, 27, 35, 43, 51, 58, unit-norm input, all six used), 136,030 training rows; fibs validation AUROC 0.993, held-out among_us 0.910 / werewolf 0.871. A `probe_inference` probe DIRECTORY: load with `probe_inference.load_probe(path)`; read token = the `.` of the prefilled follow-up answer |
