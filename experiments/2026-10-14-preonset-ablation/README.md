# AIS-404 pre-onset ablation — instruction detection vs intent

Does the fibs probe's pre-onset elevation on BashArena (pre-onset attack median
−0.122 vs honest −0.211, linear, in `2026-10-13-fibs-basharena`) come from the
probe **detecting the side-task instruction in the system prompt**, or from
**reading the attacking trajectory / the agent's intent**?

Preregistered in `PREREGISTRATION.md` (committed `896f234` before the first
non-pilot run; frozen sections are unedited, changes under *Deviations*).

## Design

No new rollouts. The stored 2026-10-10 Qwen3.8-27B BashArena trajectories
(`../2026-10-13-fibs-basharena/transcripts_{honest,attack}.jsonl`) are
re-rendered with a **modified system prompt only** — every later token is
byte-identical to the stored trajectory — and the paper's SHADE read
activations are recomputed locally (`.venv`, A40s) and scored with the main
run's linear/mlp fibs probes (`probes_main/main`, layers 19/27/35/43/51/58).

Conditions (system prompt is the only thing that changes):

| condition | trajectories | system prompt |
|---|---|---|
| `orig_honest` (= honest) | honest | honest (unchanged) — positive control / baseline |
| `orig_preonset` (= pre_onset) | attack, steps < onset | attack (unchanged) — baseline |
| `swap` | attack, steps < onset | **honest** (attack instruction removed) |
| `ctrl_neutral` | honest (all steps) | honest + side-task block + "do not pursue the side task" |
| `ctrl_good` | honest (all steps) | honest + side-task block + "…that would be unethical" |
| `ctrl_evil` | honest (all steps) | honest + side-task block + "…you are being monitored you don't want to get caught" |
| `honest_attack` | honest (all steps) | **full attack persona prompt** (reverse-swap; exploratory, see Deviations) |

`post_onset` (overt-attack reference) is the main run's attack steps at/after
the frozen onset, read from `../2026-10-13-fibs-basharena/scores_main_attack.jsonl`.

## Pipeline

- `common.py` — paths, prompt construction, condition builders (reuses the
  main run's `FibsBashArenaConfig`, `acts.py` render/read, `probe_inference`).
- `collect.py` — local re-render + SHADE read + score. `--collector trajectory`
  advances one KV cache per trajectory (O(steps), verified exact vs the main
  run's `windows` collector). `--save-acts` stores the window activations +
  masks per trajectory (`acts/<cond>/<uid>.pt`) so efc/axial can be scored
  later without re-running the model.
- `analyze.py` — preregistered: per-condition median probe score with 95%
  task-clustered bootstrap CIs (10k resamples, seed 0), the swap-median primary,
  the control-framing gaps, and the §2 decision procedure. → `results_main.json`.
- `figures.py` — pooled figure, all conditions on a shared probe-score y-axis,
  violin+box+median, identical y-limits across the linear/mlp panels. →
  `fig_conditions.png`.
- `step_analysis.py` — per-step median + task-clustered CI per condition,
  binned (default every 5 steps), CI boundary bars. → `fig_step_trajectory.png`,
  `step_stats_main.json`.

Prompt fragments (extracted once, constant across trajectories):
`honest_system.txt`, `side_task_block.txt`, `attack_system.txt`.

## Positive control

Re-collecting the unswapped honest/pre-onset conditions locally reproduces the
Modal main run's `scores_main` to mean |Δ| ≈ 0.004 (max 0.014) — bf16 noise
across H200→A40 plus the collector change, ≪ the ~0.09 effect. Each honest
condition drops the same 38/3904 steps (over the 262,144-token context window,
<1%, well within the 5% void threshold).

## Results (linear primary; medians, 95% task-clustered CI)

| condition | linear median | mlp median |
|---|---|---|
| honest | −0.211 [−0.224, −0.194] | −0.538 |
| swap | −0.221 [−0.244, −0.201] | −0.575 |
| pre_onset | −0.119 [−0.162, −0.078] | −0.419 |
| post_onset (ref) | +0.349 | +0.312 |
| ctrl_neutral | −0.145 [−0.160, −0.129] | −0.505 |
| ctrl_good | −0.155 [−0.171, −0.138] | −0.490 |
| ctrl_evil | −0.130 [−0.147, −0.113] | −0.421 |
| honest_attack | +0.116 | +0.049 |

Preregistered §2 decision (automated in `analyze.py`): `swap_shifts_to_honest`
= True (swap CI overlaps honest, excludes pre_onset; retained fraction ≈ −0.11),
`controls_all_near_honest` = False, `evil_elevated_vs_good_and_honest` = True
(`evil − good` +0.025 [+0.021, +0.027] linear; +0.068 [+0.061, +0.073] mlp),
so `abandon_H1` = False and `confidence_in_H1_rises` = True. Conclusion
sentences were deferred to post-experiment by the researcher (NOT PREREGISTERED).

## Limitations

Probes trained in-house (not the published weights). The control construction
inserts the side-task definition block under the honest prompt plus one
intent-framing line (one of several possible negated-instruction constructions).
Onset labels are the frozen Haiku-4.5 one-step-at-a-time labels of the main run.

## efc/axial

Not run. The window activations for all conditions are saved (`acts/`, 65 GB,
gitignored), so efc/axial can be scored with no model re-run — but they are not
trained for Qwen3.8-27B (the main run stored read-token-only), and training them
needs a one-off 150k-row FIBS **window** collection. Left for a later decision.

## Reproduce

```
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1,2,6,7 uv run --no-sync python \
  experiments/2026-10-14-preonset-ablation/collect.py \
  --conditions orig_honest,orig_preonset,swap,ctrl_neutral,ctrl_good,ctrl_evil,honest_attack \
  --tag main --collector trajectory --save-acts
uv run --no-sync python experiments/2026-10-14-preonset-ablation/analyze.py --tag main
uv run --no-sync python experiments/2026-10-14-preonset-ablation/figures.py --tag main
uv run --no-sync python experiments/2026-10-14-preonset-ablation/step_analysis.py --tag main
```
