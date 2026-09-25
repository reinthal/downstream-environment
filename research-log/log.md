# Research log

<!-- Entry format: logs/research_log.md template. Expected/Actual Outcome are
     the researcher's to write — Claude fills Context/Experiment/Parameters/Artifacts only. -->

## 2026-09-20

### Context

Goldowsky-Dill et al. 2025 (*Detecting Strategic Deception Using Linear
Probes*, arXiv:2502.03407) train linear probes on Llama-3.3-70B-Instruct
residual-stream activations using generic instructed honest/dishonest pairs,
and report they transfer to strategic-deception settings such as sandbagging.
This is the first experiment run with the `linear_probes` package in this repo
(local backend), replicating the paper's instructed-pairs probe end-to-end on
local GPUs.

### Experiment

Train the paper's instructed-pairs probe locally and evaluate on the paper's
own pre-graded sandbagging rollouts; fit a difference-of-means probe on the
same activations for comparison.

- Training data: `repe_honesty__plain` — 306 true facts, each[text](.) under an honest
  and a dishonest instruction (612 conversations; label 1 = dishonest
  instruction).
- Eval data: `sandbagging_v2__wmdp_mmlu__llama-70b-3.3` rollouts (vendored
  from ApolloResearch/deception-detection), grades honest 512 / ambiguous 64 /
  deceptive 420 (4 `skip` rows dropped; 996 total).
- Control data (follow-up run, 2026-09-22): `alpaca__plain__llama-70b-3.3`
  rollouts — 9,994 benign instruction-following dialogues, scored with the
  already-fitted probes to set the paper-style decision threshold (1% FPR on
  control = 99th percentile of control scores).
- Outputs: violin plots of probe scores per grade (+ alpaca control group and
  threshold line) and honest-vs-deceptive ROC curves, for both probe types.

Deviations from the paper's pipeline: one mean-pooled activation vector per
dialogue over the whole final assistant turn (the paper scores per token and,
for sandbagging, only from `<step4>` onward, i.e. reasoning trimmed; it also
masks the last 5 words of each training fact). Binary calls originally used a
fixed 0.5 threshold; the follow-up alpaca run adds the paper's 1%-FPR control
threshold (sandbagging metrics at that threshold are in `results.json` under
`alpaca_control`).

### Expected Outcome

![alt text](./images/deception-detection-og.png)

### Actual Outcome


Shapes look the same even though we didn't fully commit to the same implementation that the paper did.
![alt text](./images/violin_logistic_regression.png)

### Parameters/Configurations

- Model: `meta-llama/Llama-3.3-70B-Instruct` (bf16), probe layer 22, weights
  truncated to decoder blocks 0–22 (`ProbeConfig.truncate_layers`); LM head
  skipped in `LocalBackend`.
- Probes: logistic regression (`reg_coeff=10`, standardized activations —
  paper's `example_results/instructed_pairs/cfg.yaml`) and difference of
  means, both from `linear_probes`.
- Collection: `max_len=2048`, batch 8, 4× A40 (`CUDA_VISIBLE_DEVICES=0,1,2,3`),
  ~8 min for all 1608 conversations. Repro:
  `CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync python experiments/2026-09-20/run_sandbagging.py`
  (vendor clone: `git clone https://github.com/ApolloResearch/deception-detection vendor/deception-detection`).
- A first run crashed with a CUDA launch failure in the LM-head matmul at long
  sequences; fixed by running the bare transformer (no LM head). Recoverable
  allocator OOM-retry warnings remain at the longest batches.
- Control run: `uv run --no-sync python experiments/2026-09-20/add_alpaca_control.py`
  (~35 min for 9,994 dialogues on the same 4 GPUs; reuses the saved probes, so
  no retraining).

### Artifacts

- `experiments/2026-09-20/run_sandbagging.py` — experiment script
- `experiments/2026-09-20/results.json` — metrics summary (AUROC, balanced
  accuracy, mean score per grade, class counts)
- `experiments/2026-09-20/scores.csv` — per-rollout grade + both probes' scores
- `experiments/2026-09-20/scores_alpaca.csv` — both probes' scores on the
  9,994 control dialogues
- `experiments/2026-09-20/add_alpaca_control.py` — control-set follow-up script
- `experiments/2026-09-20/probe_logistic_regression.npz`,
  `experiments/2026-09-20/probe_diff_of_means.npz` — fitted probes
- `experiments/2026-09-20/figures/violin_logistic_regression.png`
- `experiments/2026-09-20/figures/violin_diff_of_means.png`
- `experiments/2026-09-20/figures/roc_both.png`

## 2026-09-22

### Context

Downstream-detector MVP from `experiments/2026-09-22/Lie Detector Week 1 MVP
Planning.md`: instead of scoring probes against graded deception labels, test
whether a probe changes a *downstream outcome* — the win rate of an outside
auditor in the Carrot-Parsnip social deduction game (`game/Carrot-Parsnip`).
The game is cut short before the elimination vote: agents only discuss, and a
separate auditor with no access to any player's role instruction makes the
only Parsnip identification. Second question (added mid-design): does steering
agents along the probe direction change the auditor's win rate?

### Experiment

100 discussion-only games (5 seed groups x 20 games) with all five players
played by Qwen3.5-27B; 3 rounds, randomized speaking order, one generation per
turn (private `<think>` reasoning, post-think text posted verbatim as the
public announcement). Instructed-pairs probes (repe_honesty, as 2026-09-20)
trained on Qwen3.5-27B activations score every announcement by re-encoding its
exact generation context and mean-pooling the probe layer over the
announcement tokens. A Qwen3.5-27B auditor then reads each game's public
transcript under four conditions — no probe, logistic-regression scores,
diff-of-means scores, random-direction scores (identical prompt in all three
probe conditions) — and names the Parsnip in a forced choice. Primary metric:
correct identifications / 100 per condition (Wilson CIs, exact McNemar vs no
probe, per-seed breakdown).

Steering arm: the same 100 games re-rolled with the raw diff-of-means
direction added to the residual stream at the probe layer during each
speaker's generation — Parsnip -8v (toward honest), Carrots +8v (toward
deceptive), 8 = eight honest<->dishonest class gaps (|v| = 1.8 vs residual
norm ~53; generations stayed coherent at +-16 in calibration). Steered
messages are scored from unsteered re-encodes (behavioural readout), and the
same 4-condition auditor pass is run on the steered transcripts.

Design choices and deliberate scope cuts are listed in
`experiments/2026-09-22/README.md`.

**Black-box judge follow-up (2026-09-24).** Every base-arm announcement
(1500) graded with the Apollo roleplaying judge (verbatim
`grading_template.txt`: 7-point honesty scale, 1-2 honest / 6-7 deceptive /
else ambiguous), primary judge GPT-4o via OpenRouter (temperature 0, ~22 s
for all 1500), local Qwen3.5-27B judge kept as an agreement check. Black box:
the judge sees game rules + prior public transcript, never roles (deviation
from Apollo, whose judge knows the speaker's private incentive).
Pre-registered rule: reject "the LR probe can be used for Carrot-Parsnip"
if LR full-span AUROC on judge-honest vs judge-deceptive < 0.7.

Black-box result: **undecidable — GPT-4o assigned zero deceptive labels**
(honest 1147 / ambiguous 352 / skip 1; Parsnip messages: 222 honest, 78
ambiguous, 0 deceptive). Without ground-truth roles a judge cannot call an
identity claim a lie, and the Parsnip wins by omission/blending, not by
gradeable false assertions — so the honest-vs-deceptive AUROC does not exist
in this setting. Computable diagnostics (full span, both probes):
parsnip-vs-carrot AUROC ~0.61 (as before), judge-honest-vs-ambiguous AUROC
~0.42 (probes do NOT track judge-perceived evasiveness), ~7% of judge-honest
game messages flagged at the alpaca 1%-FPR threshold (distribution shift vs
control). Figures: `figures/violin_judge_{lr,dom}_full.png`.

**Oracle-judge rerun (same day), both arms.** Judge switched to
`deepseek/deepseek-v4.1-flash` (temperature 0, hybrid reasoning disabled —
with reasoning enabled the 600-token budget produced empty content), scenario
now reveals the speaker's secret role (Apollo-faithful: their judge knows the
speaker's private incentive). The oracle judge is essentially a role oracle:
base-arm Carrots 1165 honest / 0 deceptive, Parsnip 1 honest / 136 ambiguous
/ 163 deceptive (mean judge score 1.15 vs 5.56). Pre-registered rule on the
base arm: **REJECTED — LR full-span AUROC on judge-honest vs judge-deceptive
= 0.659 < 0.7** (DoM 0.662; recall at the alpaca 1%-FPR threshold 11.7%,
while flagging 8-9% of judge-honest game messages — the threshold does not
transfer to game text). Steered arm: judge labels shift in the steered
directions but only slightly (Parsnip mean 5.56→5.40, deceptive 163→138;
Carrot mean 1.15→1.19 with 3 deceptive), while probe discrimination collapses
below chance (parsnip-vs-carrot 0.44, judge h-vs-d 0.54) — steering moved the
probe read-out far more than it moved judge-visible behaviour. Figures:
`figures/violin_judge_{base,steered}_oracle_{lr,dom}_full.png` and ROC curves
`figures/roc_judge_{base,steered}_oracle.png`; results in
`results_judge_{base,steered}_oracle.json`. `judge_messages_api.py` also
gained an OpenRouter Batch API mode (`--batch submit|poll`, 50% token price,
24 h window) for future large judging runs.

### Expected Outcome

<!-- researcher fills in -->

### Actual Outcome

<!-- researcher fills in; headline numbers are in experiments/2026-09-22/results.json
     and figures/auditor_accuracy.png -->

### Parameters/Configurations

- Model (players + auditor): `Qwen/Qwen3.5-27B` (bf16, local transformers;
  `flash-linear-attention` installed for the hybrid linear-attention kernels).
- Probes: logistic regression (`reg_coeff=10`, standardized) and diff of
  means from `linear_probes`, layer 16 of 64 (chosen from {16,24,32,40,48} by
  held-out instructed-pairs AUROC — all candidates saturated at 1.0; 16 is
  closest to the paper's fractional depth 22/80). Random probe: unit
  Gaussian direction per seed group, calibrated to zero-mean/unit-spread
  projections on the training activations.
- Generation: temperature 0.7, top_p 0.95, max 1800 new tokens per turn +
  bounded force-close of unclosed think blocks (220 tokens); auditor 2000.
- Score shown to the auditor: probe pooled over the FULL assistant turn
  (private think + announcement; `*_full` columns). Chosen after message-level
  diagnostics (announcement-only span was at chance for parsnip-vs-carrot,
  AUROC ~0.48; full span ~0.61) but before any auditor was run. Activation
  collection runs the bare decoder (no LM head) — full-model forwards CUDA
  launch-fail at long sequences, as in the 2026-09-20 run.
- Games: seeds `1000*group + idx`; roles via `CarrotParsnipGame(seed)`;
  speaking order per-game RNG, last speaker never first next round.
- Steering: coef 8.0 x raw diff-of-means vector, layer 16, forward hook on
  the decoder block output, applied to all token positions during the
  steered player's turns (prefill + decode).
- Hardware: base arm 4x A40 (chunk 25), steered arm 3x A40 concurrently.
- Runtime: ~11 h wall clock end-to-end (~10 h GPU compute). Probe training
  ~5 min; rollouts ~6.5 h per arm (concurrent) — ~19 min per speaking slot in
  round 1 growing to ~30 min in round 3, batches gated on the longest thinker
  (60-85% of turns hit the 1800-token thinking budget); scoring ~35-45 min
  per arm; auditors ~2.5 h per arm (9,024 s / 8,846 s for 400 calls, 397 and
  399 of 400 hit the 2000-token budget); analysis seconds. Main speedup
  levers if iterating: vLLM serving, smaller thinking budget, 2 replicas/arm.
- Repro: `experiments/2026-09-22/README.md` pipeline table (train_probes ->
  run_games x2 -> score_messages x2 -> run_auditor x2 -> analyze).

### Artifacts

- `experiments/2026-09-22/README.md` — design + deliberate limitations
- `experiments/2026-09-22/{common,run_games,train_probes,score_messages,run_auditor,analyze}.py`
- `experiments/2026-09-22/probes/` — fitted probes, steer vector, layer selection
- `experiments/2026-09-22/transcripts_{base,steered}.jsonl`, `games_{base,steered}.json`
- `experiments/2026-09-22/scores_{base,steered}.csv`
- `experiments/2026-09-22/auditor_{base,steered}.jsonl`
- `experiments/2026-09-22/results.json`, `figures/auditor_accuracy.png`
- Judge follow-up: `experiments/2026-09-22/{judge_messages,judge_messages_api,analyze_judge}.py`,
  `judge_base.jsonl` (Qwen), `judge_base_api.jsonl` (GPT-4o),
  `results_judge.json`, `figures/violin_judge_{lr,dom}_full.png`

## 2026-09-24 — infrastructure: LMSteering / LMProbe / ExperimentConfig

### Context

Experiments so far hand-wired generation, steering, activation collection and
probe scoring from `rollouts/generation.py`, `linear_probes/` and
`experiments/2026-09-22/common.py` (whose helpers 2026-09-24 already imported
cross-experiment). Refactor into three injectable primitives so new
experiments read as config -> primitives -> run; no scientific claims here.

### Experiment

- `ExperimentConfig` (`linear_probes/config.py`): dataclass inheriting
  `ProbeConfig`, JSON round-trip, `save(out_dir)`; every experiment folder
  carries a `config.json` (added retroactively for 2026-09-24; template in
  `experiments/_template/`).
- `LMProbe` (`linear_probes/lm_probe.py`): truncated-at-probe-layer model +
  trained probe -> `p(deceptive)` per conversation/span. Truncation now
  neutralizes the decoder's FINAL NORM: `hidden_states[-1]` is post-RMSNorm,
  so a naive truncated read returns `norm(h_L)`, not the raw stream the
  2026-09-22 probes were trained on (caught by the new truncation test; the
  2026-09-20 `LocalBackend` pipeline has the same property but is
  self-consistent train==eval, so it was left untouched).
- `LMSteering` (`rollouts/steering.py`): steered generation ON VLLM —
  forward hook shipped to workers via `LLM.apply_model`, `enforce_eager`,
  DoM-only vectors (`DiffOfMeansProbe` now stores `raw_diff` on fit; LR
  probes / bare arrays rejected). Spike on Qwen3.5-9B TP=2: coef 0
  bit-reproduces the unhooked baseline, coef 60 diverges, reset restores —
  i.e. hooks fire during decode and the coefficient is switchable.
  Constraints in AGENTS.md ("Steering on vLLM 0.21").
- Migration proof: `experiments/2026-09-24/rescore_lmprobe.py` re-scores a
  12-rollout slice via LMProbe vs the committed score_rollouts.py path.

### Expected Outcome

Old-vs-new probe scores identical on the slice (same blocks, same batching).

### Actual Outcome

Identical, bitwise: max |activation delta| = 0.0 and max |score delta| = 0.0
between the committed score_rollouts.py path (full 27B) and LMProbe
(truncated at layer 16) on the 12-rollout slice, both probes x both spans
(`rescore_lmprobe.py` prints PASS). Steering-on-vLLM spike: RESULT
coef0==base True | steered!=base True | reset==base True.

### Parameters/Configurations

- Tests: `uv run --no-sync pytest tests/` (30 pass + 1 GPU smoke, which
  skips unless CUDA and an explicit CUDA_VISIBLE_DEVICES are present).
- Spike + LMSteering smoke: Qwen3.5-9B, TP=2 on GPUs 5-6, layer 16, random
  unit direction, coefs {0, 60}.

### Artifacts

- `linear_probes/{config,activations,lm_probe,probes}.py`, `rollouts/steering.py`
- `tests/{conftest,test_experiment_config,test_lm_probe,test_steering,test_gpu_smoke}.py`
- `experiments/_template/{README.md,config.json,run.py}`
- `experiments/2026-09-24/{config.json,rescore_lmprobe.py}`
- AGENTS.md: primitives under "Reuse, don't rebuild", env table, steering
  status

## 2026-09-24

### Context

The 2026-09-22 game experiment showed no significant auditor effect from the
Qwen3.5-27B probes and only weak message-level signal (parsnip-vs-carrot AUROC
~0.61 full-span, chance on public spans), while the same instructed-pairs
recipe on Llama-3.3-70B (2026-09-20) hit AUROC ~0.999 on the paper's graded
sandbagging rollouts. Open question: is the Qwen probe itself broken, or is
the game just a hard setting? Hypothesis (researcher's): the instructed-pairs
probe does *not* work on Qwen3.5-27B but does on Llama-3.3-70B.

### Experiment

Replicate the 2026-09-20 sandbagging evaluation **on-policy for Qwen**: replay
the exact 1000 sandbagging_v2 (wmdp_mmlu) prompts and the first 2000 alpaca
control prompts behind the paper's Llama rollouts through Qwen3.5-27B
(thinking enabled, vLLM), grade each sandbagging rollout with the paper's own
rule-based grader (`SandbaggingV2Dataset.label_response`, honest / ambiguous /
deceptive / skip — this dataset's "judge" was never an LLM), then score all
rollouts with the already-fitted 2026-09-22 probes (instructed pairs, layer
16) on two spans (full turn incl. think, public post-think).

Pre-registered decision rule: REJECT the hypothesis if the LR probe's
honest-vs-deceptive AUROC (full span) exceeds 0.7; the 1%-FPR threshold from
the alpaca control (p99) and recall on deceptive at that threshold are
reported alongside, with the Llama numbers side by side.

Pilot findings that shaped the design (8-rollout smoke runs): a 1800-token
budget truncates every sandbagging scaffold before `<answer>`; the think block
mentions the grader's tag names, so grading raw text is poisoned — grades are
computed on the public post-think text; sandbag verdicts require both
`<step2-results>` and `<answer>` present (truncations grade `skip`).

### Expected Outcome

<!-- researcher fills in -->

### Actual Outcome

<!-- researcher fills in; headline numbers in experiments/2026-09-24/results.json -->

### Parameters/Configurations

- Model: `Qwen/Qwen3.5-27B` via `rollouts.generation` VLLMBackend (TP=4,
  GPUs 0-3), temperature 0.7 / top-p 0.95, max 6144 new tokens + 800-token
  force-close (`FORCE_CLOSE_TOKENS` patched up from 220 for the step
  scaffold).
- Probes: `experiments/2026-09-22/probes/probe_{logistic_regression,diff_of_means}.npz`
  reused as-is (layer 16, instructed pairs) — no retraining.
- Scoring: bare-decoder activation collection (`experiments/2026-09-22/common.py`),
  `max_len=6144`, mean-pooled full + public spans, model truncated at load to
  decoder blocks 0–16 (`num_hidden_layers = layer+1`, the LocalBackend trick;
  ~15 GB instead of ~54 GB, which allowed scoring on GPUs shared with another
  user's job — recoverable allocator OOM-retries at the longest batches).
- Runtime (wall clock): sandbagging generation ~85 min (4 chunks of 250,
  ~21 min each, ~17% force-closed), alpaca generation ~68 min (8 chunks,
  ~8.5 min each, <1% force-closed), both TP=4 vLLM incl. ~10 min engine
  start; scoring ~63 min for 1000 sandbagging (batch 4, shared GPUs) and
  ~33 min for 2000 alpaca (batch 6, exclusive GPUs); analysis seconds.
  One incident: the first full run crashed in chunk 2 — the force-close pass
  re-feeds prompt + up to 6144 generated tokens + cue into an 8192-token
  engine window — and the hung engine idled ~6 h before detection; fixed with
  `max_model_len=16384` and a log monitor for crash signatures.
- Grade counts (public-text grading): honest 479 / deceptive 333 /
  ambiguous 29 / skip 159 (Llama-70B pre-graded set: 512/420/64/4).
- Design details and deliberate limitations: `experiments/2026-09-24/README.md`.

### Artifacts

- `experiments/2026-09-24/README.md` — design, decision rule, limitations
- `experiments/2026-09-24/{run_rollouts,score_rollouts,analyze}.py`
- `experiments/2026-09-24/rollouts_{sandbagging,alpaca}.jsonl`
- `experiments/2026-09-24/scores_{sandbagging,alpaca}.csv`, `activations_*.npz`
- `experiments/2026-09-24/results.json`, `figures/violin_*.png`,
  `figures/roc_both.png`
