# Preregistration — fibs probes without social-deduction / negotiation data on Carrot-Parsnip

Filled by the `/preregister` skill from the researcher's answers, quoted
verbatim. Committed before the first non-pilot run; the sections above
"Deviations" are frozen from that commit on. Unanswered items read
`NOT PREREGISTERED`. Lines marked *(Claude)* are facts from the repo or
implementation choices, not the researcher's answers.

- Date: 2026-10-14
- Researcher: Alexander Reinthal
- Builds on: `experiments/2026-10-12-fibs-probes/` (published fibs probes on
  3p/5p Carrot-Parsnip: linear 0.989 [0.985, 0.992], MLP 0.968 [0.961,
  0.974], announcement AUROC, 2,400 announcements, 200 games; its
  supported-branch follow-up: "retrain w o social deduction game still
  good?"), `experiments/2026-10-13-fibs-acts/` (Qwen3.5-27B FIBSv1 read-token
  activations), `experiments/2026-10-14-fibs-linear-no-sd/` (no-SD linear:
  FIBS val 0.993; held-out among_us 0.910 vs 0.966 published, werewolf 0.871
  vs 0.948), `experiments/2026-10-13-fibs-basharena/` (Qwen3.8 fibs probes on
  BashArena, linear 0.963).

## 1. Question and stakes

- Decision this result informs: "The importance of social deduction games for
  probe performance on deception" and "Whether 0.989 is a valid claim".
- Claim hoped for (one sentence): "FIBS probes performance on Carrot-parsnip
  is not exclusively due to training on social deduction games."
- Next step if supported / refuted / inconclusive: NOT PREREGISTERED.

## 2. Hypothesis

- H1 (with direction): FIBS probes trained without the families "among us,
  werewolf and negotiation" keep most of their Carrot-Parsnip performance:
  the gap to probes trained without the same number of random rows is below
  0.10 (Test 1). Bold prediction, tested as well (Test 2): "dropping the
  roleplaying data will (13k rows) will have a larger effect than dropping
  13K rows at random" — gap > 0. *(Claude: the researcher's "roleplaying
  data" was clarified as among_us + werewolf + negotiation, 20,110 rows of
  the 161,864 collected, not 13k.)*
- Null: Test 1 — gap ≥ 0.10. Test 2 — gap ≤ 0.
- Rival explanation: losing data in general rather than social-deduction data
  in particular — "train a control that drops the equivalent amount of random
  rows, this controls for loosing some data, train 5 of these. no more alt
  explanation."
- Prediction for the primary metric (value, interval, confidence %): "Dropping
  roleplaying AUROC 0.96 +/- 0.01 95% CI, dropping 13K random rows 0.98 +/-
  0.01 95% CI." Basis: "I just looked at the stats from the paper. Dropping
  100K rows made the model perform slightly below .98 for all probes."
  *(Claude: implied gap ≈ 0.02. Earlier answer, superseded by this one:
  ~0.95 [0.92, 0.98] at 60%.)*
- Result that would make me abandon H1: "null is true".
- Expected baseline / control values: random-drop controls (linear) 0.98 ±
  0.01 (above). Published probes: 0.989 linear / 0.968 MLP (measured in
  2026-10-12, reference only).

## 3. Test design

- Primary metric (one): gap = B − A in announcement-level AUROC (Parsnip
  announcements positive, Carrot announcements negative; linear-ensemble
  contrast score, lie prefill minus the other, exactly as 2026-10-12), 3p
  and 5p pooled, where A = mean AUROC of the 5 no-SD linear probes and B =
  mean AUROC of the 5 random-drop linear probes ("Random-drop controls −
  no-SD").
- Secondary metrics (single run, no correction): the 3p and 5p AUROCs
  separately; the same gap for the MLP ensembles; mean-max game accuracy per
  probe condition; Parsnip / Carrot / all-Carrot-game flag rates at the
  alpaca 1%-FPR cutpoint. Published probes reported alongside as reference.
- Unit of analysis, independence, clustering: the announcement is scored;
  announcements cluster within games; the interval resamples games
  (2,000 game-bootstrap draws, recomputing B − A on each draw, 3p and 5p
  games resampled within arm). Training seeds: 5 per condition, spread
  reported.
- Arms and controls:
  - No-SD: FIBSv1 without among_us, werewolf, negotiation (out of training
    and validation), 5 training seeds (0–4), linear and MLP.
  - Random-drop control: "uniform over non-validation/test sets" — 20,110
    rows dropped uniformly at random from the non-validation rows (all
    families, SD included), 5 seeds, "Linear and MLP, 5 each". *(Claude: both
    conditions use the identical 5,000-row validation hold-out, drawn over
    the non-SD/negotiation rows with seed 0 and sharing no `pair_id` with
    training; then per-family class balance as in `fibs_probes`. Drop seed
    = training seed.)*
  - Data: the stored 2026-10-12 announcement activations (`acts_3p`,
    `acts_5p`, `acts_3p_cc`, `acts_5p_cc`, `acts_alpaca`), no new games.
- Sample size and justification: fixed by 2026-10-12 — 2,400 announcements
  (600 Parsnip), 200 games; there the linear AUROC CI half-width was ≈ 0.004
  pooled, so a 0.02 gap is resolvable unless no-SD AUROCs are much more
  dispersed.
- Fixed parameters: see `config.json` (Qwen3.5-27B rev `fc05daec`, layers
  19/27/35/43/51/58, the paper's linear / MLP recipe of
  `experiments/2026-10-14-fibs-linear-no-sd/config.json`, seeds, excluded
  families, drop size).
- Stopping rule: fixed — 20 trainings, one scoring pass.

## 4. Analysis plan

- Statistic, interval method, comparison: gap = B − A (above), percentile 95%
  interval over 2,000 game-bootstrap draws.
- Threshold for "supported": "Both confirmatory".
  - Test 1 (primary): upper 95% bound of the gap < 0.10 (δ = "0.10").
    Refuted if the lower bound ≥ 0.10; inconclusive if the interval
    contains 0.10.
  - Test 2: lower 95% bound of the gap > 0. Refuted if the upper bound < 0;
    inconclusive if the interval contains 0.
- Exclusion rules: the 2026-10-12 exclusions unchanged (games replaced
  there; announcements dropped in its `acts_*/index.json`). A probe failing
  the positive control (§5) voids the run.
- Multiple comparisons: two confirmatory tests on the same gap, no
  correction; secondaries uncorrected.
- Figure: announcement AUROC per condition (no-SD, random-drop: 5 seed dots
  + mean; published as a reference marker), linear and MLP panels with the
  same y range; gap with its interval against 0 and 0.10.
- Confirmatory analyses: Tests 1 and 2 on the linear ensemble.
- Exploratory analyses: anything else, labelled as such.

## 5. De-risking

- Pre-mortem — boring failure modes and the check for each *(Claude)*:
  activations loaded in the wrong row order after the store-layout refactor
  (positive control 1); probe layer order or read window mismatching the
  stored announcement activations (`score_probes.py` checks layers against
  the acts index); MLP training diverging (val loss in `{arch}_train.json`);
  GPU 1 occupied (check `nvidia-smi`).
- Can the metric resolve the predicted effect: 2026-10-12 pooled CI
  half-width ≈ 0.004; a predicted 0.02 gap is resolvable. Ceiling: the
  random-drop probes near 0.99 leave little room above.
- Positive control: every one of the 20 probes reaches FIBS validation AUROC
  ≥ 0.95 on its hold-out ("yes, sounds good"). No Carrot-Parsnip control on
  the random-drop probes ("no control for random probes though").
- Pilot: one seed of each condition × arch, scored on the 2026-10-12 3p
  pilot games (`acts_3p_pilot`); must run end to end and pass the positive
  control. Pilot data excluded from the analysis.
- Cost and wall-clock estimate, hard cap, abort criterion *(Claude)*: ~1
  GPU-hour on one A40 (linear ≈ 2 min per training), no API cost; abort if a
  probe fails the positive control.
- Reused primitives / new untested code: `fibs_probes` (training, saving),
  `probe_inference.load_probe`, the 2026-10-12 scoring (`score_one`) and
  bootstrap. New: random-drop selection, multi-seed driver, gap analysis.

## 6. Conclusions agreed in advance

- If supported, the report says: NOT PREREGISTERED.
- If refuted, the report says: NOT PREREGISTERED.
- If inconclusive (defined in §4 per test), the report says: NOT
  PREREGISTERED.
- Limitations stated regardless of outcome: "The published probes come from a
  different pipeline, so they're only a reference."

---

## Deviations

<!-- Append only. Each entry: date, what changed, why, decided before or
     after seeing data. -->
