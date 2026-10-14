# 2026-10-14 — fibs probes without social-deduction / negotiation data on Carrot-Parsnip

## Question

The published fibs probes separate Parsnip from Carrot announcements at
0.989 (linear) in 2026-10-12, but FIBSv1 contains Among Us and Werewolf.
Is that result "not exclusively due to training on social deduction games"?
Hypothesis, the two confirmatory tests, the prediction and every threshold:
`PREREGISTRATION.md` (committed 7859aa9 before the first run).

## Design

- **Probes** (`train.py`, `fibs_probes`, the 2026-10-14-fibs-linear-no-sd
  recipe; Qwen3.5-27B read-token store of 2026-10-13-fibs-acts):
  - `no_sd`: FIBSv1 without `among_us`, `werewolf`, `negotiation` (20,110
    rows) → 129,174 training rows after per-family class balance.
  - `random_drop`: 20,110 rows dropped uniformly from all non-validation
    rows (every family) → 129,638-129,988 training rows.
  - Both: one shared 5,000-row validation hold-out (seed 0, over the rows
    outside the three families, no shared `pair_id`); linear and MLP
    ensembles; seeds 0-4 (seed = drop draw, balance, init, order). 20
    probes in `probes_main/{condition}_s{seed}/{arch}`.
- **Scoring** (`score.py`): the stored 2026-10-12 follow-up-window
  activations, no new games; contrast = score("Yes.") − score("No."), the
  2026-10-12 read. 2,400 main announcements (3p 900 + 5p 1,500, 600
  Parsnip, 200 games), all-Carrot games 3p_cc / 5p_cc (2,400), alpaca
  (2,000). The published probes' 2026-10-12 scores are carried along as a
  reference column.
- **Analysis** (`analyze.py` → `results.json`, `figures.py` →
  `figures/auroc.svg`): gap = mean AUROC(random_drop) − mean AUROC(no_sd),
  linear, pooled; 95% percentile interval over 2,000 game-bootstrap draws
  (games resampled within arm; seed variance not in the interval, seed
  spread reported).

## Results

Positive control: all 20 probes pass (fibs validation AUROC 0.9923-0.9978,
bar 0.95).

**Primary (linear, pooled, 2,400 announcements):**

| | announcement AUROC [95% game CI] | per seed |
|---|---|---|
| no_sd (mean of 5) | 0.983 [0.979, 0.987] | 0.981 0.983 0.987 0.980 0.985 |
| random_drop (mean of 5) | 0.978 [0.972, 0.982] | 0.981 0.975 0.978 0.978 0.975 |
| published (reference) | 0.989 [0.985, 0.992] | — |
| **gap = random_drop − no_sd** | **−0.006 [−0.007, −0.005]** | |

Prediction (researcher): "Dropping roleplaying AUROC 0.96 +/- 0.01 95% CI,
dropping 13K random rows 0.98 +/- 0.01 95% CI" (gap ≈ +0.02).

- **Test 1** (gap upper bound < δ = 0.10): **supported**.
- **Test 2** (gap lower bound > 0): **refuted** — the upper bound is below
  0: removing the social-deduction and negotiation families cost *less*
  Carrot-Parsnip AUROC than removing as many random rows. The no-SD
  probes score 0.983, above the predicted 0.95-0.97.

Pre-agreed conclusion sentences: NOT PREREGISTERED (the researcher did not
give them).

Secondary (single run, no correction):

| AUROC (seed mean) | no_sd | random_drop | gap [95% game CI] | published |
|---|---|---|---|---|
| linear 3p | 0.981 | 0.974 | −0.007 [−0.010, −0.005] | 0.986 |
| linear 5p | 0.986 | 0.981 | −0.005 [−0.006, −0.003] | 0.990 |
| MLP pooled | 0.944 [0.934, 0.953] | 0.932 [0.920, 0.943] | −0.012 [−0.016, −0.008] | 0.968 |
| MLP 3p | 0.937 | 0.927 | −0.010 [−0.015, −0.005] | 0.962 |
| MLP 5p | 0.952 | 0.937 | −0.015 [−0.020, −0.009] | 0.974 |

- MLP seeds spread widely: no_sd 0.927-0.956, random_drop 0.910-0.950.
- Mean-max game accuracy (seed mean): linear 1.00 in 3p and 5p for both
  conditions (published 1.00); MLP no_sd 0.99 / 1.00, random_drop 0.98 /
  0.96.
- At the alpaca 1%-FPR cutpoint (seed mean), Parsnip / Carrot / all-Carrot
  games flagged: linear no_sd 38% / 0.06% / 0.03%, random_drop 31% /
  0.06% / 0%, published 36% / 0.06% / 0%; MLP no_sd 23% / 0.11% / 0.01%,
  random_drop 21% / 0.01% / 0%, published 30% / 0% / 0%.
- On the removed FIBS families themselves the cost is large: no_sd 0.845-
  0.848 (linear, held out) vs random_drop 0.957-0.959 (mostly in training).
  The loss on FIBS's own social-deduction rows does not carry over to
  Carrot-Parsnip.

Read with care: the gap's interval holds the games fixed per draw and
leaves training-seed noise out; the seed ranges of the two linear
conditions overlap only at one point (no_sd min 0.980, random_drop max
0.981). Both in-house conditions score below the published probes (linear
−0.006 to −0.011): the in-house pipeline (split, row set) differs from the
paper's.

## Deliberate limitations

- The published probes come from a different pipeline, so they're only a
  reference (preregistered).
- Deviation: pilot scored on the 5p pilot games, not 3p (see
  PREREGISTRATION.md).

## Run order

```sh
E=experiments/2026-10-14-fibs-no-sd-carrot-parsnip
CUDA_VISIBLE_DEVICES=6 uv run --no-sync python $E/train.py                       # 20 probes, ~30 min on one A40
uv run --no-sync python $E/score.py --arms 3p 5p 3p_cc 5p_cc alpaca               # CPU, needs 2026-10-12 acts_* (gitignored)
uv run --no-sync python $E/analyze.py && uv run --no-sync python $E/figures.py
```
