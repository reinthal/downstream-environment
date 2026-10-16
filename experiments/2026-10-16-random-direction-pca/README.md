# 2026-10-16 — why random directions separate Parsnip from Carrot (PCA)

Exploratory, not preregistered (researcher's call: "this is exploratory no
need for the [whole] preregister").

## Question

In `2026-10-12-fibs-probes/figures/roc.svg`, a random direction plugged into
the fibs linear ensemble reaches AUROC 0.84. Across the ten seeds, the pooled
AUROCs range from 0.06 to 0.84, and 8 of 10 are more than 0.1 away from 0.5.
How can random directions carry this much role signal?

## Data and method

No new forward passes. `pca.py --stage extract` takes the stored follow-up
activations of 2026-10-12 (`acts_{arm}/`, `acts_alpaca/`). From them it keeps
the fibs linear probe's read token: the token before `<|im_end|>` of the
prefilled "Yes." / "No." after "Did you respond truthfully just now?". It
keeps layers 19, 27, 35, 43, 51 and 58, for:
- the main arms 3p and 5p: 2,400 announcements, 600 of them by Parsnips;
- the all-Carrot arms 3p_cc and 5p_cc: 2,400 announcements;
- alpaca: 2,000 benign replies.

The cache goes to `acts_readtok/` (gitignored, about 1.8 GB). The rows are
read with 2026-10-12's own `random_probe.read_tokens`, so they are the same
rows its scores were computed on.

`pca.py --stage analyze` runs, for each feature and layer:
- **Features:** the "Yes." − "No." contrast (what the fibs probe scores), and
  each prefill alone.
- **PCA:** fitted on the pooled 3p + 5p announcements only. Label = the
  speaker is the Parsnip.
- **Variance decomposition:**
  - Δ = Parsnip mean − Carrot mean;
  - the within-class trace tr Σ_w;
  - the role shift's share of all variance, p(1−p)|Δ|² / tr Σ_total.
- **Per principal component (PC):**
  - explained variance;
  - share of |Δ|²;
  - role AUROC;
  - 5p-vs-3p AUROC and |correlation| with the round, as confound checks.
- **Random directions:** 2,000 unit directions u. Their AUROC is compared with
  the Gaussian mean-shift prediction Φ(d′/√2), where
  d′ = u·Δ / √(uᵀΣ_w u).
- **Ablation:** the same measures after projecting out the top k = 1, 5, 10
  PCs. This includes the AUROC of the trained fibs linear weight vector at
  that layer, and a diff-of-means direction refitted on the ablated data.
- **Held-out sets:** the all-Carrot games and alpaca, placed on the role axis.
  The axis is scaled so that 0 is the main-arm Carrot mean and 1 is the
  Parsnip mean.
- **2026-10-12's ten matched seeds:** per-layer d′ of each seed's direction,
  set next to that seed's ensemble AUROC.

Outputs: `results.json`, `figures/{spectrum,pc_scatter,random_auroc,random_predicted}.svg`.
The figures show the contrast feature.

## Results

### 1. From layer 35 on, PC1 is the role axis and holds about half of all variance

Contrast feature:

| layer | PC1 share of variance | share of \|Δ\|² in PC1 | PC1 role AUROC | role share of all variance | \|Δ\|²/tr Σ_w | participation ratio |
|---|---|---|---|---|---|---|
| 19 | 0.24 | 0.05 | 0.530 | 0.01 | 0.06 | 11.6 |
| 27 | 0.24 | 0.01 | 0.516 | 0.02 | 0.09 | 11.0 |
| 35 | 0.43 | 0.96 | 0.999 | 0.32 | 2.55 | 4.8 |
| 43 | 0.56 | 0.99 | 1.000 | 0.47 | 4.79 | 3.0 |
| 51 | 0.53 | 0.99 | 1.000 | 0.45 | 4.33 | 3.3 |
| 58 | 0.49 | 0.99 | 1.000 | 0.41 | 3.77 | 3.7 |

- **Layers 35–58:** the role mean shift is 32–47% of all variance in the 5,120
  dimensional contrast. Almost all of it (96–99%) lies on the first principal
  component, so cos(Δ̂, PC1) = 0.98–1.00. The cloud is low-dimensional
  (participation ratio 3–5): two classes separated along its longest axis
  (`figures/pc_scatter.svg`).
- **PC1 is not a confound** at those layers: 5p-vs-3p AUROC 0.36–0.41,
  |r| with the round ≤ 0.21.
- **Layers 19 and 27:** PC1 tracks the round instead (|r| = 0.86), and role is
  ≤ 2% of the variance.

### 2. A random direction's AUROC follows from that one axis

- **Why a random direction sees role:** a random unit vector u samples every
  direction equally. So its expected squared projection on Δ is |Δ|²/D, and
  its expected within-class variance is tr Σ_w / D. The typical separation is
  therefore d′ ≈ N(0, |Δ|²/tr Σ_w).
- **Typical size:** at layers 35–58, |Δ|²/tr Σ_w = 2.6–4.8, i.e. the rms
  d′ is 1.6–2.2.
- **Prediction vs measurement:**
  - the rms d′ measured over 2,000 directions (1.56 / 2.08 / 2.06 / 1.92)
    matches that prediction (1.60 / 2.19 / 2.08 / 1.94);
  - per direction, the Gaussian prediction Φ(d′/√2) matches the measured
    AUROC with r = 0.998–0.999 (`figures/random_predicted.svg`);
  - 49–60% of random directions score AUROC > 0.8 or < 0.2.
- **What drives a direction's score:** the share of its variance that comes
  from the role shift (median 20–31%). That share correlates with
  |AUROC − 0.5| at r = 0.95.
- **Why the projection is large:** u·Δ is tiny (Δ's norm spread over 5,120
  dimensions), but PC1's variance is so large that even a ~1/√5120 projection
  onto it beats the variance in all the other directions.
- **The 2026-10-12 seeds:** each seed's ensemble AUROC follows the signs of its
  per-layer d′ at layers 35–58 (results.json `source_random_seeds`):
  - seed 8 (0.056): negative at 43, 51 and 58 (−2.54, −1.30, −1.70);
  - seeds 0 and 5 (0.84, 0.82): dominated by a large positive d′ at layer 51
    (+2.12, +2.67);
  - the ensemble's sigmoid and mean over layers mix the rest.

### 3. Project out PC1 and both random and trained directions drop to chance

Contrast feature, top PC removed (k = 1):

| layer | \|Δ\|² remaining | random AUROC > 0.8 or < 0.2 (as is → k=1) | trained fibs linear weight AUROC (as is → k=1) |
|---|---|---|---|
| 35 | 3.6% | 0.49 → 0.00 | 0.990 → 0.473 |
| 43 | 0.9% | 0.60 → 0.00 | 0.996 → 0.396 |
| 51 | 1.0% | 0.60 → 0.00 | 0.996 → 0.469 |
| 58 | 1.4% | 0.57 → 0.00 | 0.995 → 0.412 |

- **The trained probe:** at each layer, the fibs linear weight separates roles
  only through PC1. Its cosine with PC1 is only 0.12–0.18, but that is about
  10× a random direction's (~0.014). It is a better-aimed read-out of the same
  axis, not a different signal.
- **Refitted diff-of-means after ablation:** it still reaches 0.58–0.73
  in-sample, even at k = 10 where |Δ|² is about 0. That is the expected
  in-sample bias of a mean-difference direction in 5,120 dimensions fitted on
  2,400 rows, not held-out signal; no held-out split was run.

### 4. The axis is specific to the Parsnip

All-Carrot-game announcements and alpaca replies sit near the Carrots (0 =
main Carrot mean, 1 = Parsnip mean; mean of each set):

| layer | main Carrot | Parsnip | 3p_cc | 5p_cc | alpaca |
|---|---|---|---|---|---|
| 35 | 0.00 | 1.00 | 0.03 | −0.01 | −0.45 |
| 43 | 0.00 | 1.00 | 0.03 | 0.00 | −0.19 |
| 51 | 0.00 | 1.00 | 0.06 | −0.02 | 0.19 |
| 58 | 0.00 | 1.00 | 0.06 | −0.02 | 0.23 |

So the axis is not "inside a Carrot-Parsnip game", and not "answering this
follow-up". It is the Parsnip's state. The 2026-10-12 README notes that the
role is stated in the Parsnip's own prompt, so this is the role being stated
or the deceptive stance it induces. This analysis cannot tell those apart (no
role-swap control).

### 5. The contrast concentrates the axis

With each prefill alone, role is a smaller share of the variance (16–32%), the
participation ratio is higher (6–8), and Δ is not always on PC1:
- "Yes." alone: 6% of |Δ|² on PC1 at layer 35, 30% at 58;
- "No." alone: 54% at 58.

Subtracting the two prefills removes the variance they share from the context,
which leaves role as the dominant axis. Random directions are still extreme in
26–47% of cases at layers 35–58 on a single prefill. The trained weight scores
"No." alone in reverse (AUROC 0.002–0.006 at layers 35–58): Parsnips score
lower after "No." and higher after "Yes.", so the contrast adds the two.

## Answer

Being the Parsnip moves the read-token state far along one direction, and from
layer 35 on that direction is the largest axis of variation in the contrast
(32–47% of all variance). In a space with a dominant axis, almost every
direction has a non-trivial projection on it, so a random direction's AUROC is
roughly Φ(N(0, 2–5)/√2): far from 0.5, with a random sign. Seeds 0 and 5 are
draws that came up positive at the heavily weighted layers, and seed 8 is one
that came up negative. "Better than a random direction" is therefore a weak bar
on this task. The relevant comparisons are:
- the trained probe against the role axis itself: the in-sample diff-of-means
  direction gives AUROC 1.000 at layers 35–58 (held-out split not run);
- any probe on a task where role is not stated in context.

## Follow-up: the Carrot-Parsnip role axis on FIBSv1 (`fibs_transfer.py`)

Researcher's question: if PC1 plus one bit of sign gives AUROC 0.99 on
Carrot-Parsnip, why train a probe at all?

Test: directions fitted on Carrot-Parsnip (3p + 5p) are applied unchanged to
the FIBSv1 validation hold-out. The FIBS activations are the stored
Qwen3.5-27B read tokens (`cfg.fibs_acts_dir`, 4,967 rows, 2,482 with a false
follow-up answer). The score is the projection.

The Carrot-Parsnip directions are:
- PC1, with the Parsnip end positive (the sign bit);
- the diff-of-means direction;
for each feature (the contrast, "Yes." alone, "No." alone).

The baselines, on the same rows, are:
- the trained fibs linear weights, per layer and as the ensemble;
- FIBS's own PC1 (sign set by the label) and its in-sample diff-of-means;
- 2,000 random directions.

The rows are also split by `base_deceptive`: whether the original response
was deceptive, i.e. deception sits in the context, which is the role-like
signal. Outputs: `results_fibs.json`, `figures/fibs_transfer.svg`.

AUROC against the FIBS label:

| layer | trained fibs weight | CP PC1, contrast | FIBS own PC1 (+ sign) | FIBS diff-of-means (in-sample) | random, max of 2,000 (folded) |
|---|---|---|---|---|---|
| 19 | 0.941 | 0.455 | 0.595 | 0.777 | 0.696 |
| 27 | 0.951 | 0.651 | 0.587 | 0.780 | 0.681 |
| 35 | 0.986 | 0.746 | 0.544 | 0.888 | 0.758 |
| 43 | 0.986 | 0.695 | 0.571 | 0.877 | 0.728 |
| 51 | 0.984 | 0.676 | 0.587 | 0.826 | 0.705 |
| 58 | 0.978 | 0.658 | 0.546 | 0.819 | 0.706 |

The trained six-layer ensemble scores 0.985 on these rows (the card reports
0.972 on its own validation rows).

- **The role axis transfers only weakly.**
  - At best (layer 35) it scores 0.75, about as good as the best of 2,000
    random directions.
  - Its cosine with FIBS's diff-of-means is 0.17–0.27 at layers 35–58.
  - It does not track deception in the context: AUROC against
    `base_deceptive` is 0.41–0.52.
  - The "No."-alone PC1 comes out flipped on FIBS (0.30–0.41 at layers 35–58):
    the sign set on Carrot-Parsnip does not carry over.
- **On FIBS, the label is not a dominant axis.** At every layer:
  - |Δ|²/tr Σ_w is 0.04–0.09, against 2.6–4.8 on Carrot-Parsnip;
  - the label is 1–2% of all variance;
  - the participation ratio is 32–48;
  - FIBS's own PC1 holds 1–18% of |Δ|² and scores 0.54–0.60;
  - no random direction scores above 0.8 or below 0.2.
- **Unsupervised PCA fails here even when fitted on FIBS itself.** The trained
  weight reaches 0.94–0.99 at every layer, and the in-sample diff-of-means
  reaches 0.78–0.89.

Answer: training is what makes the probe work. Carrot-Parsnip is a special
case where the label happens to be the largest axis of variation, because
the role is stated in context. There, any read-out aimed roughly at that axis
scores about 0.99. On a dataset where deception is one small direction among
many, the unsupervised top component misses it, and the trained probe finds
it.

Caveats:
- The contrast directions are differences between the two prefills, applied
  here to single reads; the FIBS rows have no prefill contrast.
- FIBS answers are free text rendered with an empty think block. The
  Carrot-Parsnip reads are the prefilled "Yes." / "No.".
- The FIBS baselines (own PC1, diff-of-means) are in-sample on the validation
  rows.

## Follow-up: is the axis there when the Parsnip does not lie? (`judge_split.py`)

Setup:
- the Carrot-Parsnip PC1 (contrast feature, Parsnip end positive);
- the Parsnip announcements split by 2026-10-12's oracle judge grade
  (`judge_{arm}.jsonl`: honest 1–2, ambiguous 3–5, lie 6–7);
- position on the axis measured with 0 = Carrot mean, 1 = Parsnip mean;
- output `results_judge.json`.

| layer | Carrot (n=1,800) | Parsnip honest (n=14) | Parsnip ambiguous (n=376) | Parsnip lie (n=210) | honest Parsnip vs Carrot | lie vs honest Parsnip | lie vs ambiguous Parsnip |
|---|---|---|---|---|---|---|---|
| 35 | 0.00 | 0.85 | 0.93 | 1.13 | 0.997 | 0.870 | 0.774 |
| 43 | 0.00 | 0.86 | 0.91 | 1.18 | 1.000 | 0.826 | 0.787 |
| 51 | 0.00 | 0.83 | 0.89 | 1.21 | 1.000 | 0.826 | 0.789 |
| 58 | 0.00 | 0.78 | 0.87 | 1.25 | 1.000 | 0.876 | 0.821 |

- **Most of the shift is there without a lie.** Parsnip announcements the
  judge calls honest are already 78–86% of the way to the Parsnip mean, and
  they separate from Carrots perfectly.
- **Lies add to it.** Within Parsnips, lies sit further out (1.13–1.25) and
  separate from honest and ambiguous Parsnip statements at AUROC 0.77–0.88.
- **Reading:** the axis is mostly role, with a smaller lie component along the
  same direction.
- **Small sample:** only 14 Parsnip announcements are graded clearly honest.
- **Layers 19 and 27 are left out:** the role gap there is too small for the
  0-to-1 scale to mean anything.

## Conclusion

From layer 35 on, the follow-up read-token contrast is dominated by one axis
that encodes "this speaker is the Parsnip":
- it holds 32–47% of all variance, and the space is effectively 3–5
  dimensions wide;
- most of the shift is present even when the Parsnip speaks honestly, and
  lies push further along it.

Because that one axis dominates the variance, almost every direction
projects onto it. Random directions therefore separate the classes strongly,
with a typical |d′| of about 2. The sign is arbitrary, so the AUROC is far
from 0.5 in either direction. This is not luck, and the same picture
predicts each random direction's measured AUROC (r ≈ 0.998).

On FIBS, lying is a small direction among dozens. There, random directions
and unsupervised PC1 reach only 0.55–0.75, while the trained probe reaches
0.98. So Carrot-Parsnip AUROC measures mainly role decoding, not lie
detection.

## Limitations

- **The PCA, Δ and the role axis are fitted on the same 2,400 announcements
  they are evaluated on.** At d′ = 4–6 on PC1 this does not matter. It does
  matter for the post-ablation diff-of-means numbers (see 3).
- **The random-direction measures here are per layer and linear.** The fibs
  ensemble (sigmoid, per-layer scale and bias, mean over six layers) is linked
  only through the per-seed d′ table.
- **One model and one read token** (the fibs probe's), and the experiment's own
  game prompts.

## Run

```sh
E=experiments/2026-10-16-random-direction-pca
uv run --no-sync python $E/pca.py --stage extract    # CPU; needs 2026-10-12's acts_*/ on disk
uv run --no-sync python $E/pca.py --stage analyze    # CPU
uv run --no-sync python $E/judge_split.py           # CPU
uv run --no-sync python $E/fibs_transfer.py         # CPU; needs 2026-10-13-fibs-acts' read-token store
```
