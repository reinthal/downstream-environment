# 2026-10-14 — fibs linear ensemble for Qwen3.5-27B without social-deduction data

## Why

The published fibs probes (and the ones trained in 2026-10-13-fibs-basharena)
see FIBSv1's social-deduction families in training. Our downstream task is a
social-deduction game (Carrot-Parsnip), so a probe that never saw that genre
gives a cleaner read of transfer to it.

## Design

- **Code**: the root package `fibs_probes` (copied from
  2026-10-13-fibs-basharena's `train.py` / `modal_app.py::prepare_fibs`, see
  `fibs_probes/README.md` for the differences); `run.py` here only selects
  rows and records results. All parameters: `config.json`.
- **Data**: the stored Qwen3.5-27B read-token activations of
  2026-10-13-fibs-acts (161,864 FIBSv1 rows ≤ 32,768 tokens; layers 19, 27,
  35, 43, 51, 58).
- **Excluded**: `among_us` (6,464 rows) and `werewolf` (6,692): the two
  social-deduction families of FIBSv1's 49 non-persona families. Out of both
  train and validation; they are scored afterwards as a held-out set.
  `negotiation` stays in (deception, but not a hidden-role game).
- **Split**: 5,000-row class-balanced validation hold-out sharing no
  `pair_id` with training (seed 0), drawn over the 148,708 remaining rows;
  training rows class-balanced within family → 136,030 rows (7,678 dropped).
  `split.parquet` records every row's fate.
- **Recipe**: the paper's linear ensemble (AdamW 5e-3 constant, batch 32,
  20 warm-up steps, clip 1.0, up to 6 epochs with early stop, unit-norm input
  scale from 512 rows, layer rule at train loss < 0.6). One GPU (A40), ~2 min.

## Results

Scored through `probe_inference.load_probe` on the saved directory (the
round trip matches the in-memory model).

| AUROC | this probe (no social deduction) | published linear (trained on all families) |
|---|---|---|
| fibs validation (5,000 held-out rows) | 0.993 | 0.987 (all stored rows, mostly in-sample) |
| among_us (6,464, held out) | 0.910 | 0.966 (in-sample) |
| werewolf (6,692, held out) | 0.871 | 0.948 (in-sample) |

Checkpoint: the last validation of epoch 6 (val loss still falling slowly);
all six layers pass the layer rule. Published column: 2026-10-13-fibs-acts
`check.json`.

## Deliberate limitations

- Linear only (MLP is configured but not run); single seed.
- The published comparison is in-sample on these families, so the gap
  (~0.06-0.08) is an upper bound on what excluding them costs.
- `split.parquet`'s hold-out differs from the 2026-10-13-fibs-basharena
  hold-out (different row set and model), so validation numbers are not
  directly comparable across the two.
- Not preregistered: probe training (a tool), no hypothesis tested.
