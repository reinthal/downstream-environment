# 2026-10-07 — DYL probe vs alpaca control (DYLProbe primitive)

## Question

What do the DYL ("did you lie") probe's score distributions and ROC look
like with the benign alpaca control alongside the game data — and does the
new `linear_probes.DYLProbe` primitive reproduce the 2026-10-01 pipeline
end-to-end (alpaca rescored from scratch)?

## Design

- Rescore 9,994 alpaca replies (2026-10-01 `rollouts_alpaca.jsonl`,
  thinking-off, on-policy) and the 1,500 base-game turns (2026-09-29
  transcripts) with `DYLProbe` + the `probes/` layer-18 followup LR probe.
- `figures/violins_alpaca.png` — score distributions (log-odds spacing)
  for alpaca / Carrot / Parsnip with the alpaca 1%-FPR threshold and the
  fraction of each class above it.
- `figures/roc_alpaca.png` — Parsnip-vs-Carrot and Parsnip-vs-alpaca ROC
  curves, with the operating point at the alpaca 1%-FPR threshold.

## Deliberate limitations

- Base arm only (no steered/nothink); silent turns (empty announcement)
  are scored but excluded from figures.
- The alpaca rollouts file was restored from another checkout of this repo
  (same 9,994-record count as the committed `scores_alpaca.csv`); scores
  are expected to match 2026-10-01 only up to bf16 batch-composition
  noise, not bit-exactly.
- No new auditor run: this experiment is about the detector's
  distributions, not the handoff.

## Files

- `config.json` (`DYLAlpacaConfig` in `run.py`); symlinked inputs:
  `rollouts_alpaca.jsonl` (2026-10-01), `transcripts_base.jsonl`
  (2026-09-29 → 2026-09-22).
- `run.py --stage score` (GPU, `.venv`) → `scores_{alpaca,base}.csv`;
  `run.py --stage figures` (CPU) → `figures/*.png`.
