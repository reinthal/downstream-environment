# 2026-10-03 — Statistical analysis before the handoff

## Question

What do the detectors' scores carry, and does an auditor do better when the
analysis is done for it and handed over as one calibrated summary per game?

## Design

1. `analyze_detectors.py` → `step1.json`: message-level AUROC (game-bootstrap
   CI; percentile among random directions), player-level aggregation (mean,
   max, min, last round, change), confounds (length, reasoning length, round,
   speaking position).
2. `posterior.py` → `step2.json`, `posteriors_*.json`: per-player Parsnip
   probabilities, cross-fitted by seed group (fit on 4, apply to the 5th):
   remove round / position drift, average per player, Gaussian likelihood
   ratio Parsnip vs Carrot, softmax over the 5 players with a fitted
   temperature. Random directions go through the same pipeline.
3. `build_summaries.py` → `summary_*_base.json`: per game, each player's
   probability plus the detector's top-1 accuracy on the other seed groups.
   Probe and random conditions share one generic description.
   Audited in `audit_a/`, `audit_b/` (100 games x 2 seeds each, 10k thinking
   tokens), `analyze_audits.py` → `step3.json`.

`data.py` holds the shared loaders (messages, exact per-message scores).

## Deliberate limitations

- Step 2 uses role labels from other games (cross-fitted), so it is a
  supervised calibration; 2026-10-04 is the label-free version.
- `gold_prob` and `godmode_prob` are sanity checks, not detectors under test.
- God-mode judge outputs without a parsable score count as 4
  (`UNPARSED_JUDGE` in `data.py`).

## Files

- `audit_{a,b}/config.json` (auditor runs; seeds 4242 / 5242)
- Run: `.venv/bin/python experiments/2026-10-03/{analyze_detectors,posterior,build_summaries}.py`;
  `PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python -m auditor.run --config experiments/2026-10-03/audit_a/config.json --arm base --seeds 2`
  (same for `audit_b`); then `analyze_audits.py`
