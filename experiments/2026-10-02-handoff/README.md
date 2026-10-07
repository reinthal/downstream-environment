# 2026-10-02 — Detector-to-auditor handoff

(Folder suffixed `-handoff`: `2026-10-02/` is the manual LR audit.)

## Question

The follow-up probe separates Parsnip from Carrot messages (AUROC 0.67), and
picking the player with the highest average score names the Parsnip in 72% of
games, yet the auditor shown its 1-7 grades scores 19%. Is the signal lost in
the handoff?

## Design

- `calibrate.py`: the same raw scores with two calibrations: the original
  alpaca cutpoints, and recalibrated cutpoints at the same quantiles of honest
  in-game messages (the 750 all-Carrot turns of the 2026-09-29 no-Parsnip
  games). Follow-up and LR probes.
- Auditor conditions (`config.json`, 50 games x 5 seeds, base + steered):
  `fu_recal`, `lr_recal` (recalibrated grades), and `fu_avg`, `fu_recal_avg`,
  `lr_avg`, `lr_recal_avg` (grades plus each player's average grade after the
  transcript).
- `rule.py`: the "highest average score" rule for every detector (probes
  averaged as log-odds), next to the auditor's accuracy with the same detector.
- `grade_hist.py`: distribution of the grades the auditor was shown.

## Deliberate limitations

- The recalibration set (no-Parsnip games) is in-game data, but label-free:
  every message in it is honest by construction.
- 50 audited games (`game_idx < 10` per seed group), as earlier audits.
- God-mode judge outputs without a parsable score (8/1500 base, 5/1500
  steered) count as the midpoint 4 (`UNPARSED_JUDGE` in `rule.py`).

## Files

- `config.json`, `run_all.sh`; symlinked 2026-09-29 transcripts / games
- Outputs: `scores_{base,steered}.csv`, `suspicion_scale.json`,
  `grade_rule.json`, `rule.json`, `auditor_seeds_{arm}.jsonl`, `figures/`
