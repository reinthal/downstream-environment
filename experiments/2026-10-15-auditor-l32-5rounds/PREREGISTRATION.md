# Preregistration — The auditor inside the 5-player game: five rounds, layer-32 probe

Filled by the `/preregister` skill from the researcher's answers, quoted
verbatim. Committed before the first non-pilot run; the sections above
"Deviations" are frozen from that commit on. Unanswered items read
`NOT PREREGISTERED`. Lines marked *(Claude)* are facts from the repo or
implementation choices, not the researcher's answers.

- Date: 2026-10-15
- Researcher: Salmaan Barday
- Builds on: `experiments/2026-10-14-auditor-in-game/` (same wrapper and
  scenarios; with the layer-48 probe every scenario scored 0.96-1.00, text
  only all at chance except S4 0.48), `experiments/2026-10-12-qwen38-layer-sweep/`
  (the layer-32 probe).

**The researcher chose the change (more rounds, a weaker probe, S5 dropped,
no new scenario) and gave no hypothesis, prediction or decision rule ("lets
run this"; asked whether to state a prediction, not answered). Every result
of this run is therefore exploratory.**

## 1. Question and stakes

- Decision this result informs: which way of putting the auditor into the
  game to pursue (as 2026-10-14). Researcher, verbatim: "can we run the same
  thing, but with more rounds? and maybe the old probe cause its worse so we
  can maybe differentiate better?"
- Also asked, verbatim: "do the probes run on the questions that the auditor
  asks the participants...Think it could be intersting if this happesn no?
  would this have utility?" *(Claude: the players' answers are scored and
  shown; `probe_top1_ann` vs `probe_top1` measures what they add.)*
- Claim hoped for (one sentence): NOT PREREGISTERED
- Next step if supported / refuted / inconclusive: NOT PREREGISTERED

## 2. Hypothesis

- H1 (with direction): NOT PREREGISTERED. *(Claude, the researcher's earlier
  guess, 2026-10-12: "My guess is that the mdore the model interacts the
  better it will be.")*
- Null: NOT PREREGISTERED
- Rival explanation: NOT PREREGISTERED
- Prediction for the primary metric (value, interval, confidence %): NOT PREREGISTERED
- Result that would make me abandon H1: NOT PREREGISTERED
- Expected baseline / control values: NOT PREREGISTERED. *(Claude: text only
  at 3 rounds, 2026-10-14: 0.12-0.24 except S4 0.48. Layer-32 probe alone,
  3-player games: message AUROC 0.72-0.73, top-1 0.80.)*

## 3. Test design

- Primary metric (one): NOT PREREGISTERED. *(Claude: `analyze.py` reports
  auditor accuracy — names the Parsnip — per scenario x detector first.)*
- Secondary metrics *(Claude)*: accuracy counting failed audits as wrong;
  failures; questions asked; the players' Carrot win rate; probe alone
  (top-1 over all messages and over announcements only; message AUROC over
  all messages, announcements, answers) in probe conditions.
- Unit of analysis *(Claude)*: the game; game bootstrap.
- Arms: S0 passive, S1 final questions, S2 per-round questions, S3 question
  budget (5), S4 private interviews, S6 public recommendation, each x {text
  only, live layer-32 probe scores}. "ye we can drop the s5 role claim";
  "dont worry with scenriao 7". Five discussion rounds ("more rounds";
  *(Claude)* five chosen). Probe *(Claude)*: layer 32 ("the old probe cause
  its worse"; the old Qwen3.5 layer-18 probe cannot read Qwen3.8, and its
  Qwen3.8 counterpart, layers 16-24, scored ~0.53).
- Sample size *(Claude)*: 50 games per set (9 sets, 450 games), seeds
  6000-6049. Not justified by a power analysis (about ±0.13 per accuracy).
- Fixed parameters: `config.json`, `five.yaml`.
- Stopping rule *(Claude)*: fixed N, no extension.

## 4. Analysis plan

- Statistic, interval method *(Claude)*: accuracy with a 95% game bootstrap
  (2000 resamples, seed 0).
- Threshold for "supported": NOT PREREGISTERED
- Exclusion rules *(Claude)*: an audit with no verdict is reported as
  failed — excluded from `accuracy`, counted wrong in `accuracy_all`. Void
  fraction: NOT PREREGISTERED.
- Multiple comparisons: NOT PREREGISTERED
- Figure *(Claude)*: accuracy per scenario, text only and probe side by side,
  y from 0 to 1, chance line 1/5.
- Confirmatory analyses: none. Exploratory: everything.

## 5. De-risking

- Pre-mortem *(Claude)*:
  - Pilot (2 games per set, seeds 0-1, 18 games, never analysed beyond
    these checks): 0 failed audits; all 80 auditor calls finished
    (`stop`, none hit the token cap).
  - The layer-32 probe on 5-player pilot messages: announcements AUROC
    0.67 (n 246), answers to the auditor 0.47 (n 120). Off the ceiling as
    intended; answers may carry no signal at this layer.
  - Pilot dialogues the probe reads: median 4.3k tokens, p90 8.8k, max
    14.4k; 45/366 exceeded the 8192-token window and were left-trimmed
    (losing the system prompt and role knowledge). Fixed before the run:
    `max_len` 16384, `batch_size` 2 (the longest pilot dialogues scored in
    21 s; a short one scores identically).
  - Run time: pilot games took 40-70 min with 18 in flight; the full run is
    expected at ~20 h (2026-10-14: 11.4 h for 500 3-round games).
- Can the metric resolve the effect: NOT PREREGISTERED
- Positive control *(Claude)*: none beyond 2026-10-14 (layer 48, ~1.00).
- Reused primitives / new code *(Claude)*: reused — 2026-10-14 `game.py`,
  `scorer.py`, `analyze.py` (copied; changes listed in README), the
  2026-10-12 layer-32 probe.

## 6. Conclusions agreed in advance

- If supported / refuted / inconclusive, the report says: NOT PREREGISTERED
- Limitations stated regardless of outcome: NOT PREREGISTERED

---

## Deviations

<!-- Append only. Each entry: date, what changed, why, decided before or
     after seeing data. -->
