# Preregistration — The auditor inside the 5-player game: six interaction scenarios

Filled by the `/preregister` skill from the researcher's answers, quoted
verbatim. Committed before the first non-pilot run; the sections above
"Deviations" are frozen from that commit on. Unanswered items read
`NOT PREREGISTERED`. Lines marked *(Claude)* are facts from the repo or
implementation choices, not the researcher's answers.

- Date: 2026-10-14
- Researcher: Salmaan Barday
- Builds on: `experiments/2026-10-04/` (Opus auditor with raw follow-up
  scores, 5-player games), `experiments/2026-10-09-3p-game/`,
  `experiments/2026-10-12-qwen38-layer-sweep/` and
  `experiments/2026-10-13-l48-vs-selfreport/` (the layer-48 follow-up probe:
  game message AUROC ~1.00, top-1 1.00 in 3-player games, robust to a
  "deny" instruction); the auditor-in-game options discussed 2026-10-12.

**The researcher chose the scenarios, the goal, the game, the
implementation and the auditor model, and gave no hypothesis, prediction or
decision rule ("run it"). Every result of this run is therefore exploratory.**

## 1. Question and stakes

- Decision this result informs: which way of putting the auditor into the
  game to pursue — "lets look at some of these scenrios and then we can make
  some desions as to which direction to proceed in".
- Claim hoped for (one sentence): NOT PREREGISTERED. Goal, verbatim: "the
  goal is tro try and get the auditor to get the best reults possible."
- Next step if supported / refuted / inconclusive: NOT PREREGISTERED

## 2. Hypothesis

- H1 (with direction): NOT PREREGISTERED. *(Claude, the researcher's earlier
  guess, 2026-10-12: "My guess is that the mdore the model interacts the
  better it will be.")*
- Null: NOT PREREGISTERED
- Rival explanation: NOT PREREGISTERED
- Prediction for the primary metric (value, interval, confidence %): NOT PREREGISTERED
- Result that would make me abandon H1: NOT PREREGISTERED
- Expected baseline / control values: NOT PREREGISTERED. *(Claude: the probe
  alone named the Parsnip in 100% of 3-player games (2026-10-12/13), so the
  probe conditions may sit at the ceiling; text-only conditions are where
  interaction can show.)*

## 3. Test design

- Primary metric (one): NOT PREREGISTERED. *(Claude: `analyze.py` reports
  auditor accuracy — names the Parsnip — per scenario x detector first.)*
- Secondary metrics *(Claude)*: accuracy counting failed audits as wrong;
  failures; questions asked; the players' own Carrot win rate; probe alone
  (top-1, message AUROC) in probe conditions.
- Unit of analysis *(Claude)*: the game; game bootstrap.
- Arms: "lets try them all" — S0 passive, S1 final questions, S2 per-round
  questions, S3 question budget (5), S4 private interviews, S5 role claims,
  S6 public recommendation (see README), each with no detector and with
  live layer-48 probe scores. Game: "keep the 5 player setup" (`five.yaml`).
  Implementation: "wrapper in experiemntal folder" (`game.py`). Model:
  "local qwen3.8" (players and auditor).
- Sample size *(Claude)*: 50 games per set (10 sets, 500 games), seeds
  5000-5049; S0/S1 share their discussions (paired), other scenarios are
  paired by role assignment only. Not justified by a power analysis (about
  ±0.13 per accuracy).
- Fixed parameters: `config.json`, `five.yaml`.
- Stopping rule *(Claude)*: fixed N, no extension.

## 4. Analysis plan

- Statistic, interval method *(Claude)*: accuracy with a 95% game bootstrap
  (2000 resamples, seed 0).
- Threshold for "supported": NOT PREREGISTERED
- Exclusion rules *(Claude)*: an audit with no verdict (auditor or player
  could not complete a step) is reported as failed — excluded from
  `accuracy`, counted wrong in `accuracy_all`. Void fraction: NOT PREREGISTERED.
- Multiple comparisons: NOT PREREGISTERED
- Figure *(Claude)*: accuracy per scenario, text-only and probe side by side,
  y from 0 to 1, chance line 1/5.
- Confirmatory analyses: none. Exploratory: everything.

## 5. De-risking

- Pre-mortem *(Claude)*:
  - Pilot (2 games per set, seeds 0-1, never analysed): at Qwen3.8's default
    reasoning effort ("xhigh") the text-only auditor thought past the
    12,000-token cap and gave no answer in nearly every call (probe auditor:
    ~25% of calls). With `reasoning_effort: medium` (chat-template option) the
    same failing prompts finished in ~4,000 tokens; the auditor now runs at
    medium (`config.json`). Failures are still possible and are reported.
  - Probe conditions may be at the ceiling (pilot probe_top1 1.0 in 9/10
    probe cells).
  - Qwen3.8 players sometimes announce as plain text (engine: silence).
  - Run time: ~45-60 min per game sequentially; ~6-8 h for 500 games on
    two server replicas (GPUs 1,2 + 0,3); the scorer on GPUs 6,7.
- Can the metric resolve the effect: NOT PREREGISTERED
- Positive control *(Claude)*: the probe conditions (expected near 1.0).
- Reused primitives / new code *(Claude)*: reused — the engine's game state,
  prompts and vote resolution, `auditor.prompts.parse_guess`, the 2026-10-12
  layer-48 probe, `linear_probes` activation code; new — `game.py`,
  `scorer.py`, `analyze.py`.

## 6. Conclusions agreed in advance

- If supported / refuted / inconclusive, the report says: NOT PREREGISTERED
- Limitations stated regardless of outcome: NOT PREREGISTERED

---

## Deviations

<!-- Append only. Each entry: date, what changed, why, decided before or
     after seeing data. -->

- 2026-10-15, decided after seeing data from the other nine sets. The ten
  sets were launched in parallel (14:22 UTC, 2026-10-14). Each `game.py`
  runs `cfg.save(OUT)` on start, so ten processes rewrote `config.json` at
  once; `S3_probe` read it mid-write (`JSONDecodeError`) and exited before
  playing any game. The other nine sets finished with no failed audits
  (01:45 UTC). `S3_probe` was then rerun alone, unchanged (same seeds
  5000-5049, server, scorer and config values; the server had no other
  load), 01:46-03:12 UTC, before its result was known. `config.json` now
  holds the full `RunConfig` dump written by `cfg.save` (same values, all
  defaults spelled out).
