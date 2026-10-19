# Preregistration — Auditor uplift over its detector

Filled by the `/preregister` skill from the researcher's answers. Committed
before the first non-pilot run; the sections above "Deviations" are frozen
from that commit on. Unanswered items read `NOT PREREGISTERED`. Lines marked
*(Claude)* are facts from the repo or implementation choices, not the
researcher's answers.

- Date: 2026-10-16
- Researcher: Salmaan Barday (with colleagues' input, relayed by the researcher)
- Builds on: `experiments/2026-10-14-auditor-in-game/` (layer 48: every
  scenario 0.96-1.00; the auditor names the probe's top player in 96-100% of
  games), `experiments/2026-10-15-auditor-l32-5rounds/` (layer 32: passive +
  probe 0.82 vs probe alone 0.78; questions lowered accuracy).

**How the answers below were given.** Claude proposed wording for the
hypothesis, prediction, primary metric, abandonment rule and the
inverted-oracle expectation, and the researcher adopted it; the lines marked
(adopted) are that wording. Motivation (researcher and colleagues): test the
auditor with a random probe and with a partly random probe, check that an
oracle detector gives near-perfect results, and measure the auditor's uplift
over its detector — does it agree with good probes and disagree with bad
ones — with the interactive (dynamic) auditor as the main interest.

## 1. Question and stakes

- Decision this result informs: whether the auditor (and which interactive
  setup) adds value beyond the detector, i.e. whether to keep investing in
  the dynamic auditor or in the detector itself.
- Claim hoped for (one sentence): (adopted) "In the interactive scenarios,
  the auditor's accuracy is higher than the probe's own top-scoring-player
  accuracy when the probe is weak (random, mix), and matches it when the
  probe is strong (layer 48, oracle)."
- Next step if supported / refuted / inconclusive: NOT PREREGISTERED

## 2. Hypothesis

- H1 (with direction): (adopted) "In the interactive scenarios, the
  auditor's accuracy is higher than the probe's own top-scoring-player
  accuracy when the probe is weak (random, mix), and matches it when the
  probe is strong (layer 48, oracle)."
- Null *(Claude)*: uplift (auditor − probe alone) = 0 at every detector — the
  auditor relays the detector.
- Rival explanation *(Claude)*: an apparent uplift in S4 comes from the
  Parsnip naming its own role in private (2026-10-15: 23/50 text-only games),
  not from weighing the detector; reported as uplift without confession
  games.
- Prediction for the primary metric (value, interval, confidence %):
  (adopted) "Auditor minus probe-alone ≥ +0.10 at the mix and random
  detectors in S2; about 0 at the oracle. 60% confident."
- Result that would make me abandon H1: (adopted) "Uplift ≤ 0 at every
  detector, i.e. the auditor never beats the probe it is given."
- Inverted oracle, expected: (adopted) "The auditor follows it into the
  wrong player, scoring below text only."
- Expected baseline / control values *(Claude)*: text only, 2026-10-15:
  S2 0.46, S4 0.64, S0 0.16. Probe alone by design: rand ~0.2, mix ~0.5,
  oracle ~1, inv ~0.

## 3. Test design

- Primary metric: (adopted) "uplift = auditor accuracy − probe-alone
  accuracy on the same games, pooled over S2 and S4, at the mix detector."
  *(Claude)* Probe alone = the player with the highest mean shown score over
  their announcements is the Parsnip, in the same game.
- Secondary metrics *(Claude)*: uplift per scenario and detector (S2, S4,
  passive S0 replay); follow rate (auditor names the probe's top player when
  the probe is right); override rate (auditor right when the probe is
  wrong); uplift without confession games; auditor accuracy vs text only;
  Carrot win rate; message AUROC of the shown scores.
- Unit of analysis *(Claude)*: the game (passive: mean of 3 auditor runs per
  game); paired game bootstrap.
- Arms *(Claude, as agreed in conversation; option "B")*: S2 and S4 x
  detectors l32, l48, rand, mix, oracle, inv (50 games each, seeds
  6000-6049); text-only S2 / S4 from 2026-10-15 (same seeds); passive S0
  replay of the 50 2026-10-15 S0 discussions x none, l32, l48, rand, mix30,
  mix, mix70, mix90, oracle, inv x 3 runs. Scores shown in layer-48 raw
  units with "analyse them as you see fit", as chosen by the researcher: no
  Python tool, no 1-7 grades.
- Sample size *(Claude)*: 50 games per interactive cell, 100 pooled for the
  primary metric. Not justified by a power analysis (pooled paired uplift
  resolvable to roughly ±0.10).
- Fixed parameters: `config.json`, `five.yaml`, `calibration.json` (written
  by `calibrate.py` from the 2026-10-15 text-only games before the pilot).
- Stopping rule *(Claude)*: fixed N, no extension.

## 4. Analysis plan

- Statistic, interval method *(Claude)*: mean paired difference, 95% game
  bootstrap (2000 resamples, seed 0).
- Threshold for "supported" *(Claude, derived from the adopted prediction)*:
  pooled mix uplift ≥ +0.10 with a 95% interval excluding 0. Refuted: the
  abandonment rule above (uplift ≤ 0 at every detector, S2 and S4).
  Otherwise inconclusive.
- Exclusion rules *(Claude)*: an audit with no verdict is excluded from that
  cell (reported); a game that errors is reported as failed. Void fraction:
  NOT PREREGISTERED.
- Multiple comparisons *(Claude)*: one primary metric; everything else is
  secondary / exploratory, uncorrected.
- Figure *(Claude)*: auditor accuracy vs probe-alone accuracy per detector
  and scenario, with the diagonal auditor = probe alone and the text-only
  levels; both axes 0-1.
- Confirmatory: the primary metric and the abandonment rule. Exploratory:
  everything else.

## 5. De-risking

- Pre-mortem *(Claude)*:
  - A single late-layer random direction can be informative by chance
    (2026-10-12: 0.05-0.93); the direction is the one of 200 closest to
    AUROC 0.5 / top-1 0.2 on calibration games.
  - The mix level may miss 50% on the evaluation games (calibrated on 100
    other games); probe-alone accuracy is measured in each game, so uplift
    stays well defined.
  - Different detectors could be told apart by scale; all are shown in
    layer-48 units.
  - S4 confessions (see rival explanation).
  - Run time ~40 h on two server replicas.
  - Pilot (passive replay 2 games x 10 detectors x 1 run; 2 games per
    interactive set, seeds 0-1; never analysed beyond these checks): 0
    failed games or audits. It used a first calibration that rescaled over
    all messages together, which left detectors at different announcement
    levels (rand −11 vs l48 −20); changed before this commit to a per-kind
    rescale (announcements and answers each mapped to layer 48's levels),
    same random direction. Calibration report (calibration games):
    probe-alone top-1 rand 0.20, mix30 / mix / mix70 / mix90 0.29 / 0.50 /
    0.70 / 0.90, l32 0.75, l48 / oracle 1.00, inv 0.00; the random direction
    is ~uninformative on announcements (AUROC 0.49), less so on answers
    (0.59).
- Can the metric resolve the effect: NOT PREREGISTERED
- Positive control *(Claude)*: oracle (auditor expected near 1.0).
  Negative / adversarial controls: rand, inv.
- Reused primitives / new code *(Claude)*: reused — 2026-10-15 `game.py`
  (detector hook added), the 2026-10-12 layer-32 / 48 probes,
  `linear_probes` activation code, `auditor.prompts.parse_guess`; new —
  `collect.py`, `calibrate.py`, `detectors.py`, `scorer.py`, `replay.py`,
  `analyze.py`.

## 6. Conclusions agreed in advance

- If supported / refuted / inconclusive, the report says: NOT PREREGISTERED
- Limitations stated regardless of outcome *(Claude)*: same model for
  auditor and players; 50 games per cell; oracle scores are re-drawn per
  message; calibration games share seeds (not transcripts) with the
  evaluation games.

---

## Deviations

<!-- Append only. Each entry: date, what changed, why, decided before or
     after seeing data. -->

- 2026-10-18, decided before seeing any interactive result (only progress
  counts and failure tracebacks were looked at). With 12 sets x 20
  concurrent games on two server replicas, requests waited past the 1800 s
  request timeout: 23 of the first 240 finished interactive games failed
  with `TimeoutError` (S2: l32 4, mix 2, rand 1; S4: l32 6, mix 4, rand 3,
  inv 2, oracle 1; l48 none). Because failures are uneven across
  detectors, once the run finishes every failed game is replayed with the
  same seed (`game.py --set <s> --rerun`: request timeout 7200 s, 10
  concurrent games) and `analyze.py` replaces the failed record with the
  replay. Failed games are otherwise unchanged in definition.

- 2026-10-18 (second entry), decided before seeing any interactive result.
  The game processes were tied to the Claude Code session and were killed
  when it ended (~06:25 UTC); `game.py` wrote a set's results only when the
  whole set finished, so all interactive games played so far (~250) were
  lost unseen. The passive replay (written per audit) is complete and kept.
  The 12 interactive sets are rerun from the start with the same seeds and
  design. Infrastructure changes only: each game is written as it finishes
  and a restart resumes the set; `config.json` request timeout 1800 -> 7200 s
  and concurrent games per set 20 -> 8 (fewer queued requests); jobs run
  detached from the session.

- 2026-10-19, decided before seeing any interactive result (only counts of
  finished seeds per set were looked at). The researcher stopped the
  interactive run early to free GPUs for the next experiment (Inside Job).
  At the stop (16:47 UTC) seeds 6000-6031 (g000-g031) were complete in all
  12 sets; the ~96 games in flight (later seeds) were discarded. Sample size
  per interactive cell 50 -> 32 (pooled primary metric 100 -> 64 games),
  with every analysed set restricted to the same seeds g000-g031 (also the
  2026-10-15 text-only S2 / S4 references), so cells stay paired and the cut
  does not favour shorter games. Final: the run will not be resumed or
  extended. The passive S0 replay (50 games x 3 runs) is complete and
  unchanged. Consequence: lower power; the pooled interval is about ±0.13
  rather than ±0.10, so a true uplift near +0.10 will likely be
  inconclusive under the preregistered threshold.
