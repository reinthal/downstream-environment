# Preregistration — Follow-up probe layer sweep on Qwen3.8-27B

Filled by the `/preregister` skill from the researcher's answers, quoted
verbatim. Committed before the first non-pilot run; the sections above
"Deviations" are frozen from that commit on. Unanswered items read
`NOT PREREGISTERED`. Lines marked *(Claude)* are facts from the repo or
implementation choices, not the researcher's answers.

- Date: 2026-10-12
- Researcher: Salmaan Barday
- Builds on: `experiments/2026-10-01/` (follow-up probe, layer 18, Qwen3.5-27B),
  `experiments/2026-10-09-3p-game/` (3-player games on the engine),
  `experiments/2026-09-29/` (in-game conceal-vs-admit pairs). The researcher
  recalls a lost layer sweep (`experiments/2026-10-06/`, referenced in the
  2026-10-01 README but never committed) in which a late layer gave a
  near-perfect probe.

**The researcher asked to run without answering the hypothesis, prediction,
primary-metric and conclusion questions. Every result of this run is
therefore exploratory.**

## 1. Question and stakes

- Decision this result informs: NOT PREREGISTERED. *(Claude, from the
  conversation: whether a late layer gives a near-perfect follow-up probe on
  the game, and whether that reflects lying or reading the role.)*
- Claim hoped for (one sentence): NOT PREREGISTERED
- Next step if supported / refuted / inconclusive: NOT PREREGISTERED

## 2. Hypothesis

- H1 (with direction): NOT PREREGISTERED
- Null: NOT PREREGISTERED
- Rival explanation: NOT PREREGISTERED. *(Claude, raised before the run and
  not adopted as a test: a late-layer probe can separate Parsnip from Carrot
  by reading the role line in the prompt, or reproduce the model's own
  Yes/No self-report, rather than detect lying.)*
- Prediction for the primary metric (value, interval, confidence %): NOT PREREGISTERED
- Result that would make me abandon H1: NOT PREREGISTERED
- Expected baseline / control values: NOT PREREGISTERED

## 3. Test design

- Primary metric (one): NOT PREREGISTERED
- Secondary metrics: NOT PREREGISTERED. *(Claude: `analyze.py` reports, per
  layer: roleplaying validation AUROC; game message AUROC Parsnip vs Carrot;
  conceal-vs-admit AUROC; the random-direction null.)*
- Unit of analysis, independence, clustering: *(Claude)* messages, clustered
  in games; intervals are game-bootstrap.
- Arms and controls (and what each control rules out): researcher's choices —
  - model: "the newest qwen model under 36B param" *(Claude: Qwen/Qwen3.8-27B,
    revision 1d4bf0f, 64 decoder blocks; Qwen3.8-Flash-Next is 180B)*;
  - probe: "the follow up probe" (Goldowsky-Dill et al. 2025; "Was your last
    response deceptive? (Yes/No)" + "No", read on the "No");
  - layers: "every 8 layers" *(Claude: 0, 8, 16, 24, 32, 40, 48, 56, plus the
    last block, 63, as raw block output with the final norm removed)*;
  - training data: "On-policy Qwen3.8" (fresh replies to the 371 Apollo
    roleplaying scenarios, graded as in 2026-10-01);
  - evaluation: "New Qwen3.8 3-player games";
  - controls: "Conceal vs admit (in-game)" — rules out a role read-out, since
    both variants are the Parsnip — and "Random directions per layer" — the
    null at each layer. Not chosen: all-Carrot games, black-box Yes logprob.
- Sample size and justification: *(Claude)* 100 games (seeds 1000-1099, as
  2026-10-09), 371 roleplaying scenarios (as 2026-10-01). Not justified by a
  power analysis.
- Fixed parameters: see `config.json` and `game.yaml`.
- Stopping rule: *(Claude)* fixed N, no extension.

## 4. Analysis plan

- Statistic, interval method, comparison: *(Claude)* AUROC per layer; 95%
  game-bootstrap intervals (2000 resamples, seed 0) for game sets; random
  directions: 100 per layer in the standardized space, reported as mean and
  95th percentile.
- Threshold for "supported": NOT PREREGISTERED
- Exclusion rules, and the exclusion fraction that voids the run: *(Claude,
  carried over from 2026-10-09 and 2026-09-29)* games that fail, have more
  than one announcement in a turn, or a silent player in round 3 are
  dropped; ambiguous roleplaying grades (3-5) are not trained on;
  conceal/admit pairs are kept only where the honest variant admits and the
  deceptive one does not. Void fraction: NOT PREREGISTERED.
- Multiple comparisons: NOT PREREGISTERED
- Figure (axes, shared ranges): *(Claude)* AUROC against layer, one line per
  evaluation set plus the random-direction band; y from 0.3 to 1.0.
- Confirmatory analyses: none.
- Exploratory analyses: everything.

## 5. De-risking

- Pre-mortem *(Claude)*:
  - The engine turns thinking off only for OpenRouter models; the local
    server is started with thinking off by default so the games match
    2026-10-09.
  - Late blocks: the truncated decoder replaces the final norm with the
    identity, so layer 63 is the raw residual stream, not the normed
    hidden state; if the remembered result used `hidden_states[-1]`, it read
    the normed one.
  - Memory: all 64 blocks are loaded for collection (about 55 GB on two
    A40s); batch size 4.
  - GPUs: Alexander runs servers on GPUs 1, 2, 6, 7. This run uses 0, 3
    (server) and 4, 5 (activations).
  - The probe reads dialogues that include the player's explicit reasoning
    turn, as in 2026-10-09, where the layer-18 Qwen3.5 probe was near chance
    (message AUROC 0.54).
  - Pilot findings (2 games, 4 roleplaying replies; pilot data never
    analysed): 2 of 18 discussion turns wrote the announcement as plain text
    instead of calling MakeAnnouncementTool, which the engine treats as
    silence, and 1 of 2 games was dropped by the round-3 silence rule — the
    rule is kept, so a sizeable share of games may be excluded. Qwen3.8's
    chat template prepends a "Reasoning effort is set to xhigh" system
    preamble unless thinking is off; all dialogues are generated and probed
    with thinking off (`encode_no_span(..., enable_thinking=False)`). The
    probe renders game turns without the tool schema the server put in the
    prompt, as 2026-10-09 did.
- Can the metric resolve the effect: NOT PREREGISTERED
- Positive control: *(Claude)* roleplaying validation AUROC per layer.
- Pilot: *(Claude)* 4 roleplaying replies and 2 games (seeds 0-1), never
  analysed.
- Cost and wall-clock: *(Claude)* local GPUs only, a few hours.
- Reused primitives / new code: *(Claude)* reused — the engine orchestrator,
  `serving/serve_qwen38.py`, `linear_probes.activations`,
  `linear_probes.dyl_probe`, the 2026-10-01 grading template and LR recipe,
  the 2026-10-09 log extraction; new — the server clients, multi-layer
  collection and the sweep analysis.

## 6. Conclusions agreed in advance

- If supported, the report says: NOT PREREGISTERED
- If refuted, the report says: NOT PREREGISTERED
- If inconclusive, the report says: NOT PREREGISTERED
- Limitations stated regardless of outcome: NOT PREREGISTERED

---

## Deviations

<!-- Append only. Each entry: date, what changed, why, decided before or
     after seeing data. -->
