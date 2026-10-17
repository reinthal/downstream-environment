# Preregistration — Twenty Questions clue ablation across model capability

Filled by the `/preregister` skill from the researcher's answers, quoted
verbatim. Committed before the first non-pilot run; the sections above
"Deviations" are frozen from that commit on. Unanswered items read
`NOT PREREGISTERED`.

- Date: 2026-10-17
- Researcher: accounts@reinthal.me
- Builds on: `environments/twenty_questions/` (branch `env/twenty-questions`);
  the GLM-5.3 Flash and Qwen3.5-122B-A10B pilots of 2026-10-16/17 (8-turn
  cap, low reasoning; with clues, Qwen3.5-122B-A10B solved 6/6 at a mean end
  turn of 4.3, GLM-5.3 Flash 3/6 at 5.8; no game had been played without clues)

## 1. Question and stakes

- Decision this result informs: "what size of model we need to run this env."
- Claim hoped for (one sentence): NOT PREREGISTERED (stated as the question:
  "at what capability does 20 questions become difficult for models").
- Next step if supported / refuted / inconclusive: NOT PREREGISTERED.

## 2. Hypothesis

- H1 (with direction): "weak models give too easy close for this env to be
  interesting for measuring deception." Operationalised with the researcher:
  a model's clues are "too easy" if its mean end turn **with clues** is
  "avg turn < 3.9"; weak models are too easy, strong ones are not —
  "anything less than glm flag or equivalent, maybe Qwen 400b strong enough"
  (glm flag = GLM-5.3 Flash; Qwen 400b = Qwen3.5-397B-A17B).
- Null: the clue-arm mean end turn does not differ between weak and strong
  models (difference < 1 turn, see abandonment).
- Rival explanations (researcher): "maybe auditor could be weaker as well"
  (the same model guesses, so a weak guesser changes game length
  independently of clue quality); alternative hypothesis: "data too narrow.
  not spread out across model latent space" (the secrets / concept set do not
  separate the models).
- Prediction for the primary metric: no-clue arm: "mean end turn [...] should
  be 7.5 medium confidence" (the researcher placed 7.5 on the no-clue arm;
  medium confidence = 75%). Interval: NOT PREREGISTERED (asked twice).
  Clue arm: weak models < 3.9 (the H1 threshold); no point prediction.
- Result that would make me abandon H1: "there is less then a turns
  difference" — between the weak and the strong models' clue-arm mean end
  turns.
- Expected baseline / control values: the no-clue arm is the baseline,
  predicted at 7.5 (above).

## 3. Test design

- Primary metric (one): mean end turn per model × arm ("nr rounds w /w/o
  clues"), unsolved games counted as the turn cap (12).
- Secondary metrics: median end turn, solve rate within 12 turns, number of
  clues and clue requests per game.
- Unit of analysis, independence, clustering: one game. The same 10 secrets
  (seeds 0–9, `bounded_random` draws from the 512-concept set) are used for
  every model and arm, so cells are paired by secret.
- Arms and controls: per model, `clues` (clues on answers + clue requests,
  the current default) vs `no_clues` (`clues: false`, `clue_requests:
  false`: no clue parameter, no clue lines in the rules). The no-clue arm is
  the control: game length without the clue channel.
- Models (OpenRouter; the same model plays Player 1, Player 2 and the judge):
  qwen3.5-9b, qwen3.5-27b, qwen3.5-35b-a3b, qwen3.5-122b-a10b,
  qwen3.5-397b-a17b, glm-5.3-flash, glm-5.3. ("1b" dropped: no ~1B Qwen with
  tool calls on OpenRouter; "Qwen 3.8 400b MoE" is not on OpenRouter →
  Qwen3.5-397B-A17B; "3.8 35b Moe" is not on OpenRouter → Qwen3.5-35B-A3B,
  researcher's choice.)
- Sample size and justification: "10 rollouts each" — 10 games per model ×
  arm, 140 games. No power analysis (researcher's choice); with a per-game
  SD of ~3 turns a 95% interval on a cell mean is about ±2 turns.
- Fixed parameters: `config.json` — 12 turns, the 512-concept THINGS subset
  (`concepts/things_n512_s0.json`), competitive goals, clue policy `either`,
  final answer ends the game, provider-default reasoning, max_tokens 16,000
  per call, seeds 0–9, prompts = `environments/twenty_questions/configs/default.yaml`
  at the commit of this file.
- Stopping rule: fixed N (10 per cell); no new cell is started once spend
  reaches 80% of the $50 cap.

## 4. Analysis plan

- Statistic, interval method, comparison: "study results qualitativly" —
  qualitative only. `analyze.py` reports, per model × arm, the mean and
  median end turn, solve rate and the clue-arm mean against 3.9 as a point
  estimate, and writes every game's exchanges (`clues_<cell>.md`) for
  reading. No confirmatory statistic: NOT PREREGISTERED.
- Threshold for "supported": clue-arm mean end turn < 3.9 for the weak
  models and not for the strong ones (applied by reading, not by a test).
- Exclusion rules: a game that errors (a player fails its tool call 3 times,
  or a provider error) is re-run once on the same seed; still failing →
  excluded and reported. A model × arm cell with > 20% of its games excluded
  is void.
- Multiple comparisons: none (no tests).
- Figure: `figures/end_turn.svg` — x: models in the fixed order above; y:
  end turn, 0–12.5, one axis for both arms; per game dots and per-cell means;
  dashed line at 3.9.
- Confirmatory analyses: none (qualitative).
- Exploratory analyses: everything, including reading clue texts per model.

## 5. De-risking

- Pre-mortem — boring failure modes and the check for each:
  - tool-call failures on untested models (9B, 27B, 35B, 397B, GLM-5.3;
    only GLM-5.3 Flash and Qwen3.5-122B were piloted) → pilot below;
  - default reasoning running past max_tokens without a tool call →
    max_tokens 16,000, pilot checks stop reasons;
  - the 12-turn cap: no-clue games that need more than 12 turns all read 12,
    compressing the no-clue arm (set-size note: 9 bits at b = 0.6–0.8 →
    12–15 turns) — accepted by the researcher;
  - OpenRouter provider routing / quantisation differences between calls;
  - the judge is each model itself (lenient rulings shorten games);
  - spend → checked after every cell, cap $50.
- Can the metric resolve the predicted effect: the threshold (3.9) is far
  from the predicted no-clue 7.5 and from the cap; a 1-turn abandonment
  difference is within the ±2-turn noise of a 10-game cell — the
  qualitative reading has to carry it.
- Positive control: NOT PREREGISTERED.
- Pilot: 1 game per model per arm (14 games), seed 1000, never analysed; it
  must show every model completing games in both arms (tool calls, judge,
  no-clue rules) and give the real cost per game.
- Cost and wall-clock estimate, hard cap, abort criterion: ~$10–40 for 140
  games at default reasoning; hard cap $50 (the key's daily limit); no new
  cell after $40 spent; models run in parallel, ~1–2 h.
- Reused primitives / new untested code: the twenty_questions engine,
  Inspect runner and judge (tested with scripted players, piloted on two
  models); new: `clues: false` (unit-tested, not yet played by a model),
  this folder's `run.py` / `analyze.py`.

## 6. Conclusions agreed in advance

- If supported, the report says: NOT PREREGISTERED.
- If refuted, the report says: NOT PREREGISTERED.
- If inconclusive (defined as NOT PREREGISTERED), the report says: NOT
  PREREGISTERED.
- Limitations stated regardless of outcome: NOT PREREGISTERED.

---

## Deviations

<!-- Append only. Each entry: date, what changed, why, decided before or
     after seeing data. -->
