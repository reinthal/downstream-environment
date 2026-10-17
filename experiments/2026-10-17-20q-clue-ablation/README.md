# 2026-10-17 — Twenty Questions clue ablation across model capability

## Question

At what capability does Twenty Questions become difficult, and do weak
models give clues so easy that the game is uninteresting for measuring
deception? Hypothesis, prediction and every threshold:
`PREREGISTRATION.md`.

## Design

- Environment: `environments/twenty_questions/` with its `configs/default.yaml`
  prompts (two players, competitive goals, no deception wording), 12 turns,
  secrets drawn uniformly from the 512-concept THINGS subset; seeds 0–9 give
  the same 10 secrets to every model and arm.
- Arms: `clues` (clues on answers + Player 2 may ask for one) vs `no_clues`
  (`clues: false`: no clue parameter, no clue lines in the rules).
- Models (OpenRouter, provider-default reasoning; the same model plays both
  seats and judges): Qwen3.5 9B / 27B / 122B-A10B / 397B-A17B (35B-A3B
  dropped after the pilot, see Deviations),
  GLM-5.3 Flash, GLM-5.3.
- All parameters: `config.json` (`common.AblationConfig`; not a
  `linear_probes` ExperimentConfig — no probes here, and that package imports
  torch, which the env does not carry).

## Run

```bash
uv run --project environments/twenty_questions python experiments/2026-10-17-20q-clue-ablation/run.py --pilot
uv run --project environments/twenty_questions python experiments/2026-10-17-20q-clue-ablation/run.py
uv run --project environments/twenty_questions python experiments/2026-10-17-20q-clue-ablation/analyze.py
```

Per cell (`<model>__<arm>`): `game_<cell>.yaml`, `games_<cell>.json`,
`transcripts_<cell>.jsonl`, `inspect_logs_<cell>/`; analysis: `results.json`,
`figures/end_turn.svg`, `clues_<cell>.md`.

## Deliberate limitations

- 10 games per cell; qualitative analysis only (preregistered).
- 12-turn cap with unsolved = 12 compresses the no-clue arm.
- The same model plays both seats and judges, so clue quality and guessing
  skill move together.
- OpenRouter may route a model to different providers / quantisations.

## Results

120 games (6 models × 2 arms × 10), $3.10; no cell void (one Qwen3.5-27B
no-clue game errored and passed on its preregistered re-run). Secrets
(seeds 0–9, every cell): slide, deer, brooch, lemon, lava, melon, cd player,
puppy, knife, umbrella.

| model | clues: mean end turn | median | solved | no clues: mean | median | solved |
|---|---|---|---|---|---|---|
| qwen3.5-9b | 6.1 | 5.5 | 9/10 | 11.8 | 12 | 2/10 |
| qwen3.5-27b | **3.6** | 3 | 10/10 | 11.8 | 12 | 1/10 |
| qwen3.5-122b-a10b | 5.0 | 4.5 | 9/10 | 11.1 | 12 | 3/10 |
| qwen3.5-397b-a17b | 5.0 | 3.5 | 9/10 | 11.7 | 12 | 3/10 |
| glm-5.3-flash | 7.4 | 7.5 | 8/10 | 11.6 | 12 | 1/10 |
| glm-5.3 | 5.5 | 4.5 | 9/10 | 11.3 | 12 | 3/10 |

Unsolved games count as 12. Figure: `figures/end_turn.svg`; every game's
exchanges: `clues_<cell>.md`.

**Against the preregistration** (analysis qualitative; conclusion sentences
NOT PREREGISTERED, so none is applied):

- "Too easy" (clue-arm mean < 3.9): only Qwen3.5-27B (3.6). Of the models
  "less than glm flag", Qwen3.5-9B (6.1) and Qwen3.5-122B (5.0) are not.
- Abandonment rule ("less then a turns difference", weak vs strong clue-arm
  means): weak = Qwen 9B / 27B / 122B, mean 4.9; strong = GLM-5.3 Flash /
  GLM-5.3, mean 6.45 → 1.55 turns. With Qwen3.5-397B counted strong ("maybe
  Qwen 400b strong enough"): 4.9 vs 5.97 → 1.07 turns; counted weak: 4.93 vs
  6.45 → 1.52. Above one turn under every grouping, so the rule does not
  trigger; the difference is within the ±2-turn noise of 10-game cells.
- No-clue prediction 7.5 (75%): observed 11.1–11.8 for every model; 70–90%
  of no-clue games are unsolved at the 12-turn cap — the ceiling the
  pre-mortem flagged.

**Exploratory (labelled as such):**

- Clues shorten every model's games by 4.2–8.2 turns (pooled 5.4 with clues
  vs 11.6 without).
- Clues are volunteered from turn 1 in almost every game (5.1 per game for
  Qwen3.5-9B, 2.4–3.4 for the others); Player 2 asked for a clue once in 60
  clue-arm games (GLM-5.3 Flash).
- GLM-5.3 Flash is the slowest clue arm (7.4): its first clue came later
  (turn 4 and 9 in two games, never in one) and vaguer ("It moves around on
  its own legs." / "It has exactly four legs." for deer, vs Qwen3.5-27B's
  "It has antlers that it sheds every year.").
- Qwen3.5-9B gives the most clues yet has a slower clue arm than 27B–397B:
  consistent with the researcher's rival "maybe auditor could be weaker as
  well" (its own guessing is weaker).
- Without clues, "knife" is solved by 5 of 6 models, and 5 of the other 9
  secrets (slide, brooch, melon, cd player, umbrella) by none — secret
  difficulty, not only model size, drives the few solves.

### Exploratory: posterior entropy over turns (Jev)

Asked for after the run; not preregistered. `entropy_analysis.py` scores every
question (886 turns, 531 unique) against all 1,796 THINGS concepts with Jev
(`jev-1.13.0`, 93.6M input tokens, ~$3.90) and tracks the posterior over THINGS
after the suspect's actual yes/no answers (clues ignored). Start: 10.81 bits.
Outputs: `entropy_summary.json`, `entropy_<cell>.json`, `jev_scores.jsonl`,
`figures/entropy_by_turn.svg`. Per-step means of all games suffer drop-out
(easy games leave early), so the tables below use fixed cohorts.

No clues, games with ≥ 9 questions (7–10 per model): soft H after 9 questions
and the median number of concepts consistent with every answer (hard count).

| model | soft H after 3 / 6 / 9 questions (bits) | consistent concepts after 3 / 6 / 9 |
|---|---|---|
| qwen3.5-9b | 9.42 / 8.65 / 8.26 | 313 / 41 / 14 |
| qwen3.5-27b | 9.89 / 8.93 / 8.45 | 1324 / 333 / 71 |
| qwen3.5-122b-a10b | 9.75 / 9.43 / 8.56 | 1006 / 692 / 78 |
| qwen3.5-397b-a17b | 9.72 / 8.55 / 6.82 | 1349 / 105 / 21 |
| glm-5.3-flash | 9.76 / 8.37 / 6.75 | 740 / 122 / 13 |
| glm-5.3 | 9.35 / 8.44 / 7.56 | 481 / 80 / 73 |

- Soft posterior: 0.25–0.48 bits per question without clues (pooled per
  question index 0.23–0.52, flat across turns). Jev never returns 0 for a wrong
  concept, so a soft "yes" removes few bits; this understates information.
- Hard count: 1,796 → 13–78 concepts in 9 questions, i.e. 4.5–7.1 bits, about
  0.5–0.8 bits per question — the range the concept-set-size note assumed.
  The 512-concept games needed ~9 bits, consistent with 70–90% of no-clue
  games ending unsolved at 12 turns.
- With clues (games with ≥ 3 questions): the Qwen models' guessers reach
  2–28 consistent concepts by question 3 (their questions follow the clues,
  e.g. "Is it a deer?"); GLM-5.3 Flash 271 and GLM-5.3 128 — the slower clue
  arms of the main table. Pooled clue-arm questions carry 0.82–1.04 soft bits
  each from question 2 on, against ~0.4 without clues.
- Jev's answer for the true secret disagrees with the suspect's on 49 of 886
  question turns (6%); over a long game that often removes the true secret
  from the hard count (secret still consistent at the end in 3–9 of 10 games
  per cell), so hard counts are a lower bound on remaining uncertainty only
  where Jev and the suspect agree.

## Conclusion

### Predicted vs outcome

| preregistered item | predicted (researcher) | outcome |
|---|---|---|
| "Too easy" (clue-arm mean end turn < 3.9) for models "less than glm flag" | Qwen3.5-9B, 27B, 122B too easy ("maybe Qwen 400b strong enough") | only Qwen3.5-27B (3.6); 9B 6.1, 122B 5.0, 397B 5.0 |
| Strong models not too easy | GLM-5.3 Flash, GLM-5.3 (≥ 3.9) | 7.4, 5.5 — not too easy |
| No-clue mean end turn | 7.5, 75% confidence (interval NOT PREREGISTERED) | 11.1–11.8 for every model; 70–90% unsolved at the 12-turn cap |
| Abandon H1 if weak − strong clue-arm difference < 1 turn | — | 1.07–1.55 turns depending on where Qwen3.5-397B is counted: not triggered (within the ±2-turn noise of 10-game cells) |
| Conclusion sentences | NOT PREREGISTERED | — |

### Actual outcome (researcher)

"size of model seems invariant when paired equally given that the length of games w and wo clues are roughly equally long [...] moderate confidence." Revised 2026-10-17 at the researcher's instruction ("the entropy lowering conclusion was wrong. they differ between clue/no clue"): the clause "and entropy for q set is resolved with equal pace for both clue games and no clue games" is withdrawn. Entropy falls faster in clue games: pooled over models, a clue-game question removes 0.82–1.04 soft bits (question positions 2–5) against 0.34–0.52 without clues, and the Qwen guessers reach 2–28 consistent concepts by question 3 with clues against 313–1,300 without.

### Conclusion (researcher; post-hoc — no conclusion sentence was preregistered)

"conclusions: entropy lowering experiment predicted that 512 size secret pool would be barely solvable with 12 questions. the avg solve was just below 12 although confounded by the max 12 questions. needs more experiments and to run for longer. it will be interesting to study how entropy lowering lines up with solving the task low entropy should imply high likelihood of solving. this could be a good metric for studying sanity in the game. still not sure what the difference between hard count and soft entropy is - guess soft entropy captures uncertainty for vagueness of clues maybe? The entropy analysis also is missing a table for clue games - maybe because the number of questions was substantially lower? clues should be included in the entropy analysis as they contribute to lowering it. The question also becomes if it is game theoretically advantageous to use clues if they consume a turn and lowers entropy more than a y/n question. I expect that they do as positive traits lowers more than negative."

### Measurements bearing on the points above (Claude, factual)

- **Pace of entropy reduction, clue vs no-clue games.** Pooled over models,
  per question position, a clue-game question removes 0.82–1.04 soft bits
  (positions 2–5) against 0.34–0.52 without clues. Clue-game questions
  follow the clues ("Is it a deer?").
- **Clue-game entropy table** (requested). Cohort: games with ≥ 3 questions
  (clue games are short: 35–65 questions per cell vs 93–108 without clues, so
  a longer cohort would be nearly empty). Soft H and median consistent
  concepts after 1 / 2 / 3 questions, both arms on the same cohort rule:

| model | clues: n | soft H | consistent | no clues: n | soft H | consistent |
|---|---|---|---|---|---|---|
| qwen3.5-9b | 9 | 10.40 / 9.38 / 8.58 | 1507 / 91 / 10 | 10 | 10.37 / 9.90 / 9.42 | 1512 / 1328 / 313 |
| qwen3.5-27b | 10 | 10.39 / 9.45 / 8.18 | 1518 / 91 / 2 | 9 | 10.36 / 9.99 / 9.54 | 1518 / 1485 / 1300 |
| qwen3.5-122b-a10b | 10 | 10.39 / 9.54 / 8.33 | 1518 / 133 / 28 | 10 | 10.39 / 9.75 / 9.21 | 1518 / 1314 / 581 |
| qwen3.5-397b-a17b | 9 | 10.36 / 9.82 / 8.73 | 1518 / 1313 / 28 | 10 | 10.39 / 10.05 / 9.42 | 1518 / 1485 / 742 |
| glm-5.3-flash | 10 | 10.70 / 9.97 / 9.14 | 1753 / 572 / 271 | 10 | 10.52 / 10.25 / 9.82 | 1480 / 1217 / 742 |
| glm-5.3 | 10 | 10.63 / 9.75 / 8.74 | 1758 / 437 / 128 | 10 | 10.41 / 10.00 / 9.43 | 1479 / 1327 / 594 |

- **Soft entropy vs hard count.** Both use only the yes/no answers; clues are
  not in either. Soft H weights every concept by Jev's p(answer | concept) and
  takes the entropy of that posterior: a concept Jev gives p = 0.05 for the
  observed answer keeps 5% of its weight, so Jev's uncertainty about each
  (question, concept) pair keeps soft H high. The hard count drops a concept
  as soon as one answer disagrees with Jev's p > 0.5 call and counts what is
  left: it reads Jev as certain, so it falls faster — and it also drops the
  true secret when Jev and the suspect disagree (6% of turns).
- **Clues in the entropy analysis.** Not included (the calculator reads only
  the yes/no channel), so clue-game entropy here understates what the
  guesser knew; including clues needs a per-(clue, concept) Jev judgment
  ("is this clue correct for c?"), the same scoring applied to clue texts.
- **Clue requests vs questions.** Player 2 asked for a clue once in 60
  clue-arm games, so this run has almost no data on whether spending a turn
  on a clue beats a question.

### Follow-up: clues scored against the concepts — clue vs question (exploratory)

Requested by the researcher after the conclusion. `entropy_analysis.py --clues`
scores every clue of the clue arm (198 clues, 195 unique; 30.9M Jev input
tokens, ~$1.30) against all 1,796 THINGS with a Noul "is this clue correct if
the secret were c?" and adds it to the posterior (a given clue holds for the
secret, so the posterior is multiplied by p_clue). Bits of a clue are measured
at the point it was given (after the question it came with); "hard" bits =
log2 of the drop in consistent concepts. Outputs: `entropy_clues_summary.json`,
`entropy_clues_<cell>.json`.

| model (clue arm) | questions | clues | soft bits / question | soft bits / clue | hard bits / question | hard bits / clue | final soft H (with clues) |
|---|---|---|---|---|---|---|---|
| qwen3.5-9b | 52 | 51 | 0.92 | 1.02 | 0.52 | 2.78 | 0.81 |
| qwen3.5-27b | 35 | 24 | 1.31 | 1.70 | 0.97 | 2.77 | 2.14 |
| qwen3.5-122b-a10b | 41 | 30 | 1.17 | 1.31 | 0.75 | 3.18 | 2.09 |
| qwen3.5-397b-a17b | 41 | 26 | 1.11 | 1.37 | 0.72 | 3.02 | 2.71 |
| glm-5.3-flash | 65 | 32 | 0.66 | 0.88 | 0.61 | 1.95 | 3.56 |
| glm-5.3 | 46 | 34 | 1.04 | 0.97 | 0.88 | 2.04 | 2.76 |
| **pooled** | 280 | 197 | **0.99** (median 0.48) | **1.16** (median 1.03) | **0.73** (median 0.29) | **2.58** (median 2.02) | |

**Which is better on average: a clue.** Pooled, a clue removes 1.16 soft bits
against 0.99 for a yes/no question (medians 1.03 vs 0.48), and 2.58 vs 0.73
bits by hard count (medians 2.02 vs 0.29); the clue is ahead for 5 of 6
models on soft bits (GLM-5.3: 0.97 vs 1.04) and for all 6 on hard bits. A
question turn that came with a clue removed 2.05 soft bits in total (median
1.67), a question turn without one 1.24 (median 0.53).

**Game-theoretic follow-up (clue request vs question).** Asking for a clue
costs Player 2 a turn, exactly like a question. On these averages a clue is
worth more than a question (1.16 vs 0.99 soft, 2.58 vs 0.73 hard), so
requesting clues would pay on average — *if* a requested clue were as
informative as a volunteered one. That is not established: Player 2 asked for
a clue once in 60 games (that clue: 1.45 soft bits, n = 1), and every other
clue was volunteered by Player 1, who chose when to give it; under competitive
goals a holder answering a request may give vaguer clues. Testing it needs
games where Player 2 requests clues (e.g. a guesser instructed or incentivised
to request, or `clue_request_uses_turn: false` vs `true`).

With clues in the measure, the final soft entropy of clue-arm games is
0.8–3.6 bits (question-only measure: ~6–8 bits), in line with 80–100% of them
being solved — the researcher's "low entropy should imply high likelihood of
solving". The true secret is still consistent with every observation at the
end in 3–9 of 10 games per model: Jev judges some clues incorrect for the
secret, as it judged 6% of answers.
