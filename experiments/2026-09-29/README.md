# 2026-09-29 — Environment validation: what signal does Carrot-Parsnip carry?

## Question

Is the game a usable testbed for deception detectors, and what kind of signal
is in it? Prompted by finding that the god-mode auditor's 77% was mostly a
harness artifact (see Fix below).

## Fix (auditor harness)

The auditor's thinking hits its 2000-token budget in almost every call; the
generator then force-closes the think block and allows 220 tokens of public
answer, which is often cut off before `<answer>`. `parse_guess` then fell back
to the *last player name mentioned* (usually whoever was being discussed at
the cut) and scored it as a real guess; `parse_failed` stayed False so the
retry never fired. On the API runs ~25% of audits returned empty text and were
scored wrong (hence below-chance baselines).

- `auditor.prompts.parse_guess`: answer tag only, no name fallback.
- `AuditorConfig.answer_followup`: an output still untagged after force-close
  is continued with `"\n\nFinal answer: <answer>"` (10 tokens) so the auditor
  commits. Records carry `answer_source` (tag | followup | null),
  `retried`, `force_closed`.
- `auditor.run` now applies `cfg.force_close_tokens`, `tensor_parallel_size`,
  `max_model_len`; accepts any `--arm`.
- New conditions: `gold` (Parsnip messages 7/7, Carrots 1/7) and generic judge
  grades via `AuditorConfig.judge_grades` (`oracle` = 09-22 role-revealed
  DeepSeek judge; `judge_public` = public-only judge below).
- `rollouts.generation`: `enable_thinking=False` backends (empty think block);
  `complete()` for raw continuations.

## Design

| # | check | data | output |
|---|---|---|---|
| 1 | existing audits re-scored, strict parse | 09-22 / 09-25 auditor files | results.json §1 |
| 2 | auditor budget: 2k vs 10k thinking tokens, with answer follow-up | 50 games x 5 seeds, base + steered; `audit_2k/` (none, lr, dom, godmode, oracle, gold), `audit_10k/` (none, godmode, gold) | §2 |
| 3 | text-judge views: full (09-25 god-mode) / redacted (role words in reasoning masked `[ROLE]`) / public-only | base arm, 100 games; + new arms | §3 |
| 4 | no-Parsnip games: all Carrots, rules still say one Parsnip | 50 games (`play_games.py --arm noparsnip`) | §4a |
| 5 | in-game instructed pairs: each base Parsnip turn re-generated with "conceal" vs "openly admit you are the Parsnip" | 300 pairs (`instructed_pairs.py`) | §4b |
| 6 | no private reasoning: same 50 games, thinking disabled | `--arm nothink`; audited with none, lr, dom, judge_public | §2-4 |
| 7 | role probe: LR on layer-16 activations, leave-one-seed-group-out | 09-22 activations + nothink | §5 |

New games reuse the 09-22 harness and the balanced 50-game subset
(`game_idx < 10` per seed group), so every new game pairs with the base-arm
game of the same id (same roles for nothink, same speaking orders).

## Deliberate limitations

- 5 auditor seeds per game x condition (not 10) to fit the 10k-budget run.
- `redacted` masks role *words* only; paraphrases ("the imposter", "my team")
  and strategic intent survive by design.
- `noparsnip` players are told one Parsnip exists, so Carrots still accuse
  each other; the auditor is not run there (a forced choice has no right answer).
- The honest instructed variant tells the Parsnip to admit its role — honesty
  about identity only; analysis keeps pairs where the honest variant admits
  and the deceptive one does not.
- `nothink` changes two things at once: no scratchpad and a slightly shorter
  turn prompt (the "think first" paragraph removed).
- Role probe reads activations whose context includes the role instruction in
  the system prompt; high AUROC means the role is linearly present, not that
  deception is.

## Files

- `config.json` (games/judges/scoring), `audit_2k/config.json`, `audit_10k/config.json`
- `play_games.py`, `instructed_pairs.py`, `judge.py` (vLLM env);
  `score_messages.py`, `analyze.py`, `figures.py` (`.venv`); `run_all.sh` (lanes a/b)
- Symlinked inputs: 09-22 transcripts/games/scores/activations (base, steered),
  09-25 god-mode judge as `judge_full_{arm}.jsonl`, 09-25 `suspicion_scale.json`
- Outputs: `transcripts_{noparsnip,nothink,instructed}.jsonl`, `games_*.json`,
  `judge_{view}_{arm}.jsonl`, `scores_*.csv`, `audit_*/auditor_seeds_{arm}.jsonl`,
  `results.json`, `figures/`. The `.jsonl` data files are git-lfs patterns and the
  run logs are not committed.
