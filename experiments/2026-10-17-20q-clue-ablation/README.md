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

(after the run)
