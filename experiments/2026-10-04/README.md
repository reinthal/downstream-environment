# 2026-10-04 — Auditor analyses the raw scores itself (label-free)

## Question

Can the auditor do the statistical analysis itself, using only the current
game's data, and does a Python tool or a stronger model (Claude Opus 5.5)
help?

## Design

- `build_scores.py` → `raw_{fu,lr,rand}_base.json`: each game's 15 messages
  with round, speaking position, player and raw detector score (log-odds).
  Nothing from other games, no roles.
- The auditor sees the transcript plus a score table. The system prompt says
  only that these are raw scores to analyse as it sees fit (`raw_note`); tool
  conditions add a sandboxed Python tool with the table as `scores.csv`
  (`auditor/tools.py`, `auditor/sandbox.py`: no network, 30 s, 2 GB).
- Conditions: `none`, `fu_tool`, `fu_notool`, `lr_tool`, `rand_tool`,
  `gold_prob`. Qwen3.5-27B: `audit_a/`, `audit_b/` (100 games x 1 seed each,
  10k thinking tokens). Opus 5.5 via OpenRouter (`auditor/api_run.py`,
  adaptive thinking, effort medium): `audit_opus/`, 100 games x 1 seed, none /
  fu_tool / fu_notool / rand_tool.
- `analyze_audits.py`, `analyze_opus.py` → `results.json`, `results_opus.json`:
  accuracy with game-bootstrap CIs next to the detector alone (highest raw
  average; highest average after a within-game speaking-position correction),
  overrides, tool use.

## Deliberate limitations

- One Opus seed (cost); 100 games, so accuracies carry roughly ±0.08.
- No hints in the prompts about drift or position effects.
- "Code adjusted for position / round" is a regex over the tool code — a
  rough indicator.

## Files

- `audit_{a,b,opus}/config.json`
- Run: `.venv/bin/python experiments/2026-10-04/build_scores.py`;
  `PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python -m auditor.run --config experiments/2026-10-04/audit_a/config.json --arm base`
  (same for `audit_b`); `.venv-api/bin/python -m auditor.api_run --config experiments/2026-10-04/audit_opus/config.json --arm base`
  (key in `OPENROUTER_API_KEY` or the repo-root `.env`)
