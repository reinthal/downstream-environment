# 2026-10-09 — 3-player Carrot-Parsnip: auditor speaking-position bias

## Question

In a 3-player game played on the new engine (`environments/carrot_parsnip`,
YAML `GameConfig`, default prompts), does the outside auditor over-pick a
speaking position — the first speaker of round 3 — and do raw probe scores
change that? See `PREREGISTRATION.md` for the hypothesis, the single primary
metric and the decision rule.

## Design

- **Games**: `game.yaml` — 2 Carrots + 1 Parsnip, ejection threshold 2, three
  discussion rounds, engine default prompts. Played by the engine's own
  Inspect orchestrator (`Carrot_Parsnip_Agents.run_games`): each turn is an
  explicit reasoning call (hidden reasoning off) followed by a tool-call
  announcement; the players then vote. Qwen3.5-27B via OpenRouter, 100 games,
  seeds 1000-1099.
- **Transcripts**: `extract.py` reads announcements, realised speaking orders
  and votes back out of the per-game Inspect logs into the repo's
  `transcripts_/games_` files (viewer- and auditor-compatible).
- **Probe**: `score_probe.py` — the L18 follow-up ("did you lie") probe served
  on RunPod; one raw log-odds score per announcement, in the 2026-10-04
  `raw_fu_{arm}.json` format.
- **Auditor**: Claude Opus 5.5 via OpenRouter (`auditor.api_run`,
  `audit/config.json` copied from `experiments/2026-10-04/audit_opus/`):
  `none` (transcript only) and `fu_notool` (transcript + raw score table), one
  audit per game per condition.
- **Analysis**: `analyze.py` → `results.json`.

## Deliberate limitations

- One game arm: no arm without the "what would each role have said" line and
  no 5-player arm, so a bias (or its absence) cannot be attributed to the
  prompt or to the 3-player shape.
- The engine draws speaking orders from an unseeded RNG; orders are recorded
  from the logs, not reproducible from the seed.
- The engine has no provider setting: OpenRouter routes each call freely
  across providers with different quantization (recorded per turn where the
  log keeps the raw call).
- The probe was trained on single-turn roleplay replies with reasoning
  dropped; here it reads a dialogue that includes the player's explicit
  reasoning turn. No random-direction control (not served; no GPU here).
- One auditor seed; ~67 games carry the primary metric (interval about ±0.11).
- `audit/suspicion_scale.json` is a copy of the 2026-10-04 one: the auditor
  loader requires the file, but neither condition here uses grades.

## Files

- `PREREGISTRATION.md`, `game.yaml`, `config.json` (`common.GameRunConfig`),
  `audit/config.json`
- Run (repo root; `uv` comes from `nix shell nixpkgs#uv` on this host):

  ```sh
  set -a; source .env; set +a
  E=experiments/2026-10-09-3p-game
  uv run --no-project --with-requirements environments/carrot_parsnip/requirements.txt python $E/play_games.py --arm base
  uv run --no-project --with-requirements environments/carrot_parsnip/requirements.txt python $E/extract.py --arm base
  uv run --no-project --with requests --with transformers --with jinja2 python $E/score_probe.py --arm base
  .venv-api/bin/python -m auditor.api_run --config $E/audit/config.json --arm base
  .venv-api/bin/python $E/analyze.py
  ```

  Pilot: `--arm pilot --games 2` (seeds 0-1; `*_pilot*` files are never analysed).
- Outputs: `inspect_logs_{arm}/`, `summary_{arm}.json`,
  `transcripts_{arm}.jsonl`, `games_{arm}.json`, `excluded_{arm}.json`,
  `scores_{arm}.csv`, `raw_fu_{arm}.json`, `audit/auditor_{arm}.jsonl`,
  `results.json`
- Viewer: `http://localhost:8000/rollouts/viewer/?data=/experiments/2026-10-09-3p-game&arms=base`
