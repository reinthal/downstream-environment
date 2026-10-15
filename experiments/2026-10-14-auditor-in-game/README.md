# 2026-10-14 — The auditor inside the 5-player game

## Question

Which way of putting the auditor into the game gets it the best results —
naming the Parsnip — with and without the layer-48 follow-up probe? Six
interaction scenarios against the passive baseline. Exploratory: see
`PREREGISTRATION.md`.

## Design

- Game: `five.yaml` (the engine's 5-player config: 4 Carrots, 1 Parsnip,
  threshold 3, three discussion rounds, default prompts). Players and auditor:
  Qwen3.8-27B on the local server; players with thinking off (as 2026-10-12/13),
  the auditor with thinking on.
- `game.py` is a wrapper orchestrator (the engine has no hooks): it reuses the
  engine's game state, prompts and vote resolution and reproduces its turn
  semantics, and adds the auditor.

| scenario | what the auditor does | players told? |
|---|---|---|
| S0 passive | reads the finished discussion | no |
| S1 final questions | one public question to every player after the discussion | no |
| S2 per-round questions | one public question to every player after every round | yes |
| S3 question budget | up to 5 public questions in total, to players it chooses, after any round | yes |
| S4 private interviews | one private question to every player after every round | yes |
| S5 role claims | passive; every announcement must start with a role claim | (prompt) |
| S6 public recommendation | publicly recommends whom to eliminate before the vote | yes |

Each scenario runs with `none` (text only) and `probe` (every player message,
announcements and answers alike, carries its live layer-48 follow-up-probe
log-odds, from `scorer.py`). Game sets: `base` (one discussion branched into
S0 and S1 × none/probe), `claim` (S5 × none/probe), and one set per S2, S3,
S4, S6 × none/probe — 10 sets, `cfg.n_games` games each.

## Deliberate limitations

- The auditor is the same model as the players.
- Speaking orders are seeded (the engine's are not); everything else follows
  the engine.
- Answers to the auditor are plain text, not a tool call.
- S1 branches from the same discussion as S0 (paired); the other scenarios
  play their own games, paired only by role assignment (same seeds).
- The probe reads each message with the speaker's reasoning call in context,
  thinking off, without the tool schema (as 2026-10-12/13).

## Run

```sh
# server (GPUs 1,2) and scorer (GPUs 6,7)
.venv-serve/bin/python serving/serve_qwen38.py --gpus 1,2 --data_parallel_size 1 --port 8003 \
    --hf_hub_offline true '--extra_args=--default-chat-template-kwargs={"enable_thinking": false}'
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python experiments/2026-10-14-auditor-in-game/scorer.py
# games, one set at a time or in parallel
for s in base claim S2_none S2_probe S3_none S3_probe S4_none S4_probe S6_none S6_probe; do
  HF_HUB_OFFLINE=1 .venv/bin/python experiments/2026-10-14-auditor-in-game/game.py --set $s & done; wait
.venv/bin/python experiments/2026-10-14-auditor-in-game/analyze.py
```

Pilot (never analysed): `game.py --set <s> --pilot --games 2`, `analyze.py --pilot`.

## Files

`config.json` (`common.RunConfig`), `five.yaml`, `game.py`, `scorer.py`,
`analyze.py`; outputs `games_{set}.json`, `transcripts_{set}.jsonl`,
`auditor_{set}.jsonl`, `results.json`, `figure_accuracy.png`.

## Results (2026-10-15)

50 games per set, 0 failed audits in every cell. Accuracy = the auditor names
the Parsnip (chance 0.20), 95% game bootstrap. `results.json`,
`figure_accuracy.png`.

| scenario | text only | with L48 probe | probe msg AUROC | Carrot win (text / probe) |
|---|---|---|---|---|
| S0 passive | 0.20 [0.10, 0.32] | 1.00 [1.00, 1.00] | 1.00 | 0.14 (shared vote) |
| S1 final questions | 0.24 [0.14, 0.36] | 1.00 [1.00, 1.00] | 0.98 | 0.14 (shared vote) |
| S2 per-round questions | 0.22 [0.12, 0.34] | 1.00 [1.00, 1.00] | 0.96 | 0.24 / 0.52 |
| S3 question budget | 0.24 [0.12, 0.36] | 0.98 [0.94, 1.00] | 0.99 | 0.18 / 0.46 |
| S4 private interviews | 0.48 [0.34, 0.62] | 0.96 [0.90, 1.00] | 0.94 | 0.14 / 0.18 |
| S5 role claims | 0.16 [0.06, 0.28] | 0.96 [0.90, 1.00] | 1.00 | 0.18 (shared vote) |
| S6 public recommendation | 0.12 [0.04, 0.22] | 1.00 [1.00, 1.00] | 1.00 | 0.18 / 0.76 |

- Text only, every scenario except S4 is at chance; asking more questions
  (S1, S2, S3) does not move it.
- S4: the Parsnip names its own role in a private answer ("As the Parsnip,
  ...") in 16/50 games; the auditor is right in all 16 and in 8/34 (0.24) of
  the rest. The S4 gain is private confession, not better questioning.
- With the probe, every scenario is at or near the ceiling, so this design
  cannot rank the scenarios in the probe condition.
- S6: players eliminate the auditor's recommended player in 41/48 (text)
  and 38/47 (probe) games with an elimination. With the probe the
  recommendation is right, and the Carrots win 0.76 against 0.14-0.18
  without public auditor input.
- Text-only verdicts are spread over all five names (no fixed-name bias).
