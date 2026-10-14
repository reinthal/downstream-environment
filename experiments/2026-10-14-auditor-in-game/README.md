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
