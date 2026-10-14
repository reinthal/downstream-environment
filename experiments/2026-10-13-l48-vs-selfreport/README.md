# 2026-10-13 — Layer-48 follow-up probe vs the model's own answer

## Question

The 2026-10-12 sweep found the follow-up probe near perfect from layer 40 on.
Does it see more than the model would say if simply asked? Compared with the
model's own answer to the follow-up question, P(Yes), in normal games and in
games where the Parsnip is told to deny lying when asked. Exploratory: see
`PREREGISTRATION.md`.

## Design

| step | script | env | what |
|---|---|---|---|
| 0 | `serving/serve_qwen38.py` | `.venv-serve` | Qwen3.8-27B on the first two free GPUs, port 8003, thinking off by default |
| 1 | `play_games.py --arm {normal,deny}` | engine reqs | 100 games per arm, seeds 2000-2099 (`normal.yaml`, `deny.yaml`) |
| 2 | `extract.py --arm …` | engine reqs | Inspect logs → `transcripts_{arm}.jsonl`, `games_{arm}.json` |
| 3 | `pyes.py --arm …` | `.venv` | log P(Yes) − log P(No) on the follow-up question, per turn |
| 4 | `collect.py --arm …` | `.venv`, the same two GPUs after the server stops | follow-up "No" activations at layers 32, 40, 48 |
| 5 | `analyze.py` | `.venv` | 2026-10-12 probes (via `probes/`) vs P(Yes) → `results.json`, figure |

## Deliberate limitations

- The deny line is in the Parsnip's private knowledge for the whole game, so
  it can change play as well as the follow-up answer.
- Probes are the 2026-10-12 fits (roleplay-trained); layer 48 was chosen on
  roleplay held-out AUROC, but the 2026-10-12 game results were seen before
  this run was designed.
- P(Yes) and the probe read the same rendered dialogue (thinking off, no tool
  schema), which is not exactly what the server showed the player.
- Speaking orders are unseeded (engine), so the arms are paired by role
  assignment only.

## Run

`bash experiments/2026-10-13-l48-vs-selfreport/run_all.sh games` (server up), then
stop the server and `run_all.sh probe <gpu,gpu>` on the same two GPUs, from the repo
root, server up (command as in `experiments/2026-10-12-qwen38-layer-sweep/README.md`).
Pilot (never analysed): `play_games.py --arm normal_pilot --games 1` (and
`deny_pilot`), then `extract.py` / `pyes.py` / `collect.py` with the same arm.
