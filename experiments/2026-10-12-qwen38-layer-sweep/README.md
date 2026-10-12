# 2026-10-12 — Follow-up probe layer sweep on Qwen3.8-27B

## Question

Does a follow-up ("did you lie") probe read at a late layer separate the
Parsnip far better than at the ~28%-depth layer used so far — and if so, is
it reading lying, or the role? The researcher recalls such a result from a
lost sweep (`experiments/2026-10-06/`, cited in the 2026-10-01 README, never
committed). Exploratory: see `PREREGISTRATION.md`.

## Design

| step | script | env | what |
|---|---|---|---|
| 0 | `serving/serve_qwen38.py` | `.venv-serve` | Qwen3.8-27B on GPUs 0,3 (TP2, one replica), port 8003, thinking off by default |
| 1 | `generate_roleplay.py` | `.venv` | on-policy replies to the 371 Apollo roleplaying scenarios (2026-10-01 recipe) |
| 2 | `grade.py` | `.venv` | 1-7 deception grades, Qwen3.8 grading its own replies (2026-10-01 local grader) |
| 3 | `play_games.py` | engine reqs | 100 three-player games (`game.yaml`, seeds 1000-1099), engine orchestrator |
| 4 | `extract.py` | engine reqs | Inspect logs → `transcripts_base.jsonl`, `games_base.json` (copy of 2026-10-09) |
| 5 | `instructed.py` | `.venv` | every Parsnip turn replayed twice: conceal (unchanged) vs admit |
| 6 | `collect.py` | `.venv`, GPUs 4,5 | residual stream on the follow-up "No" at layers 0, 8, …, 56, 63 |
| 7 | `analyze.py` | `.venv` | per-layer LR probe (2026-10-01 recipe) → `results.json`, figure |

## Deliberate limitations

- Qwen3.8 grades its own roleplaying replies (as 2026-10-01's local grader),
  not the paper's GPT-4o.
- One reply per scenario: about 60 labelled held-out replies per layer.
- Layer 63 is the raw last-block output (the final norm is removed by the
  truncated decoder), not the normed `hidden_states[-1]`.
- The probe reads each game turn with the speaker's explicit reasoning turn
  in context, as 2026-10-09 did; it never saw that format in training.
- No all-Carrot control and no black-box Yes/No baseline (not chosen).
- The engine's speaking order uses an unseeded RNG (as 2026-10-09).

## Run

From the repo root (`uv` in `~/.local/bin`):

```sh
E=experiments/2026-10-12-qwen38-layer-sweep
.venv-serve/bin/python serving/serve_qwen38.py --gpus 0,3 --data_parallel_size 1 --port 8003 \
    --hf_hub_offline true '--extra_args=--default-chat-template-kwargs={"enable_thinking": false}'
export HF_HUB_OFFLINE=1
.venv/bin/python $E/generate_roleplay.py
.venv/bin/python $E/grade.py
uv run --no-project --with-requirements environments/carrot_parsnip/requirements.txt python $E/play_games.py --arm base
uv run --no-project --with-requirements environments/carrot_parsnip/requirements.txt python $E/extract.py --arm base
.venv/bin/python $E/instructed.py
for s in roleplaying games instructed; do CUDA_VISIBLE_DEVICES=4,5 .venv/bin/python $E/collect.py --set $s; done
.venv/bin/python $E/analyze.py
```

Pilot (never analysed): `generate_roleplay.py --limit 4`, `grade.py --pilot`,
`play_games.py --arm pilot --games 2`, `extract.py --arm pilot`,
`instructed.py --arm pilot`, `collect.py --set <s> --pilot`.

## Files

- `PREREGISTRATION.md`, `config.json` (`common.SweepConfig`), `game.yaml`
- Outputs: `rollouts_roleplaying.jsonl`, `judge_roleplaying.jsonl`,
  `inspect_logs_base/`, `summary_base.json`, `transcripts_base.jsonl`,
  `games_base.json`, `excluded_base.json`, `transcripts_instructed.jsonl`,
  `probe_followup_layer{L}.npz`, `results.json`, `figure_auroc_by_layer.png`;
  `activations_*.npz` are untracked.
