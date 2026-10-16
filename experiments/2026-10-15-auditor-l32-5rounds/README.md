# 2026-10-15 — The auditor inside the 5-player game: five rounds, layer-32 probe

## Question

The same question as `experiments/2026-10-14-auditor-in-game/` — which way of
putting the auditor into the game gets it the best results — with a detector
that is useful but imperfect, so the scenarios can differ in the probe
condition. In 2026-10-14 the layer-48 probe put every scenario at 0.96-1.00.
Also: what do the players' answers to the auditor add to the probe?
Exploratory: see `PREREGISTRATION.md`.

## Design (changes from 2026-10-14)

- Probe: the layer-32 follow-up probe
  (`probes/qwen3.8-27b_layer32_followup_logistic_regression.npz`, 2026-10-12
  fit). 3-player game message AUROC 0.72-0.73, top-1 0.80 (2026-10-12/13).
  The old Qwen3.5 layer-18 probe cannot read Qwen3.8; its Qwen3.8 counterpart
  (layers 16-24) scored ~0.53 in the sweep.
- Five discussion rounds instead of three (`five.yaml`); otherwise the same
  5-player game (4 Carrots, 1 Parsnip, threshold 3).
- Scenarios S0, S1, S2, S3, S4, S6 (S5 role claims dropped) x {none, probe}.
  9 game sets: `base` (S0 + S1 branched, x none/probe), S2/S3/S4/S6 x
  none/probe. Game seeds 6000-6049.
- The auditor's detector note now adds: answers to its questions score
  systematically higher than announcements for every player (2026-10-14:
  ~+10 log-odds for Carrots and Parsnip alike), so compare like with like.
- `analyze.py` adds probe-only metrics: top-1 from announcements only vs
  from announcements + answers, and AUROC per message kind.
- `game.py` writes `config.json` atomically (the 2026-10-14 race: parallel
  sets read it mid-write).

Unchanged from 2026-10-14: `game.py` wrapper, scenario prompts, auditor
(Qwen3.8, thinking on, reasoning effort medium), scorer, server setup.

## Deliberate limitations

- As 2026-10-14: same model for auditor and players; seeded speaking orders;
  plain-text answers; S1 paired with S0, other scenarios paired by role
  assignment only; the probe reads each message without the tool schema.
- `probe_top1` averages announcements and answers per player. In S3 only the
  questioned players have answers, which score higher for everyone, so
  questioning a player raises their mean; read S3's `probe_top1` with that
  in mind (`probe_top1_ann` is unaffected).
- The scorer left-trims dialogues to `cfg.max_len` tokens. 2026-10-14 used
  8192; five rounds made 45/366 pilot dialogues longer (max 14.4k), so this
  run uses 16384 (`batch_size` 2).
- Text-only S4 can be won by a Parsnip naming its role in private (16/50
  games in 2026-10-14); the player prompts are unchanged, for comparability.

## Run

```sh
# server (GPUs 1,2 + 0,3) and scorer (GPUs 6,7)
.venv-serve/bin/python serving/serve_qwen38.py --gpus 1,2,0,3 --data_parallel_size 2 --port 8003 \
    --hf_hub_offline true '--extra_args=--default-chat-template-kwargs={"enable_thinking": false}'
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python experiments/2026-10-15-auditor-l32-5rounds/scorer.py
for s in base S2_none S2_probe S3_none S3_probe S4_none S4_probe S6_none S6_probe; do
  HF_HUB_OFFLINE=1 .venv/bin/python experiments/2026-10-15-auditor-l32-5rounds/game.py --set $s & done; wait
.venv/bin/python experiments/2026-10-15-auditor-l32-5rounds/analyze.py
```

Pilot (never analysed): `game.py --set <s> --pilot --games 2`, `analyze.py --pilot`.

## Files

`config.json` (`common.RunConfig`), `five.yaml`, `game.py`, `scorer.py`,
`analyze.py`; outputs `games_{set}.json`, `transcripts_{set}.jsonl`,
`auditor_{set}.jsonl`, `results.json`, `figure_accuracy.png`.

## Results (2026-10-16)

450 games, 0 failed audits. Accuracy = the auditor names the Parsnip (chance
0.20), 95% game bootstrap. `results.json`, `figure_accuracy.png`. All
exploratory.

| scenario | text only | with L32 probe | probe alone, top-1 (announcements) | Carrot win (text / probe) |
|---|---|---|---|---|
| S0 passive | 0.16 [0.06, 0.28] | **0.82 [0.72, 0.92]** | 0.78 | 0.14 (shared vote) |
| S1 final questions | 0.30 [0.18, 0.42] | 0.56 [0.44, 0.70] | 0.78 | 0.14 (shared vote) |
| S2 per-round questions | 0.46 [0.32, 0.60] | 0.60 [0.46, 0.74] | 0.78 | 0.40 / 0.44 |
| S3 question budget | 0.22 [0.12, 0.34] | 0.64 [0.50, 0.76] | 0.76 | 0.24 / 0.40 |
| S4 private interviews | 0.64 [0.52, 0.78] | 0.76 [0.64, 0.88] | 0.64 | 0.22 / 0.38 |
| S6 public recommendation | 0.32 [0.20, 0.44] | 0.70 [0.56, 0.82] | 0.78 | 0.26 / 0.54 |

- The layer-32 probe is off the ceiling: announcements AUROC 0.72-0.77,
  probe-alone top-1 0.64-0.78.
- Answers to the auditor carry no layer-32 signal: answer AUROC 0.50 (S1, S2,
  S3) and 0.36 in S4 private answers (inverted). Averaging them in lowers the
  probe's own top-1 (S2 0.78 -> 0.46, S4 0.64 -> 0.14).
- With the probe, passive (S0, 0.82) is best and questioning hurts. S1 plays
  the same discussions as S0: S0 right / S1 wrong in 16 games, the reverse in
  3 (sign test p = 0.004). The auditor follows the announcement-probe ranking
  in 80% of S0 games but 54-70% elsewhere; when that ranking is right it is
  right 36/39 times in S0 but 25-33/39 in S1, S2, S3, S6 — the uninformative
  answer scores pull it off.
- In S0 the auditor adds a little over the probe alone (0.82 vs 0.78): it is
  right in 5/11 games where the probe's top player is wrong.
- Text only, interaction now helps: S2 0.46, S4 0.64 (3 rounds: 0.22, 0.48).
  Much of it is still confession (S4: the Parsnip names its role in private
  in 23/50 games, auditor 22/23; otherwise 10/27). S2 without a public
  confession: 16/39 = 0.41 (binomial vs 0.20, p = 0.002).
- Carrot win: S6 with the probe 0.54 (players alone 0.14); S2 0.40-0.44,
  S3 / S4 with probe 0.38-0.40.
- Confessions are counted with a regex ("I am the Parsnip", "as the
  Parsnip", "my role is Parsnip"), which may miss some.
