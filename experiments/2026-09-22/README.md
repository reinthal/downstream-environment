# Carrot-Parsnip: does a deception probe change an outside auditor's win rate?

An outside auditor (no access to ground truth or to any player's private role
instruction) reads the public transcript of a Carrot-Parsnip game and names the
Parsnip. We measure how many of 100 games the auditor calls correctly under
four detector conditions, and separately whether steering agents along the
probe direction changes the outcome.

Model everywhere (players + auditor): `Qwen/Qwen3.5-27B` (local HF transformers).

## Pipeline

| step | script (env) | output |
|---|---|---|
| 1 | `train_probes.py` (`.venv`) | `probes/probe_{logistic_regression,diff_of_means}.npz`, `probes/steer_diff_of_means.npz`, layer choice |
| 2 | `run_games.py --arm base` (`.venv-vllm`) | `transcripts_base.jsonl`, `games_base.json` |
| 3 | `run_games.py --arm steered --backend hf --steer probes/steer_diff_of_means.npz --coef 8` (`.venv`) | steered arm of the same 100 games |
| 4 | `score_messages.py --arm {base,steered}` (`.venv`) | `scores_{arm}.csv` (LR / DoM / random probe per message) |
| 5 | `run_auditor.py --arm {base,steered}` (`.venv-vllm`) | `auditor_{arm}.jsonl` (4 conditions x 100 games) |
| 6 | `analyze.py` (`.venv`) | `results.json`, `figures/auditor_accuracy.png` |

Browse transcripts + scores + auditor verdicts with the reusable viewer:
`python3 -m http.server 8000 --bind 127.0.0.1` from the repo root, then open
`http://localhost:8000/rollouts/viewer/?data=/experiments/2026-09-22`
(tunnel the port if the repo is on a remote box).

Generation goes through the reusable `rollouts.generation` backends (see
`AGENTS.md`): vLLM by default (`.venv-vllm/bin/python`, ~10x faster), HF
transformers (`uv run --no-sync python`) for steering. The original
2026-09-22/23 results were produced entirely on the HF path; the vLLM backend
was added afterwards, so re-runs on vLLM will differ in sampled tokens (same
seeds for roles/orders).

## Design

- **Games.** 5 seed groups x 20 games (seeds `1000*group + idx`). 5 players
  (4 Carrots, 1 Parsnip, assigned by the game engine's seeded RNG), 3 public
  discussion rounds, randomized speaking order (previous round's last speaker
  never speaks first). The game **stops before any vote** — the auditor's guess
  is the only identification, so "win rate" = auditor accuracy (Carrot side).
- **Turns.** One generation per turn: the model thinks privately
  (`<think>...</think>`) and the post-think text is posted verbatim. Role
  instructions live only in the system prompt; the public transcript never
  contains them. If a think block doesn't close within budget it is force-closed
  and the announcement generated in a bounded continuation.
- **Probes.** Instructed-pairs recipe from experiments/2026-09-20
  (Goldowsky-Dill et al. 2025; `repe_honesty__plain`, 306 facts x
  honest/dishonest instruction) on Qwen3.5-27B residual stream, layer chosen
  from {16,24,32,40,48} by held-out AUROC (all tied at 1.0 -> layer 16, closest
  to the paper's fractional depth 22/80). Scores = sigmoid of the linear read-out,
  mean-pooled over the exact generation context. Two poolings per message:
  public announcement tokens only (`*_pub`) and the full assistant turn incl.
  the private think block (`*_full`). The auditor is shown `*_full` — chosen
  from message-level diagnostics (pub-span parsnip-vs-carrot AUROC ~0.48 =
  chance; full-span ~0.61) before any auditor was run. The auditor never sees
  the reasoning text or the system prompt, only the score.
- **Conditions.** `none` (transcript only), `lr` (logistic-regression probe),
  `dom` (difference-of-means probe), `random` (random unit direction per seed
  group, calibrated on training activations — controls for "any number next to
  each message" effects). Identical auditor prompt in all probe conditions;
  the auditor is told scores come from an imperfect activation-reading lie
  detector, never which one.
- **Steering arm.** Diff-of-means direction in raw activation space
  (`deceptive mean - honest mean`, layer 16, |v| = 1.8 = one honest<->dishonest
  class gap, vs mean residual norm ~53) added to the residual stream at the
  probe layer during each speaker's generation: Parsnip gets `-C*v` (toward
  honest), Carrots `+C*v` (toward deceptive). C = 8 (8 class gaps), picked by a
  small calibration: generations stayed coherent at +-8 (and even +-16);
  smaller C had little visible effect. Steered messages are re-encoded
  **without** the steering hook when scored, so probe scores in the steered arm
  reflect behavioural change in the text, not the mechanically injected vector.
- **Primary metric.** Correct Parsnip identifications out of 100 per condition,
  with per-seed-group breakdown, Wilson 95% CIs, and exact McNemar tests vs
  `none` (paired by game).

## Deliberate limitations (scope kept small)

1. **One probe layer, mean-pooled message scores.** The paper scores per token
   and tunes layers per model; we pick one layer by instructed-pairs AUROC
   (which saturated at 1.0, so the choice is under-determined) and pool one
   vector per message. No per-token display to the auditor.
2. **Role != lying.** Ground truth is the *role*, not message-level deception.
   A cautious Parsnip may never assert a falsehood; probes may track role
   strategy, concealment, or style rather than lies. We report
   parsnip-vs-carrot message AUROC as a diagnostic, not truth.
3. **Instructed-pairs training only.** Probes are trained on
   "pretend to be dishonest" pairs (the paper's cheapest recipe) and
   transferred zero-shot to the game. No on-policy or game-specific
   calibration; the decision threshold shown to the auditor is the raw
   sigmoid score, not an FPR-calibrated one.
4. **Same model plays everyone.** Players, Parsnip, and auditor are all
   Qwen3.5-27B; no model diversity, so results may not transfer to
   mixed-model games.
5. **Sampling is only approximately reproducible.** Batched sampling with a
   per-slot torch seed; changing batch geometry (chunk size, GPU count)
   changes the drawn tokens. Game seeds fully determine roles and speaking
   order, not the text.
6. **Steered and base arms share seeds but not trajectories.** Steering
   changes what is said from turn 1, so arms are compared at the game level
   (same roles/order), not message level. One steering coefficient (C = 8, a
   single calibrated point, not a sweep), one layer, both signs applied
   simultaneously (Parsnip down, Carrots up) — effects of each sign alone are
   not isolated. Steered-arm probe scores are computed from unsteered
   re-encodes (behavioural readout); a monitor reading steered activations
   directly would additionally see the injected vector itself.
7. **No DYL ("did you lie") probe, no shuffled-scores or oracle baselines**
   from the MVP planning doc — only the minimal 4 conditions.
8. **Auditor forced-choice.** One guess per game, no confidence elicitation;
   unparseable answers after one retry count as incorrect (tracked as
   `parse_failed`).
