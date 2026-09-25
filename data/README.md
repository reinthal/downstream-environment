# data/ — reusable datasets (symlinks)

Canonical entry points to sampled data produced by experiments. Each link
points into the experiment folder that generated it (the target files are
git-lfs-tracked). Load through these paths; don't reach into other
experiments' directories.

| link | from experiment | contents |
|---|---|---|
| `qwen3.5-27b_sandbagging_rollouts.jsonl` | `2026-09-24` | 1000 on-policy Qwen3.5-27B rollouts on the Apollo sandbagging_v2 (wmdp_mmlu) prompts; fields incl. `reasoning`, `public`, rule-grader `grade` (honest 479 / deceptive 333 / ambiguous 29 / skip 159) |
| `qwen3.5-27b_alpaca_rollouts.jsonl` | `2026-09-24` | 2000 on-policy Qwen3.5-27B rollouts on alpaca control prompts (benign; used for the 1%-FPR threshold) |
| `carrot-parsnip_qwen3.5-27b_transcripts_base.jsonl` | `2026-09-22` | 1500 game messages (100 discussion-only games x 15 turns), unsteered; per-message `input_messages`, private `reasoning`, public `announcement`, role |
| `carrot-parsnip_qwen3.5-27b_transcripts_steered.jsonl` | `2026-09-22` | same 100 games re-rolled with diff-of-means steering (Parsnip −8v honest, Carrots +8v deceptive) |
| `carrot-parsnip_qwen3.5-27b_games_base.json` | `2026-09-22` | per-game metadata (seeds, roles, speaking orders), base arm |
| `carrot-parsnip_qwen3.5-27b_games_steered.json` | `2026-09-22` | same, steered arm |
| `carrot-parsnip_qwen3.5-27b_judge_oracle_base.jsonl` | `2026-09-22` | per-message deception labels from the DeepSeek V4.1 Flash oracle judge (Apollo 7-point scale, speaker role revealed), base arm |
| `carrot-parsnip_qwen3.5-27b_judge_oracle_steered.jsonl` | `2026-09-22` | same, steered arm |

Full experiment designs and limitations: `experiments/<date>/README.md`;
run history: `research-log/log.md`.
