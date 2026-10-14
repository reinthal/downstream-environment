# FIBSv1 activations of Qwen3.5-27B (2026-10-13)

Data preparation, not a hypothesis test: residual-stream activations of
`Qwen/Qwen3.5-27B` on every row of FIBSv1 (`AlignmentResearch/fibs-v1`, the
training set of the "Caught in the Act" probes), so that new probes can be
trained on them here. Training or evaluating a probe on these activations is a
new experiment and starts with `/preregister`.

## What is stored

- **Rows**: all 163,016 rows of the four quadrant files, numbered in the order
  `honest_honest, honest_deceptive, deceptive_honest, deceptive_deceptive`
  (`rows.parquet`, column `row`). 1,152 rows render to more than 32,768 tokens
  and are dropped, not trimmed (the paper's rule, B.2); 161,864 rows
  (167,332,285 tokens) are kept.
- **Input**: `messages_json` plus the assistant `completion`, rendered with the
  model's chat template and an empty think block
  (`<|im_start|>assistant\n<think>\n\n</think>\n\n{completion}<|im_end|>`),
  with the row's tool schemas when it has any. No generation: the text is the
  dataset's.
- **Token**: the one before the final `<|im_end|>` (the read window
  `second-last-token-generation` of the paper's linear and MLP probes).
- **Layers**: decoder blocks 19, 27, 35, 43, 51, 58 (block output = HF
  `hidden_states[k + 1]`), bf16, model revision `fc05daec`.
- **Files**: `acts_fibs/layer_{L}.pt`, a `(161864, 5120)` bf16 tensor per
  layer; row `i` is the `rows.parquet` row with `act_index == i`.
  `rows.parquet` also carries `quadrant`, `base_deceptive`,
  `followup_deceptive`, `source`, `deception_type`, `follow_up_category`,
  `language`, `pair_id`, `pair_type`, the content md5 and `n_tokens`.
  `acts_fibs/` is gitignored (about 10 GB, regenerable).

## Checks

- Pilot (401 rows, including the longest kept row at 32,550 tokens): the hook
  output equals `hidden_states[L + 1]` at the read token on 8 rows, all six
  layers, bit for bit.
- `check` stage: the published linear ensemble
  (`AlignmentResearch/probe-inference-weights@05def1c1`, `qwen3.5-27b/linear`)
  scored on the stored activations, follow-up deceptive vs honest. Pilot:
  AUROC 0.989 on 401 rows (the card reports 0.972 on its validation rows).
  Full run: AUROC 0.987 on all 161,864 kept rows (`acts_fibs/check.json`;
  its per-source breakdown is over 683 `source` values, many tiny, so it is
  not meaningful family by family).

## Deliberate limitations

- **One token only.** The researcher chose "Read token, six fibs layers"
  (2026-10-13). EFC- and axial-style probes need the whole read window (final
  user turn through end of turn: 76M tokens, about 4.7 TB at six layers), which
  is not stored. Other layers are not stored either.
- **No train/validation split.** FIBSv1 carries no marker for the paper's
  152,980 / 5,000 split; choosing one is part of the probe-training
  experiment. The published probes were trained on most of these rows, so the
  `check` AUROC is mostly in-sample.
- **Reasoning is not in context.** `completion_reasoning` (8,937 rows) is not
  rendered (`reasoning_in_context` is false on every row); `reasoning_content`
  on earlier assistant turns (73 caving rows) is left to the chat template,
  which drops it for turns before the last user message.
- **Tool-call arguments** are stored in fibs as JSON strings and are parsed to
  mappings before templating (the template iterates over them).
- **HF bare decoder, not vLLM.** The paper captured activations with vLLM
  hooks; these come from the truncated HF decoder
  (`linear_probes.activations.load_truncated_decoder`), with bf16 batch noise.

## Run

```
uv run --no-sync python experiments/2026-10-13-fibs-acts/collect.py --stage render
for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((2*s)),$((2*s+1)) uv run --no-sync python \
  experiments/2026-10-13-fibs-acts/collect.py --stage collect --shard $s > experiments/2026-10-13-fibs-acts/logs/collect_shard$s.log 2>&1 & done
uv run --no-sync python experiments/2026-10-13-fibs-acts/collect.py --stage merge
uv run --no-sync python experiments/2026-10-13-fibs-acts/collect.py --stage check
```

`collect` continues after the chunks already written if it is re-run. All
eight GPUs were used with the researcher's approval; the Qwen3.8 server was
stopped for the run and, on the researcher's instruction, not restarted. The
first pass ran out of memory on all four shards once ~200-token rows reached
batches above 79 rows; `acts_max_batch` went from 128 to 64 and the remaining
rows were resumed (`logs/collect_shard*_resume.log`).
