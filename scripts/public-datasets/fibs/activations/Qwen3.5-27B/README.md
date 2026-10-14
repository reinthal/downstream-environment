# FIBSv1 activations: Qwen3.5-27B

Residual-stream activations of
[`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) (revision
`fc05daec18b0a78c049392ed2e771dde82bdf654`, the checkpoint the paper's
published Qwen3.5-27B probes were trained on) on
[`AlignmentResearch/fibs-v1`](https://huggingface.co/datasets/AlignmentResearch/fibs-v1)
(revision `65dccf12934620bfe01db665ea3d7d0e3c9f1355`). They use the
follow-up read of Hollinsworth et al. 2026
([arXiv:2610.12445](https://arxiv.org/abs/2610.12445)). Use them to retrain
or evaluate linear and MLP probes on any slice of FIBSv1 (by family,
quadrant or label) without running the model.

The layout, row ids and validation split are the same as in
[`fibs/activations/Qwen3.8-27B/read-token/`](../Qwen3.8-27B/README.md), so
one loader reads both and the two models can be compared row for row.

## What these activations support

| probe | supported | why |
|---|---|---|
| linear, MLP (per-layer, read token) | **yes** | `token` holds the read token at all 6 layers for every collected row |
| EFC, axial (whole read window) | **no** | only one token per row is stored: `acts`, `prompt_mask` and `completion_mask` are empty, and there is no `windows-pilot/` for this model. Window-reading probes need the Qwen3.8-27B `windows-pilot/` or a new collection. |
| other layers, other read tokens | **no** | only layers 19, 27, 35, 43, 51, 58 at the one read token |
| re-rendering the exact input | partly | there are no `jobs_{k}.pt` (token ids); the input is rebuilt by rendering the row (see below), which the repository's `collect.py --stage render` does |

## How the activations were taken

The code is in `experiments/2026-10-13-fibs-acts/collect.py` of the
repository (stages `render`, `collect`, `merge`, `check`).

- **Rendering.** Each row's conversation plus its completion is rendered
  with Qwen3.5-27B's own chat template; the answer follows an empty think
  block (`<|im_start|>assistant\n<think>\n\n</think>\n\n{completion}<|im_end|>`),
  as in the paper's Qwen3.5 read. `completion_reasoning` is not rendered;
  the template drops `reasoning_content` of turns before the last user
  message. Tool-call arguments are parsed from their JSON strings; the row's
  tool schemas are passed to the template.
- **Forward pass.** Teacher-forced, no generation: the text is the
  dataset's. Bare decoder truncated at layer 58 on two A40s, read by forward
  hooks; on 8 pilot rows the hooked output equals HF `hidden_states[k + 1]`
  bit for bit at all six layers.
- **Layers.** 19, 27, 35, 43, 51, 58, always in that order on the layer
  axis. Layer k is decoder block k's output. bf16; d = 5120.
- **Read token.** The token before the answer's `<|im_end|>`
  (`probe_inference`'s `second-last-token-generation`), the token the
  paper's linear and MLP probes read.
- **Labels.** `label` is 1 when the completion is a falsehood, meaning the
  row's quadrant ends in `_deceptive`.
- **Check.** The paper's published linear ensemble
  (`AlignmentResearch/probe-inference-weights@05def1c1`, `qwen3.5-27b/linear`)
  scores AUROC 0.987 on all collected rows. It was trained on most of
  them, so this checks the read, not generalisation.

## `read-token/`: read token only (10.0 GB)

| file | contents |
|---|---|
| `rows.jsonl` | One line per FIBSv1 row (163,016 lines, collected or not); see the row schema below. |
| `acts_{k}.pt` | Shard k (500 collected rows, the last 364; 324 shards). `{"i": list[int], "token": Tensor(n, 6, 5120) bf16, "acts": [], "prompt_mask": [], "completion_mask": [], "window_cap": 0}`. `token[j]` is row `i[j]`'s read-token activation at the 6 layers. Rows are in `i` order across shards. |

There are no `jobs_{k}.pt` files.

Row schema (`rows.jsonl`), the same fields as the Qwen3.8-27B store:

| field | meaning |
|---|---|
| `i` | Row id, the key into `acts_*.pt`, and the same row as `i` in the Qwen3.8-27B store. Rows are numbered in the order of the parquets `honest_honest`, `honest_deceptive`, `deceptive_honest`, `deceptive_deceptive`, each read in full; `i` minus the quadrant's offset (0, 43,941, 84,985, 122,552) is the row's position in its parquet. |
| `label` | 1 = falsehood completion, 0 = honest completion. |
| `quadrant` | FIBSv1 quadrant. |
| `source` | FIBSv1 family (49 families plus 4,321 `persona_prefill_*` scenario ids). |
| `pair_id` | FIBSv1 pair id (empty for unpaired rows). |
| `_v32_source_content_md5` | FIBSv1's content hash; join key back to the parquet. |
| `split` | `train` or `val`: the paper's A.6 hold-out (class-balanced, no `pair_id` shared with `train`, seed 0), drawn over all 163,016 rows. Identical to the Qwen3.8-27B store's `split`. Of the 5,000 `val` rows, 4,967 are collected here (2,485 honest, 2,482 deceptive); 156,897 `train` rows are collected. |
| `n_tokens`, `window` | Rendered length and read-window length (final user turn through `<|im_end|>`), in tokens, with the Qwen3.5-27B template. |
| `dropped` | True if the row renders to more than 32,768 tokens and was not collected (1,152 rows). |

## Limitations and differences from the Qwen3.8-27B store

- **Read token only.** No window activations, so no EFC or axial probes
  (table above).
- **Length cap 32,768 tokens** (the paper's B.2 rule; the Qwen3.8-27B store
  used 262,144 and dropped nothing). 161,864 of 163,016 rows are collected.
  All 1,152 dropped rows are long-context families, some nearly entirely:
  `longcontext_lc5flip` 322 of 331, `longcontext_lc3flip` 210 of 314,
  `longcontext_v3_bookqa` 156 of 218, `longcontext_lcflip` 144 of 211,
  `longcontext_v5c3_intrace` 132 of 132, `longcontext_v2` 113 of 157,
  `longcontext_v5c5_novels` 47 of 47, `longcontext_v4_offpolicy_concat:*`
  28 of 43. Results on these families are not comparable across the two
  stores.
- **Rendering differs on 1,642 rows.** The Qwen3.5-27B template renders an
  empty system message as `<|im_start|>system\n<|im_end|>\n`, the
  Qwen3.8-27B template omits it: in `liarsbench` (888 rows) and
  `liarsbench_softtrigger_raw` (754) `n_tokens` is 5 higher here. The
  difference is at the start of the input; the read token and `window` are
  unaffected. Every other row has the same token count in both stores
  (token ids compared on a sample of 50 rows: identical).
- **The split is a redraw**, not the paper's own (FIBSv1 does not mark it);
  the published Qwen3.5-27B probes were trained on most rows, `val`
  included.

## Loading

With the repository (`fibs_probes` reads this store and the Qwen3.8-27B
one):

```python
from pathlib import Path
from fibs_probes import load_rows, load_read_token

root = Path("read-token")                                   # a local copy of the prefix
rows = [r for r in load_rows(root) if r["source"] not in ("among_us", "werewolf")]   # any slice
X = load_read_token(root, rows)                             # (N, 6, 5120) bf16; layers 19, 27, 35, 43, 51, 58
```

Without it, the Qwen3.8-27B README's loading snippet works unchanged: skip
rows with `dropped` true, they are in no shard.

The paper's training recipe (per-family class balance, per-layer RMS input
scaling, AdamW, early stopping on `val`) is `fibs_probes.train`.

## Trained probes

In the repository's `probes/` folder (see the Qwen3.8-27B README,
"Trained probes", for the setup):

- `qwen3.5-27b_layers19-58_fibs_linear_no-social-deduction`: the linear
  ensemble trained on this store without the social-deduction families
  (`among_us`, `werewolf`), with its own hold-out over the remaining rows
  (`experiments/2026-10-14-fibs-linear-no-sd/`): validation AUROC 0.993;
  held-out `among_us` 0.910, `werewolf` 0.871.
- The paper's own Qwen3.5-27B probes are on the Hugging Face Hub
  (`AlignmentResearch/probe-inference-weights`, `qwen3.5-27b/{linear,mlp,efc,axial}`).
