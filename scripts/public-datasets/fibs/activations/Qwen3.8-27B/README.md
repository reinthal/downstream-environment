# FIBSv1 activations: Qwen3.8-27B

Residual-stream activations of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`) on
[`AlignmentResearch/fibs-v1`](https://huggingface.co/datasets/AlignmentResearch/fibs-v1)
(revision `65dccf12934620bfe01db665ea3d7d0e3c9f1355`). They use the
follow-up read of Hollinsworth et al. 2026
([arXiv:2610.12445](https://arxiv.org/abs/2610.12445)). Use them to retrain
or evaluate probes on any slice of FIBSv1 (by family, quadrant or label)
without running the model.

## How the activations were taken

The code is in `experiments/2026-10-13-fibs-basharena/` (`modal_app.py`,
`acts.py`).

- **Rendering.** Each row's conversation plus its completion is rendered
  with the model's own chat template, with `enable_thinking=False` and
  `preserve_thinking=False`. No reasoning is in the input, and the answer
  follows an empty think block, as in the paper's Qwen3.5 read. Tool-call
  arguments are parsed from their JSON strings.
- **Read window.** The window runs from the first token of the final user
  turn (the follow-up question) through the answer's `<|im_end|>`.
  `followup_start` and `completion_start` in `jobs_*.pt` index the rendered
  sequence.
- **Layers.** 19, 27, 35, 43, 51, 58, always in that order on the layer
  axis. Layer k is decoder block k's output (HF `hidden_states[k + 1]`),
  taken from a decoder truncated at layer 58 with the final norm
  neutralised. bf16; d = 5120.
- **Read token.** For linear/MLP probes this is the token before the
  answer's `<|im_end|>` (`probe_inference`'s
  `second-last-token-generation`).
- **Labels.** `label` is 1 when the completion is a falsehood, meaning the
  row's quadrant ends in `_deceptive`.

## `read-token/`: every row, read token only (10.8 GB)

163,016 rows: all four quadrant parquets, none dropped for length (longest
window 23,215 tokens; median 180). This is the training set of the probes
`probes/qwen3.8-27b_layers19-58_fibs_{linear,mlp}` in the repository.

| file | contents |
|---|---|
| `rows.jsonl` | One line per row; see the row schema below. |
| `acts_{k}.pt` | Shard k (500 rows; 327 shards). `{"i": list[int], "token": Tensor(n, 6, 5120) bf16, "acts": [], "prompt_mask": [], "completion_mask": [], "window_cap": 0}`. `token[j]` is row `i[j]`'s read-token activation at the 6 layers. The window lists are empty in this collection. |
| `jobs_{k}.pt` | The rendered input of shard k. A list of `{"i", "ids": list[int], "followup_start", "completion_start", "n_tokens"}` in the same order as `acts_{k}.pt`. |

Row schema (`rows.jsonl`):

| field | meaning |
|---|---|
| `i` | Row id, the key into `acts_*.pt` and `jobs_*.pt`. In `read-token/`, rows are numbered in the order of the parquets `honest_honest`, `honest_deceptive`, `deceptive_honest`, `deceptive_deceptive`, each read in full. So `i` minus the quadrant's offset is the row's position in its parquet. The offsets are 0, 43,941, 84,985 and 122,552. |
| `label` | 1 = falsehood completion, 0 = honest completion. |
| `quadrant` | FIBSv1 quadrant (`honest_honest`: 43,941 rows; `honest_deceptive`: 41,044; `deceptive_honest`: 37,567; `deceptive_deceptive`: 40,464). |
| `source` | FIBSv1 family, for example `instructed_lie_neutral`, `mask_known_facts` or `among_us`. There are 4,370 values: 49 families plus 4,321 `persona_prefill_*` scenario ids. |
| `pair_id` | FIBSv1 pair id (empty for unpaired rows). |
| `_v32_source_content_md5` | FIBSv1's content hash; join key back to the parquet. |
| `split` | `train` (158,016) or `val` (5,000). `val` is class-balanced and shares no `pair_id` with `train`; it is the paper's A.6 hold-out, seed 0. |
| `n_tokens`, `window` | Rendered length and read-window length, in tokens. |
| `dropped` | True if the row exceeded 262,144 tokens and was not collected. False for every row here. |

## `windows-pilot/`: 2,000 rows, whole read window (36.9 GB)

This is a seeded balanced subset: 500 rows per quadrant, 200 of them
validation. It was collected in the experiment's pilot. Use it for
window-reading probes such as EFC and axial, which need the activation at
every window token.

The files have the same names as in `read-token/`, with these differences:

- `acts_{k}.pt` also fills `acts`: one `Tensor(6, W, 5120)` bf16 per row.
  It also fills `prompt_mask` and `completion_mask`: one `Tensor(W,)` bool
  each, marking the follow-up question and the answer (through
  `<|im_end|>`).
- `token` is filled as in `read-token/`.
- Windows are capped at 1,024 tokens (`window_cap`). A longer window
  keeps its last 1,024 tokens, with the masks cut to match; this
  affects 60 of the first shard's 500 rows, and the longest window is
  11,382 tokens. Otherwise W equals the row's `window`.
- `i` is the row's index in the sampled subset, not its parquet position.
  Join to FIBSv1 on `_v32_source_content_md5`.

## Loading a slice

```python
import json, torch
from pathlib import Path

root = Path("read-token")                       # a local copy of the prefix
rows = {r["i"]: r for r in map(json.loads, open(root / "rows.jsonl"))}
want = lambda r: r["source"].startswith("instructed_")   # any slice
X, y, split = [], [], []
for shard in sorted(root.glob("acts_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
    blob = torch.load(shard, weights_only=True)
    for j, i in enumerate(blob["i"]):
        r = rows[i]
        if want(r):
            X.append(blob["token"][j]); y.append(r["label"]); split.append(r["split"])
X = torch.stack(X)                               # (N, 6, 5120) bf16; layer axis = 19, 27, 35, 43, 51, 58
```

The paper's training recipe uses per-family class balance, per-layer RMS
input scaling, AdamW and early stopping on `val`. Its implementation is
`experiments/2026-10-13-fibs-basharena/train.py`.

## Trained probes

The trained probes live in the repository
([Cadenza-Labs/downstream-environment](https://github.com/Cadenza-Labs/downstream-environment)), not in this bucket:

```bash
git clone https://github.com/Cadenza-Labs/downstream-environment && cd downstream-environment
git lfs pull --include "experiments/2026-10-13-fibs-basharena/probes_main/**"
git submodule update --init vendor/caught-in-the-act-probes      # the probe_inference loader
pip install ./vendor/caught-in-the-act-probes
```

```python
from probe_inference import load_probe
probe = load_probe("probes/qwen3.8-27b_layers19-58_fibs_linear")    # or ..._fibs_mlp
```

`probes/` holds the repo's probe symlinks. Its README indexes them:
`qwen3.8-27b_layers19-58_fibs_{linear,mlp}` are the linear and MLP
ensembles trained on `read-token/` with the paper's recipe (fibs validation
AUROC 0.987 / 0.996). `load_probe` takes the local directory. `score()`
returns the mean of the per-layer sigmoids, read at the token before the
answer's `<|im_end|>`.
