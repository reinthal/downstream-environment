# fibs_probes

Train fibs probes (Hollinsworth et al. 2026, arXiv:2610.12445) on stored
FIBSv1 activations, in the paper's `probe_inference` directory format
(vendored at `vendor/caught-in-the-act-probes`, put on `sys.path` by
`import fibs_probes`). Registered in `pyproject.toml`; import from any
experiment run with `uv run --no-sync`.

| module | contents |
|---|---|
| `data` | `load_rows` (`rows.jsonl`, collected rows in `i` order, with `label` and the store's `split`), `load_read_token` → `(N, layers, d)` bf16 in row order, `split_validation` (redraw the class-balanced, `pair_id`-disjoint hold-out over a subset, paper A.6), `balance_within_family` (per-`source` class balance of the training rows) |
| `train` | `train_layer_ensemble` (per-layer linear / MLP ensemble, paper recipe), `save_probe` (writes the directory `probe_inference.load_probe` reads), `auroc` |

## Stores

One layout (`rows.jsonl` + `acts_{k}.pt`, see `data.py`) for every model;
`i` is the same FIBSv1 row in all of them, and `split` is the same hold-out.

| model | where | rows collected | length cap |
|---|---|---|---|
| Qwen3.5-27B (rev fc05daec) | `experiments/2026-10-13-fibs-acts/acts_fibs/read-token/` (local, gitignored) | 161,864 | 32,768 |
| Qwen3.8-27B (rev 1d4bf0f2) | `fibs/activations/Qwen3.8-27B/read-token/` in the public-datasets bucket (download: its README) | 163,016 | 262,144 |

Both: read token (before the answer's `<|im_end|>`), layers 19, 27, 35, 43,
51, 58 in that order, bf16, d = 5120.

```python
from fibs_probes import load_rows, load_read_token
rows = [r for r in load_rows(root) if r["source"] != "werewolf"]   # any slice
x = load_read_token(root, rows)                                     # (N, 6, 5120)
```

Example: `experiments/2026-10-14-fibs-linear-no-sd/run.py`.

Copied from `experiments/2026-10-13-fibs-basharena/train.py` and
`modal_app.py::prepare_fibs`, with these differences:

- per-layer archs only: the stored activations hold one token per row, so
  EFC and axial (whole follow-up window, Muon) stay in that experiment;
- the layer rule uses the mean training loss over the last 5 steps *before
  the selected checkpoint* (the original took the last 5 steps of training,
  which differ when early stopping picks an earlier checkpoint);
- the hold-out is drawn over the rows the caller passes (e.g. after
  excluding families), not over all FIBSv1 rows.
