# fibs_probes

Train fibs probes (Hollinsworth et al. 2026, arXiv:2610.12445) on stored
FIBSv1 activations, in the paper's `probe_inference` directory format
(vendored at `vendor/caught-in-the-act-probes`, put on `sys.path` by
`import fibs_probes`). Registered in `pyproject.toml`; import from any
experiment run with `uv run --no-sync`.

| module | contents |
|---|---|
| `data` | `load_rows` (rows.parquet, kept rows in `act_index` order, label = follow-up deceptive), `split_validation` (class-balanced hold-out, no shared `pair_id`, paper A.6), `balance_within_family` (per-`source` class balance of the training rows), `load_layer_acts` → `(N, L, d)` |
| `train` | `train_layer_ensemble` (per-layer linear / MLP ensemble, paper recipe), `save_probe` (writes the directory `probe_inference.load_probe` reads), `auroc` |

Input layout: `experiments/2026-10-13-fibs-acts/acts_fibs/` (Qwen3.5-27B,
read token, layers 19/27/35/43/51/58). Example: `experiments/2026-10-14-fibs-linear-no-sd/run.py`.

Copied from `experiments/2026-10-13-fibs-basharena/train.py` and
`modal_app.py::prepare_fibs`, with these differences:

- per-layer archs only: the stored activations hold one token per row, so
  EFC and axial (whole follow-up window, Muon) stay in that experiment;
- the layer rule uses the mean training loss over the last 5 steps *before
  the selected checkpoint* (the original took the last 5 steps of training,
  which differ when early stopping picks an earlier checkpoint);
- the hold-out is drawn over the rows the caller passes (e.g. after
  excluding families), not over all FIBSv1 rows.
