# fibs probes: Qwen3.8-27B

These are the probes trained on `fibs/activations/Qwen3.8-27B/read-token/`
(train split, class-balanced within each family). Hyperparameters are from
Table 2 of arXiv:2610.12445. The code is
`experiments/2026-10-13-fibs-basharena/train.py`.

| path | contents |
|---|---|
| `linear/`, `mlp/` | `probe_inference` directory format: `probe_metadata.json` and `layer_{19,27,35,43,51,58}/{config.json,model.pt}`. Load with `probe_inference.load_probe(dir)`. `score()` returns the mean of the per-layer sigmoids over the used layers, read at the token before the answer's `<|im_end|>`. |
| `linear_train.json`, `mlp_train.json` | Training summary: used layers, best step, validation AUROC (linear 0.987, MLP 0.996 on the 5,000 `val` rows). |

`probe_inference` is the loader from the paper's probe release
(`vendor/caught-in-the-act-probes` in the repository).
