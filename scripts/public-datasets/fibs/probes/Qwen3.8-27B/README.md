# fibs probes: Qwen3.8-27B

These are the probes trained on `fibs/activations/Qwen3.8-27B/read-token/`
(train split, class-balanced within each family). Hyperparameters are from
Table 2 of arXiv:2610.12445. The code is
`experiments/2026-10-13-fibs-basharena/train.py`.

| path | contents |
|---|---|
| `linear/`, `mlp/` | `probe_inference` directory format: `probe_metadata.json` and `layer_{19,27,35,43,51,58}/{config.json,model.pt}`. Load with `probe_inference.load_probe(dir)`. `score()` returns the mean of the per-layer sigmoids over the used layers, read at the token before the answer's `<|im_end|>`. |
| `linear_train.json`, `mlp_train.json` | Training summary: used layers, best step, validation AUROC (linear 0.987, MLP 0.996 on the 5,000 `val` rows). |

`load_probe` takes a local directory, so download the files first. The
bucket cannot be listed anonymously, so the names are spelled out here:

```bash
B=https://public-datasets.s3.eu-central-003.backblazeb2.com/fibs/probes/Qwen3.8-27B
for arch in linear mlp; do
  curl -sSf --create-dirs -o probes/$arch/probe_metadata.json $B/$arch/probe_metadata.json
  for L in 19 27 35 43 51 58; do for f in config.json model.pt; do
    curl -sSf --create-dirs -o probes/$arch/layer_$L/$f $B/$arch/layer_$L/$f
  done; done
done
```

```python
from probe_inference import load_probe
probe = load_probe("probes/linear")          # the local directory downloaded above
```

`probe_inference` is the loader from the paper's probe release:
`pip install git+https://github.com/AlignmentResearch/caught-in-the-act-probes@3941138776ea8e9784769de59298b006d2b1a3ac`
(the commit vendored in the repository).
