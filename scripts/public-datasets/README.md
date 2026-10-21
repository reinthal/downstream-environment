# public-datasets

Datasets released by the deception-probe project
([Cadenza-Labs/downstream-environment](https://github.com/Cadenza-Labs/downstream-environment)).
Every object is public; download over HTTPS:

```
https://public-datasets.s3.eu-central-003.backblazeb2.com/<key>
```

Anonymous listing is disabled, so you cannot browse the bucket. Each
prefix's README names every file: shard counts are given, and per-trajectory
files are named by the `uid`s in the transcripts. For example, to fetch the
read-token activations:

```bash
B=https://public-datasets.s3.eu-central-003.backblazeb2.com/fibs/activations/Qwen3.8-27B/read-token
curl -O $B/rows.jsonl
for k in $(seq 0 326); do curl -O $B/acts_$k.pt; done
```

With credentials, any S3 client works (endpoint
`https://s3.eu-central-003.backblazeb2.com`, bucket `public-datasets`).

This index covers only the prefixes below. Other objects in the bucket are
not part of this collection.

| prefix | contents | size |
|---|---|---|
| [`fibs/activations/Qwen3.8-27B/`](fibs/activations/Qwen3.8-27B/README.md) | Residual-stream activations of Qwen3.8-27B on every FIBSv1 row, read with the follow-up question at 6 layers. `read-token/` holds 163,016 rows at the probe read token. `windows-pilot/` holds 2,000 rows with the full read window. Enough to retrain linear/MLP probes on any slice of FIBSv1 without a GPU forward pass. | 10.8 GB + 36.9 GB |
| [`fibs/activations/Qwen3.5-27B/`](fibs/activations/Qwen3.5-27B/README.md) | The same read for Qwen3.5-27B, same layout, row ids and `split`. `read-token/` only: 161,864 of 163,016 rows (rows over 32,768 tokens are not collected, all from long-context families). Supports linear/MLP probes; no window activations, so not EFC or axial. | 10.0 GB |
| [`downstream-environments/bash-arena/`](downstream-environments/bash-arena/README.md) | BashArena rollouts of Qwen3.8-27B (88 honest / 94 attack trajectories, 47 tasks). Includes per-step side-task labels from a Claude Haiku 4.5 judge, fibs probe scores for every step, and the step activations they were computed from. `preonset-ablation/` re-reads the same steps under seven system-prompt conditions. | 17.4 GB + 68.9 GB + 45 MB |
| [`downstream-environments/carrot-parsnip/`](downstream-environments/carrot-parsnip/README.md) | Carrot-Parsnip social-deduction games of Qwen3.5-27B (3- and 5-player, plus all-Carrot calibration arms; 100 games each, 4,800 announcements) and 2,000 alpaca replies as a benign control. Includes oracle-judge honesty grades, fibs probe scores, the follow-up-window activations they were computed from, and pooled reads for the Apollo and UK AISI probes. | 20.0 GB + 0.2 GB |

Each prefix's README gives the schema, the provenance (code commit, model
revision, dataset revision) and a loading snippet. The probes trained on
these activations are in the repository's `probes/` folder, not in the
bucket; the READMEs show how to load them.

The code that produced the data and the analyses that use it are in
[`experiments/2026-10-13-fibs-basharena/`](https://github.com/Cadenza-Labs/downstream-environment/tree/main/experiments/2026-10-13-fibs-basharena)
(Qwen3.8-27B BashArena),
[`experiments/2026-10-14-preonset-ablation/`](https://github.com/Cadenza-Labs/downstream-environment/tree/main/experiments/2026-10-14-preonset-ablation)
(its pre-onset ablation),
[`experiments/2026-10-13-fibs-acts/`](https://github.com/Cadenza-Labs/downstream-environment/tree/main/experiments/2026-10-13-fibs-acts)
(Qwen3.5-27B FIBSv1) and
[`experiments/2026-10-12-fibs-probes/`](https://github.com/Cadenza-Labs/downstream-environment/tree/main/experiments/2026-10-12-fibs-probes)
(Qwen3.5-27B Carrot-Parsnip). `fibs_probes` in the repository loads either activation store.
`scripts/push_b2.py` in that repository uploads the data, and
`scripts/public-datasets/` holds these READMEs.

Files are stored uncompressed. `.pt` files are `torch.save` pickles of
tensors, lists and dicts only, so `torch.load(path, weights_only=True)`
reads them. Activations are bf16.
