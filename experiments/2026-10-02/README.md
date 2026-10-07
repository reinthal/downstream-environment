# 2026-10-02 — Manual audit of the LR probe's strongest hits

Qualitative follow-up to 2026-09-29: the LR probe is near chance on natural
play (AUROC 0.59 full-turn), so what *is* it firing on when it fires hard?
This packages the probe's 10 strongest base-arm hits for human reading in
`inspect view` — no model is run.

## Selection (`config.json`)

From `scores_base.csv` (2026-09-22 base arm, non-silent announcements),
ranked by the column in `config.json` — currently **`lr_pub`** (the probe on
the public announcement text alone):

- **5 TP** — highest-scoring **Parsnip** announcements
- **5 FP** — highest-scoring **Carrot** announcements
- at most one flagged turn per game per class

No suspicion grade is shown: the 2026-09-25 scale is calibrated on the
`full` span, so its cutpoints do not apply to `lr_pub` (the task shows
grades automatically when the selection span matches the scale span, i.e.
when `column` is set back to `lr_full`). Instead each sample reports the
span-matched **alpaca 1%-FPR threshold** from `alpaca_thresholds.json`
(`compute_thresholds.py`, p99 of the LR probe on the 2000 cached 2026-09-24
alpaca control rollouts per span: full 0.168 — identical to the scale's top
cutpoint — and pub 0.228). Each Inspect sample shows the
flagged announcement (the scored span), the speaker's private reasoning
(context only — not scored by `lr_pub`), then the full public transcript
with the turn marked `>>>`; probe scores, role, and TP/FP label are in the
header and sample metadata.

## Run / view

```sh
uv run --no-sync inspect eval experiments/2026-10-02/audit_task.py@lr_probe_audit \
    --model mockllm/model --log-dir experiments/2026-10-02/inspect_logs
uv run --no-sync inspect view --log-dir experiments/2026-10-02/inspect_logs
```

The solver is a no-op; `mockllm/model` only satisfies Inspect's requirement
that an eval names a model. The committed log under `inspect_logs/` is the
deliverable.

## Deliberate limitations

- TP/FP labels and roles are visible in the sample — this is a sighted
  audit, not a blinded one. Blind it by dropping the header/metadata if the
  researcher wants to re-score by hand first.
- `config.json` is a plain selection record, not an `ExperimentConfig`:
  nothing is generated or fitted here.
- Top-of-ranking only; says nothing about behavior near the threshold.
