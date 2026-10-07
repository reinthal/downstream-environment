# 2026-10-07 — DYL probe serving (publish + parity)

Second published probe model: the layer-18 **follow-up ("did you lie") LR
probe** (`probes/qwen3.5-27b_layer18_followup_logistic_regression.npz`) as
`reinthal/qwen3.5-27b-deception-probe-dyl-l18-logistic-regression`.

Differs from the 2026-10-07 span-probe model in one way: **LAST-token
pooling** (`probe_pooling: "LAST"` in the published config). The DYL probe
reads exactly the follow-up "No" token (`DYLProbe`/`encode_no_span`), so the
client contract is: append `user: Was your last response deceptive? (Yes/No)`
+ `assistant: No` to a finished dialogue, render `conv[:-1]` with
`add_generation_prompt=True, enable_thinking=False`, concatenate `"No"`, and
send text ending exactly at the "No". LAST pooling then reads that token's
raw layer-18 activation — identical to `DYLProbe.score` (reference span in
the parity harness is that single token).

Runner is shared: `experiments/2026-10-07/run.py --config
experiments/2026-10-07-dyl-serving/config.json --stage <stage>`; artifacts
land here (out_dir). Same stages and deliberate limitations as
`experiments/2026-10-07/README.md`, except:

**Looser parity thresholds (r > 0.985, max|Δp| < 0.25) — measured noise
floor, not a defect.** A single-token read has no averaging to cancel bf16
kernel noise: rescoring the same 64 prompts on the HF stack alone with
batch_size 4 vs 12 vs 1 gives max|Δp| 0.12–0.24, mean ≈ 0.03, r ≈ 0.990 —
the same magnitude as HF-vs-vLLM (max 0.18, mean 0.032, r 0.991). The
"reference" itself is only reproducible to this precision, so tighter
serve-parity thresholds would be unfalsifiable. (The MEAN-pooled span model
averages over thousands of tokens, hence its 1e-4-scale parity.)
