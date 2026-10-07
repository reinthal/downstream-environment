# 2026-10-07 — Probe serving via vLLM (publish truncated model + probe head)

Build a servable HF model out of the layer-16 logistic-regression deception
probe: Qwen3.5-27B truncated to decoder blocks 0..16 plus one fp32 `score`
Linear holding the probe with its standardizer folded in
(`score.weight = w/sd`, `score.bias = b − (mu/sd)·w`). vLLM's classify task
returns `p(deceptive) = sigmoid(score(mean(h)))` per prompt. Infrastructure
lives in `serving/` (publish script + vLLM plugin); this folder is the build
+ parity harness for the first published model:
`reinthal/qwen3.5-27b-deception-probe-l16-logistic-regression`.

## Design

- **Scoring contract**: probe over the raw (pre-final-norm) block-16 residual
  stream, mean-pooled over ALL tokens of the submitted prompt. Span selection
  is the client's job (researcher decision) — this deliberately differs from
  `LMProbe.score`, which pools only the final assistant turn.
- **Why a plugin**: vLLM applies the final RMSNorm (fused with the last
  residual add) before pooling, and registers no text-only Qwen3.5 arch. The
  plugin's `ResidualAddNorm` keeps the residual add and skips normalization,
  so pooled states equal the activations the probe was trained on.
- **Parity check**: reference scores come from the established HF-side path
  (`load_truncated_decoder` + `collect_activations`) with a whole-sequence
  span `(0, len(ids))`, probe `predict_proba` on top — the ground truth for
  the serving contract. vLLM scores the exact same token ids
  (`TokensPrompt`), removing tokenizer drift from the comparison; a 5-prompt
  string-path check guards the server-side tokenization separately.
- Conversations: first `n_parity=64` rows of
  `experiments/2026-09-22/transcripts_base.jsonl` whose rendered conversation
  fits `max_len=6144` untruncated.

## Run

See stage list in `run.py` (publish → score_hf → score_vllm → compare →
push → serve_smoke); venv per stage in the docstring.

## Deliberate limitations

- Pass thresholds (`r > 0.995` on logits, `max|Δp| < 0.05`) are provisional:
  HF and vLLM are two different bf16 kernel stacks (incl. GDN linear
  attention), so bit-reproducibility is not expected; tighten after first
  numbers if warranted.
- `enforce_eager=True` everywhere for the first pass; compile/CUDA-graph
  speedups unexplored.
- Parity runs on TP=1 only.
- The published model scores all submitted tokens — scores are NOT comparable
  to the 2026-09-22 per-message `LMProbe` span scores.

## Artifacts

- `published/<repo-name>/` — built checkpoint (gitignored, regenerable)
- `parity_tokens.jsonl` — the exact token ids + rendered text scored
- `scores_hf.json`, `scores_vllm.json` — logits + probs per side
- `parity_report.json` — Pearson r, max/mean |Δp|, thresholds
