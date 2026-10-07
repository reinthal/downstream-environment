# Probe serving

Turn a trained linear probe into a servable HF model: the base decoder
truncated at the probe layer plus one fp32 `score` Linear holding the probe
(standardizer folded in). vLLM's classify task then returns `p(deceptive)`
for any prompt instead of generated text.

**Scoring contract**: the score is the probe applied to the raw
(pre-final-norm) block-`layer` residual stream, **mean-pooled over ALL tokens
of the submitted prompt**. Span selection is the client's job — send exactly
the text you want scored (normally a finished conversation rendered with the
bundled chat template, `add_generation_prompt=False`). This deliberately
differs from `LMProbe.score`, which pools only the final assistant turn.

## Pieces

- `serving/config.py` — `ServingConfig(ExperimentConfig)`: publish + parity
  parameters; imports in both venvs.
- `serving/publish.py` (`.venv`) — builds the checkpoint (streams only the
  needed tensors out of the base model's shards, ~17 GB for 27B/layer 16,
  never a full model in RAM) and uploads it. Weight folding:
  `score.weight = w/sd`, `score.bias = b − (mu/sd)·w`, so
  `sigmoid(score(x)) == probe.predict_proba(x)` exactly.
- `serving/vllm_plugin/` (`.venv-vllm`) — pip-installable, zero-dependency
  vLLM plugin registering `Qwen3_5ProbeForSequenceClassification`. Needed
  because (a) vLLM applies the final RMSNorm (fused with the last residual
  add) before pooling — the plugin's `ResidualAddNorm` keeps the add and
  skips the normalization, matching `load_truncated_decoder`'s Identity; and
  (b) vLLM has no registered text-only Qwen3.5 architecture. It must be a
  real entry-point install — registry registration from the driver does not
  reach engine-core/worker processes.

## Workflow

```bash
# 1. build (CPU; writes cfg.publish_dir, gitignored)
uv run --no-sync python -m serving.publish --config experiments/<date>/config.json --local-only

# 2. one-time plugin install into the vLLM env
uv pip install --python .venv-vllm/bin/python -e serving/vllm_plugin

# 3. offline scoring
#    LLM(model=<publish_dir or repo>, runner="pooling").classify([...])
#    -> .outputs.probs[0] is p(deceptive)

# 4. serve
CUDA_VISIBLE_DEVICES=<free> PATH=$PWD/.venv-vllm/bin:$PATH \
    .venv-vllm/bin/vllm serve <repo-or-dir> --runner pooling --max-model-len 8192
curl -s localhost:8000/classify -H 'Content-Type: application/json' \
    -d '{"input": ["<rendered conversation>"]}'

# 5. upload (creates the repo private by default)
uv run --no-sync python -m serving.publish --config experiments/<date>/config.json
```

Parity harness (reference scores from the HF-side raw-activation path vs the
served model): `experiments/2026-10-07/run.py` — stages
`publish | score_hf | score_vllm | compare | push | serve_smoke`.

## Gotchas

- The published config is the base model's `text_config` with
  `num_hidden_layers = layer+1`, `layer_types` sliced, `num_labels = 1`,
  `architectures = ["Qwen3_5ProbeForSequenceClassification"]`, and a
  `probe_meta` provenance block (probe kind/sha256/source).
- The checkpoint has no `lm_head` and unnormalized final hidden states —
  only the classify path is meaningful; it cannot generate.
- vLLM's MEAN pooling asserts no partial prefill: if long prompts trip it,
  disable chunked prefill at engine init.
- `remap_weight_name` raises on tensor names it doesn't know — extend it
  deliberately when publishing a new model family.
