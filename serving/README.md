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

---

# Qwen3.8-27B generation server

`serving/serve_qwen38.py` (`.venv-serve`, built by `requirements/setup.sh serve`)
serves `Qwen/Qwen3.8-27B` (revision pinned) as an OpenAI-compatible API on
**port 8001** of this box, for agentic / eval clients that used to hit the
RunPod endpoint (`iac/`). All tunables are fields of `ServeConfig` in the
script; override with `--<field> <value>`, inspect with `print` /
`--dump-config`.

```bash
.venv-serve/bin/python serving/serve_qwen38.py              # serve (logs: serving/logs/serve_8001.log)
.venv-serve/bin/python serving/serve_qwen38.py print        # the vllm serve command it would run
.venv-serve/bin/python serving/serve_qwen38.py bench --concurrency 64        # load test a running server
curl localhost:8001/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B","messages":[{"role":"user","content":"ping"}],"max_tokens":50}'
```

Thinking is on by default (reasoning lands in `message.reasoning`, parsed by
`--reasoning-parser qwen3`); turn it off or down per request with
`"chat_template_kwargs": {"enable_thinking": false}` or
`{"reasoning_effort": "low"}`. Tool calls are parsed by `qwen3_xml` (the
chat template's `<function=..><parameter=..>` format); `tool_choice: "auto"`
works. Qwen's recommended sampling: thinking `temperature 1.0, top_p 0.95,
top_k 20`; instruct `0.7 / 0.8 / 20, presence_penalty 1.5`.

## Layout (4 GPUs, shared box)

- **GPUs 1,2,6,7 only** (`gpus` field). The box is shared; four is the
  allowance. Those four are the two PCIe-sibling pairs (`PHB` in
  `nvidia-smi topo -m`: 1-2 and 6-7); every other pair crosses the
  inter-socket link. vLLM hands consecutive visible devices to a replica, so
  the order of `gpus` decides the pairs.
- **2 replicas x tensor parallel 2 on one port** (`--data-parallel-size 2
  --tensor-parallel-size 2`, vLLM's internal load balancer, 2 API-server
  processes). bf16 weights are 55.6 GB, so one 46 GB A40 cannot hold them
  and TP=2 is the minimum; with PCIe-only GPUs two independent replicas
  beat one TP=4 engine on throughput (all-reduce per layer vs none).
- Memory per GPU: 27.8 GB weights + ~4 GB activations/CUDA graphs; the rest
  is KV + GDN state. KV is cheap on this architecture (16 of 64 layers are
  full attention, 4 KV heads): at the default `gpu_memory_utilization 0.88`
  each replica holds ~300k KV tokens (1.2 full-length requests, hundreds of
  normal ones); 0.92 gave 427k but OOMs under load with MTP (gotchas).
- `--language-model-only`: text only. The checkpoint is multimodal; the
  vision encoder would otherwise be loaded and profiled at startup.
- Startup ~4 min (weights 1 min, torch.compile 35 s, CUDA-graph capture per
  replica, warm-up). The compile cache (`~/.cache/vllm/torch_compile_cache`)
  makes later starts faster.

## Kernels (verified in the startup log, 2026-10-10)

- Gated DeltaNet (48 of 64 layers): `Using Triton/FLA GDN prefill kernel`
  — vLLM 0.31's vendored flash-linear-attention Triton ops
  (`vllm.third_party.flash_linear_attention`) plus its compiled
  `causal_conv1d` custom op; no external `fla` / `causal-conv1d` package
  and no PyTorch reference path. Decode: `Falling back to the Triton GDN
  decode path: torch.ops._C.fused_gdn_decode_post_conv_mtp is not built`
  — the fused CUDA decode op is not in the cu129 wheel, the Triton decode
  kernel is used (this is the normal, fast path; the fused op is an
  optimisation on top of it).
- Full attention (16 layers): `FLASH_ATTN` backend, FlashAttention **2**
  (A40 is sm86; FA3 is Hopper-only). FlashInfer is installed but has no
  nvcc here to JIT; `VLLM_USE_FLASHINFER_SAMPLER=0` is set by the script.
- All-reduce: vLLM custom all-reduce + pyNCCL over PCIe. (The NCCL P2P hang
  seen on RunPod did not occur here.)
- CUDA graphs + torch.compile on (not eager) — `FULL_AND_PIECEWISE`.

## Throughput (measured 2026-10-10, `bench` subcommand, 64 concurrent)

All runs: 4x A40, bf16, 256 requests, 64 in flight, `--ignore-eos`.
"random" = 2048 random input tokens / 512 output (prefill-heavy, and MTP
drafts cannot predict random text); "ShareGPT" = real conversations
(~224 in / ~225 out tokens on average, decode-heavy). Output tok/s is the
total across the server.

| layout | MTP | random: out tok/s (TPOT) | ShareGPT: out tok/s (TPOT) | notes |
|---|---|---|---|---|
| 2 x TP2 | off | 632 (80 ms) | 738 (69 ms) | util 0.92, 256 seqs/replica |
| **2 x TP2** | **3** | 567 (73 ms) | **1046 (51 ms)** | **default**: util 0.88, 128 seqs/replica |
| 1 x TP4 | 3 | 257 (208 ms) | 466 (153 ms) | util 0.88, 256 seqs |

- MTP acceptance (from `/metrics`, `vllm:spec_decode_*`): 0.53-0.57 of
  draft tokens accepted on ShareGPT, 1.6 extra tokens per step on average,
  hence the 42% gain on real text. On random-token prompts the drafter
  mostly misses and MTP costs ~10%: keep it on for real workloads, pass
  `--mtp_tokens 0` for synthetic ones.
- TP4 loses >2x to two TP2 replicas: every layer's all-reduce crosses PCIe
  and the inter-socket link, and the four GPUs serve one batch instead of
  two. The KV cache it gains (1.19M tokens) is not the constraint.
- TTFT at 64 in flight with 2k-token prompts is ~10-14 s: prefill is the
  bottleneck in that regime (chunked at 8192 tokens per step per replica).
- The RunPod endpoint on the same GPU type measured 780 tok/s at 64
  concurrent (`iac/README.md`) with TP4 and no MTP.

## Gotchas

- **`mamba_cache_mode "all"` is not supported for qwen3_5 in vLLM 0.31.0**
  (`falling back to 'align' mode` warning), so the config says `align`.
  Prefix caching on this hybrid works in 784-token blocks (attention block
  size is raised to match the GDN page size): prompts shorter than that get
  no cache hits, long agent conversations do.
- **`gpu_memory_utilization 0.92` OOMs under load with MTP** (CUDA OOM in
  `execute_model` at 64 concurrent requests, ~3.5 GB over budget: vLLM's
  profiling run does not cover the speculative sampler's peak). Without MTP
  0.92 survived the same load. The default is set with margin.
- `vllm bench serve` takes no `--revision`; the tokenizer comes from the
  HF cache, which only holds the pinned snapshot.
- `pkill -f "vllm serve"` also matches any shell whose command line
  contains that string (it killed the shell that was running it). Use
  `pkill -f "vllm/entrypoints"` or the pid from the log instead.
- torchcodec from PyPI links `libnvrtc.so.13` and kills `vllm serve` at
  import on this CUDA-12 driver; `.venv-serve` pins `torchcodec==0.17.0+cpu`
  (vLLM only uses it for audio, which `--language-model-only` never touches).
