# Hand-off — Turnip experiment (2026-10-15), paused before the main run

**State:** preregistered (`PREREGISTRATION.md`, commit 418d972), pipeline
built, **no non-pilot data**. Aborted by the researcher on 2026-10-15 ~20:40
UTC so the games don't run overnight on Modal GPUs; to be resumed on the box's
A40s. The last configuration (Deviations 3 + 4: answer effort low, Turnip cap
32K) has **not yet completed a pilot** — that pilot was stopped while the Qwen
endpoint was cold-starting.

## What exists

| piece | where | status |
|---|---|---|
| Turnip role + public interrogation + detector hook | submodule `environments/carrot_parsnip`, branch `turnip-role` (2049e85, 08f92f2) | engine tests pass (23); exercised by 11 pilot games |
| game configs | `game_interrog.yaml`, `game_probe.yaml` (differ only in `interrogation.detector`: `silent` / `shown`) | current = Deviations 1-4 |
| run config | `config.json` (`common.TurnipRunConfig`) | Modal endpoint URLs in `providers`, probe read, seeds, analysis |
| live probe | `modal_probe.py` → Modal app `turnip-probe` (workspace `cadenza`, H100) | deployed; works (pilot scores below) |
| play / extract / calibrate / analyze | `play_games.py`, `extract.py`, `calibrate.py`, `analyze.py` | play/extract/calibrate run in pilots; `analyze.py` not run yet |

Engine additions (all in the submodule branch): `Role.TURNIP` (Carrot team,
identity public in every state view); `GameConfig.turnip` (count 0/1),
`GameConfig.interrogation` (`exchanges`, `answer_max_tokens`,
`thinking_budget`, `answer_reasoning_effort`, `detector`:
off/silent/shown); per-role `reasoning_effort` and `max_tokens`;
`run_games(..., detector=async fn(record) -> dict)`. A QA turn that fails 3
re-samples (same prompt, no feedback) raises → the game fails → excluded by
`extract.py` (preregistered "missing QA answer").

## Pilot history (seeds 0-2, never analysed; logs kept as `*_pilot_<tag>*`)

| tag | setting | outcome |
|---|---|---|
| `1k` | answer budget 1,000 + 200 | 3/3 failed: every answer hit 1,200 tokens |
| `authfail` | 4,000 + 200 | 3/3 failed: Modal client login clash (fixed) |
| `4k` | 4,000 + 200 | 3/3 failed: 9 of 20 answer attempts fit (713-3,397 tokens) |
| `12k` interrog / probe | 11,800 + 200 | 2/3 + 2/3 valid; 1 each lost to a 12K answer (genuine deliberation, ~50k chars) |
| `lowans` | + answer effort low | 2/3 valid; all 18 answers 1st attempt (489-5,667 tokens); 1 lost to the Turnip hitting 12K on a question |
| `aborted` | + Turnip cap 32K | stopped during the endpoint cold start, 0 games |

Deviations 1-4 in `PREREGISTRATION.md` record each change and the
researcher's choice. Signal seen so far (pilot, n tiny, not analysis):
Carrot answers' contrast −0.33 to −0.66, Parsnip −0.59 to +0.59, most
Parsnip answers above −0.2; the 12K probe pilot's Turnip voted the Parsnip in
both valid games; all 6 valid pilot games across arms were Carrot wins.

## Moving to the A40s — what changes

1. **Suspects (Qwen3.8-27B):** `serving/serve_qwen38.py` (`.venv-serve`,
   vLLM 0.31, same revision `1d4bf0f2`, served name `Qwen/Qwen3.8-27B`, port
   8001, four GPUs). Set `carrot.model` / `parsnip.model` / `eval_model` to
   `openai-api/vllm/Qwen/Qwen3.8-27B` and `VLLM_BASE_URL` /
   `VLLM_API_KEY=EMPTY` (the 2026-10-12 pattern); add `VLLM` to
   `config.providers` or drop the provider env logic for it. **Re-verify
   `reasoning_effort: low` on vLLM** — measured only on the Modal SGLang
   endpoint (742-1,958 → 100-170 reasoning tokens); if vLLM's template ignores
   it, answers go back to the 12K-deliberation exclusions.
2. **Turnip (DeepSeek-V4.1-Flash):** too big for the box. Keep the Modal
   shared endpoint (billed per token, no idle GPU) or OpenRouter (used for the
   2026-10-08 monitor). Researcher's call.
3. **Live probe:** port `modal_probe.py`'s `Detector.score` to a local
   service: same code path (`experiments/2026-10-13-fibs-basharena/acts.py`
   `load_model` / `render_followup` / `collect_windows`, `probe_inference`),
   `.venv`, decoder truncated at layer 58 (~49 GB bf16 → two A40s, as
   2026-10-12 did with `CUDA_VISIBLE_DEVICES=1,2`; needs
   `flash-linear-attention`). `play_games.py` runs in the engine's uv env
   without torch, so expose it over HTTP (e.g. a small FastAPI on localhost)
   and swap `make_detector` to call it. Check parity against the Modal scores
   on the pilot records (`qa_*_pilot_*.jsonl` hold full inputs + Modal
   contrast).
4. **GPU budget:** 4 GPUs for the server + 2 for the probe = 6 of 8; GPU 4 is
   often taken — check `nvidia-smi` first.
5. **Deviation 5** before any non-pilot data: serving stack Modal SGLang
   (+DFlash speculative decoding) / Modal H100 probe → local vLLM / local A40
   probe.
6. **Re-pilot 3 games per arm** on the new stack (gates: games complete, every
   answer scored, Turnip responds, answers rarely > 200 tokens), then:
   interrog main → `extract` → `calibrate` → probe main → `extract` →
   `analyze` (commands in `README.md`).

## Gotchas learned

- `.env`'s `MODAL_TOKEN_ID` / `Modal_TOKEN_SECRET` are a Modal **proxy token**
  (`wk-`/`ws-`), the bearer for the endpoints — not a Modal login. Inspect
  loads `.env` into the environment at eval time, which hijacked the Modal SDK
  ("Token ID is malformed"); `common.modal_client()` now passes the `cadenza`
  login from `~/.modal.toml` explicitly. Never `source .env` before running.
- The dedicated Qwen endpoint scales to zero ~2 min after the last request;
  cold start is minutes, and twice an H200 took ~20 min to be assigned;
  during it requests fail at once with 503 "no upstreams available".
  `play_games.wake_endpoints` polls first (20 min limit).
- Qwen3.8 thinks 1-12K tokens per discussion turn (several hit 12K and are
  re-sampled); DeepSeek at effort low similar.
- `inspect_ai.model._openai.openai_chat_tools` is private API (engine
  `tool_schemas`); Inspect 0.3.280.

## Spend (Modal `cadenza`, billing summary, period totals)

Endpoint $12.74 (Qwen dedicated, since its creation 15:37 CEST), deployed apps
$9.79 (≈ $5.35 since the probe app's deploy), LLM tokens (DeepSeek) $1.17.
The Modal resources stay: endpoints `qwen3.8-27B` (dedicated; scales to zero
but is still deployed) and `deepseekv41-flash`, app `turnip-probe`. Stopping
the dedicated endpoint is permanent (`modal endpoint stop`) — not done.
