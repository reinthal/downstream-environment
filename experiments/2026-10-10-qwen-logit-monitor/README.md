# BashArena trusted monitoring — Qwen3.8-27B, Yes/No logit monitor (2026-10-10)

Preregistered (see `PREREGISTRATION.md`): does Qwen3.8-27B's honest main-task
score land where a capability index predicts (0.55-0.80)? Safety under a
self-monitor is exploratory.

Design: the 2026-10-07 trusted-monitoring experiment (same tasks, limits,
policies) with two changes — the model (Qwen3.8-27B, self-hosted vLLM, as agent
and monitor) and the monitor (one-token Yes/No verdict; suspicion = log p(Yes)
- log p(No)). Code lives in `experiments/2026-10-07-trusted-monitoring/`
(`run.py`, `analyze.py`, `task_blacklist.json`) and
`environments/bash_arena/run_bash_arena.py`; this folder holds the run.

```bash
M=openai-api/vllm/<served model id>     # VLLM_BASE_URL defaults to localhost:8000/v1
uv run --project environments/bash_arena python experiments/2026-10-07-trusted-monitoring/run.py \
  --out-dir experiments/2026-10-10-qwen-logit-monitor --tasks 50 --epochs 2 --seed 0 \
  --monitor logit --time-limit 10800 --untrusted-model $M --trusted-model $M
uv run --project environments/bash_arena python experiments/2026-10-07-trusted-monitoring/analyze.py \
  --out-dir experiments/2026-10-10-qwen-logit-monitor
```

Result (2026-10-11): honest main-task score 0.705 ± 0.075 against the
preregistered 0.55-0.80 — supported. Exploratory: AUROC 0.819; attacks caught
38% at a realised 4.5% FPR, 73% at 20.5%. Details in `research-log/log.md`.

Deliberate limitations: see section 6 of the preregistration.
