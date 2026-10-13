# 2026-10-13 — fibs probes trained for Qwen3.8-27B on BashArena trajectories

Do fibs probes (Hollinsworth et al. 2026, arXiv:2610.12445), trained in-house
on Qwen3.8-27B's activations over FIBSv1, separate side-task (attack) steps
from honest steps in the existing Qwen3.8-27B BashArena rollouts
(`experiments/2026-10-10-qwen-logit-monitor/inspect_logs/`)? Design,
primary metric (per-step AUROC of the linear ensemble, judge-labelled onset,
threshold 0.90) and decision rule: `PREREGISTRATION.md`. Not yet run.
