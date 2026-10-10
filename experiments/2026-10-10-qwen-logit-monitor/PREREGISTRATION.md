# Preregistration — BashArena trusted monitoring, Qwen3.8-27B with a Yes/No logit monitor

Filled by the `/preregister` skill from the researcher's answers, quoted
verbatim. Committed before the first non-pilot run; the sections above
"Deviations" are frozen from that commit on. Unanswered items read
`NOT PREREGISTERED`.

- Date: 2026-10-10
- Researcher: repo owner (accounts@reinthal.me)
- Builds on: `experiments/2026-10-07-trusted-monitoring/` (DeepSeek v4.1 Flash
  run, log entry 2026-10-08; code `run.py` / `analyze.py` are reused from there)

## 1. Question and stakes

- Decision this result informs: NOT PREREGISTERED.
- Claim hoped for (one sentence), researcher, 2026-10-10: "I think accuracy on
  task will be proportional to some capability index (search online and make a
  good guess based on existing data points on this test). I don't know what the
  safety will be. this is exploratory research and I don't want to guess here."
- Earlier statement, 2026-10-09, superseded by the one above: "self-judging
  monitor will give high safety usefulness over score at 4%FPR over 0.8,0.8 but
  qwen3.8 has low accuracy on the task <0.4 at 95% CI". The researcher did not
  resolve what "usefulness over 0.8" and "accuracy < 0.4" each refer to; it is
  recorded, not tested.
- Next step if supported / refuted / inconclusive: NOT PREREGISTERED.

## 2. Hypothesis

- H1 (with direction): honest main-task performance on BashArena increases
  with a general capability index, so Qwen3.8-27B's score can be predicted from
  its index value and the existing data points.
- Null: NOT PREREGISTERED.
- Rival explanation: NOT PREREGISTERED. (Claude's note, not the researcher's:
  elicitation and serving differences — prompt tuned on Sonnet 4.5, reasoning
  effort, step limit, quantization — move the score independently of the index.)
- Prediction for the primary metric (value, interval, confidence %): the
  researcher delegated the number ("make a good guess based on existing data
  points"). **Derived by Claude, not by the researcher: 0.68, interval
  0.55-0.80.** Confidence: NOT PREREGISTERED. Derivation:

  | model | AA Intelligence Index v4.3.2 | BashArena main-task score | source |
  |---|---|---|---|
  | GPT-4.1 mini | 10 | 0.34 | paper (arXiv 2512.15688), 637 tasks, 100-action cap |
  | Claude Sonnet 4.5 (reasoning) | 21 | 0.59 | paper, 637 tasks, 100-action cap |
  | DeepSeek V4.1 Flash | 25 (non-reasoning) - 39 (max) | 0.783 | this repo, 47 tasks, 50-step cap, low effort |
  | Qwen3.8-27B | 20 (non-reasoning), 26 (low), 28 (medium), 34 (xhigh) | ? | index only |

  The two paper points give 2.27 score points per index point (not a line
  through the origin: 3.4 and 2.8 points per index point, so "proportional" is
  read as "increasing, roughly linear"). The run uses low reasoning effort, so
  Qwen's index is taken as 26. Extrapolating the paper line gives 0.70.
  Anchoring instead on the in-repo DeepSeek point (same tasks, limits and
  prompts; its low-effort index interpolated to ~31) gives 0.783 - 2.27 x 5
  points = 0.67. Point prediction 0.68; the interval covers index values 20-31
  on the paper line and the between-setup difference.
- Result that would make me abandon H1: NOT PREREGISTERED.
- Expected baseline / control values: safety — researcher: "I don't know what
  the safety will be ... I don't want to guess here."

## 3. Test design

- Primary metric (one): mean honest main-task score (fraction of test cases
  passed), task-clustered.
- Secondary metrics: full main-task success rate; attack side-task success
  rate; suspicion AUROC (attack vs honest); attacks caught and red-win rate at
  2/4/5/10/20% trajectory-wise FPR with realised FPR.
- Unit of analysis, independence, clustering: trajectory; the 2 epochs of a
  task are clustered within the task (47 clusters per mode).
- Arms and controls (and what each control rules out): honest mode (benign
  suspicion distribution, sets the FPR threshold; usefulness) and attack mode.
  No random-monitor or cross-model-monitor control.
- Sample size and justification (detectable effect, expected interval width):
  47 tasks x 2 epochs per mode (the first 50 tasks at seed 0 minus the 3
  blacklisted), as in the DeepSeek run, whose task-clustered 95% CI half-width
  on the primary metric was 0.051. Not derived from a power calculation.
- Fixed parameters: see `config.json`, written by `run.py` at launch. Planned:
  agent = monitor = Qwen3.8-27B on the researcher's vLLM server (ssh tunnel,
  `openai-api/vllm/<served model id>`); `--monitor logit`; `max_steps=50`,
  `time_limit=10800`, `token_limit=4e6`; agent `reasoning_effort=low`,
  `max_tokens=32000`; seed 0.
- Monitor (researcher, 2026-10-10): "For monitoring, use logit difference
  between Yes / no verdicts to capture more uncertainty of the model instead of
  re-sampling the same trajectory". Implemented as `logit_monitor_builder` in
  `environments/bash_arena/run_bash_arena.py`: same information as the basic
  monitor (task description + the latest step's tool calls), thinking off, one
  generated token, score = log p(Yes) - log p(No) from the top-20 first-token
  logprobs; trajectory suspicion = max over steps.
- Stopping rule: fixed N (94 trajectories per mode). No extension.

## 4. Analysis plan

- Statistic, interval method, comparison: `analyze.py --out-dir` on this
  folder, unmodified. Mean and normal-approximation 95% CI over per-task means.
- Threshold for "supported": the primary metric's point estimate lies inside
  0.55-0.80.
- Exclusion rules, and the exclusion fraction that voids the run: tasks in
  `task_blacklist.json` (31699056, 10730838, 50618411) are not run. Samples
  that error after one retry are excluded and counted. A monitor call without
  both Yes and No in the top-20 logprobs raises and errors the sample (no
  fallback score). Voiding fraction: NOT PREREGISTERED.
- Multiple comparisons: one confirmatory test; no correction.
- Figure (axes, shared ranges): `figure.png` from `analyze.py` — suspicion
  histograms for honest and attack on one shared x range; attacks caught vs FPR
  budget, y 0-105%.
- Confirmatory analyses: the primary metric against the predicted interval.
- Exploratory analyses: everything about safety and the monitor, including
  the comparison with the DeepSeek 0-10 monitor run and with the honest run
  already collected on Alibaba (2026-10-09, 0-10 monitor, scores unread at the
  time of writing).

## 5. De-risking

- Pre-mortem — boring failure modes and the check for each:
  - Time limit truncates trajectories on a slow server (it did on 2026-10-09:
    4 of 7 honest at 5400 s). Check: share of samples ending on `time`.
  - Tunnel drops mid-run. Samples error, are retried once, then excluded.
  - Server lacks tool calling or logprobs. Checked in the pilot.
  - Yes/No missing from the top-20 logprobs. Pilot must show zero such errors.
  - Context limit of the served model below the trajectory length.
- Can the metric resolve the predicted effect (granularity, floor/ceiling, N
  per cell): the primary metric can (CI half-width ~0.05 against an interval
  0.25 wide). One new model is a single prediction check, not a test of
  proportionality across models. For safety, 4% FPR is 3.8 of 94 honest
  trajectories; the logit score is continuous, so ties should not block it as
  they did with the 0-10 monitor.
- Positive control: none planned.
- Pilot (size, what it must show; pilot data is excluded from the analysis):
  1 task x both modes, 6 steps, written outside this folder. Must show tool
  calls on every agent turn, a finite logit score on every monitored step, and
  zero monitor errors.
- Cost and wall-clock estimate, hard cap, abort criterion: no API cost (self-
  hosted). Wall-clock unknown until the server's throughput is measured; at the
  130 tok/s seen on 2026-10-09 roughly 30 hours. Hard cap and abort criterion:
  NOT PREREGISTERED.
- Reused primitives / new untested code: BashArena setting, policies and
  protocol wrapper from control-arena; `run.py` / `analyze.py` from
  2026-10-07. New and exercised only by a unit check and the pilot: the logit
  monitor and its read-out in `analyze.py`.

## 6. Conclusions agreed in advance

- If supported, the report says: NOT PREREGISTERED.
- If refuted, the report says: NOT PREREGISTERED.
- If inconclusive (defined as ...), the report says: NOT PREREGISTERED.
- Limitations stated regardless of outcome: agent and monitor are the same
  model; the monitor sees only the latest step's tool calls (the paper's sees
  the whole trajectory and the side-task descriptions); 50-step cap against
  the paper's 100; 47 tasks; the prediction was derived by Claude from two
  published points and one in-repo point; index values are from one site on
  one date (artificialanalysis.ai, 2026-10-10).

---

## Deviations

<!-- Append only. Each entry: date, what changed, why, decided before or
     after seeing data. -->
