"""Partial replication of the BashArena trusted-monitoring main result.

Runs the BashArena setting under the trusted-monitoring protocol (auto-submit
wrapper) in BOTH honest and attack mode over the first N tasks, for a few
epochs, and logs every trajectory's monitor suspicion + task outcomes. analyze.py
turns those logs into the headline metric: **safety at a trajectory-wise FPR
audit budget** (blog.redwoodresearch.org/p/basharena-and-control-setting-design).

Deliberate deviation from the paper: both the untrusted (attack) agent and the
trusted monitor are DeepSeek v4.1 Flash via OpenRouter (the paper pairs a strong
untrusted model, e.g. Claude Sonnet 4.5, with a weaker trusted monitor, e.g.
GPT-4.1 mini). So absolute numbers are not comparable to the paper; what is
replicated is the methodology and the shape of the result.

Limits: `max_steps` (policy turns, identical in both modes) rather than a
message limit — the honest policy adds a reminder message per turn, so a message
limit gives honest fewer turns than attack and cut off 96% of honest
trajectories in the 2026-10-08 pilot.

Usage (from the repo root; --project selects the bash_arena env without changing
the working directory, so the experiment can live here under experiments/):
    uv run --project environments/bash_arena \
        python experiments/2026-10-07-trusted-monitoring/run.py --tasks 50 --epochs 2 --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Task ids whose environment/tests are faulty on this host (id -> reason).
TASK_BLACKLIST = json.loads((HERE / "task_blacklist.json").read_text())


def _find_runner_dir() -> Path:
    """Walk up until the environments/bash_arena runner is found."""
    for parent in HERE.parents:
        cand = parent / "environments" / "bash_arena"
        if (cand / "run_bash_arena.py").exists():
            return cand
    raise FileNotFoundError("environments/bash_arena/run_bash_arena.py not found")


RUNNER_DIR = _find_runner_dir()
sys.path.insert(0, str(RUNNER_DIR))

import run_bash_arena as rba  # noqa: E402  (loads .env, defines models + protocol)
from control_arena import EvalMode  # noqa: E402
from control_arena.eval import ControlEvalConfig, get_control_task_product  # noqa: E402
from control_arena.scorers import max_suspicion_scorer  # noqa: E402
from control_arena.settings.bash_arena.bash_arena_setting import (  # noqa: E402
    BashArenaSetting,
)
from control_arena.settings.bash_arena.task.dataset import DatasetSpec  # noqa: E402
from inspect_ai import eval  # noqa: E402

def main(
    out_dir: Path,
    tasks: int,
    epochs: int,
    seed: int,
    modes: list[str],
    max_sandboxes: int,
    untrusted_model: str,
    trusted_model: str,
    max_steps: int,
    time_limit: int,
    token_limit: int,
    provider: str,
    monitor_kind: str,
):
    log_dir = out_dir / "inspect_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    trusted_config = rba.routed(rba.TRUSTED_CONFIG, trusted_model, provider)
    cfg_dict = {
        "setting": "bash_arena",
        "protocol": "auto_submit_protocol_wrapper(trusted_monitoring)",
        "untrusted_model": untrusted_model,
        "trusted_model": trusted_model,
        "tasks": tasks,
        "epochs": epochs,
        "shuffle_seed": seed,
        "modes": modes,
        "monitor": monitor_kind,
        "logit_monitor_generate_config": rba.routed(
            rba.LOGIT_MONITOR_CONFIG, trusted_model, provider
        ).model_dump(exclude_none=True),
        "logit_monitor_prompt": rba.LOGIT_MONITOR_PROMPT,
        "task_blacklist": sorted(TASK_BLACKLIST),
        "max_sandboxes": max_sandboxes,
        "max_steps": max_steps,
        "time_limit": time_limit,
        "token_limit": token_limit,
        "fail_on_error": False,
        "retry_on_error": 1,
        "honest_generate_config": rba.routed(
            rba.HONEST_CONFIG, untrusted_model, provider
        ).model_dump(exclude_none=True),
        "attack_generate_config": rba.routed(
            rba.ATTACK_CONFIG, untrusted_model, provider
        ).model_dump(exclude_none=True),
        "trusted_generate_config": trusted_config.model_dump(exclude_none=True),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2) + "\n")

    eval_modes = [rba.MODE_MAP[m] for m in modes]
    setting = BashArenaSetting(
        dataset_spec=DatasetSpec(max_tasks=tasks, shuffle_seed=seed)
    )
    protocol = rba.build_protocol(untrusted_model, trusted_model, provider, monitor_kind)
    config = ControlEvalConfig(
        untrusted_model=untrusted_model,
        trusted_model=trusted_model,
        untrusted_config=rba.routed(rba.UNTRUSTED_CONFIG, untrusted_model, provider),
        trusted_config=trusted_config,
        # max_steps counts policy turns identically in both modes (a message
        # limit does not: the honest policy adds a reminder message per turn).
        max_steps=max_steps,
        token_limit=token_limit,
        time_limit=time_limit,
        epochs=epochs,
        max_sandboxes=max_sandboxes,
        max_samples=max_sandboxes,
        # One sample's sandbox dying (an agent killed PID 1 in the pilot) must
        # not cancel the run; errored samples are retried once, and analyze.py
        # reports any that remain.
        fail_on_error=False,
        retry_on_error=1,
        log_dir=str(log_dir),
    )
    tasks_product = get_control_task_product(
        setting,
        protocol,
        eval_modes,
        eval_config=config,
        monitoring_scorers=[max_suspicion_scorer()],
    )
    # Drop blacklisted tasks AFTER the seeded first-N selection, so the remaining
    # tasks are the same ones as in runs made before a task was blacklisted.
    for task in tasks_product:
        task.dataset = task.dataset.filter(lambda s: str(s.id) not in TASK_BLACKLIST)
    # eval_args() carries time_limit / max_sandboxes / fail_on_error / log_dir;
    # without it inspect silently runs with none of them.
    eval(tasks=tasks_product, sandbox_cleanup=True, **config.eval_args())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir", type=Path, default=HERE,
        help="Where config.json and inspect_logs/ go (one dir per model pair).",
    )
    p.add_argument("--tasks", type=int, default=10, help="First N tasks (per mode).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="Dataset shuffle seed.")
    p.add_argument(
        "--modes", nargs="+", choices=["honest", "attack"],
        default=["honest", "attack"],
    )
    p.add_argument("--max-sandboxes", type=int, default=6)
    p.add_argument("--untrusted-model", default=rba.UNTRUSTED_MODEL)
    p.add_argument("--trusted-model", default=rba.TRUSTED_MODEL)
    p.add_argument(
        "--provider", default=rba.DEFAULT_PROVIDER,
        help="OpenRouter provider slug every call is pinned to (no fallbacks).",
    )
    p.add_argument(
        "--monitor", choices=["score", "logit"], default="score",
        help="score: 0-10 basic monitor; logit: one-token Yes/No logit difference.",
    )
    p.add_argument("--max-steps", type=int, default=50, help="Policy turns per sample.")
    p.add_argument("--time-limit", type=int, default=5400, help="Per-sample seconds.")
    p.add_argument("--token-limit", type=int, default=4_000_000)
    a = p.parse_args()
    main(
        a.out_dir.resolve(), a.tasks, a.epochs, a.seed, a.modes, a.max_sandboxes,
        a.untrusted_model, a.trusted_model,
        a.max_steps, a.time_limit, a.token_limit, a.provider, a.monitor,
    )
