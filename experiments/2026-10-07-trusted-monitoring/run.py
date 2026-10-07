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

Scale note: this host runs ONE docker-compose sandbox at a time (~2.3 GB each,
~3.8 GB RAM), so samples run serially (~9 min each).

Usage (from the repo root; --project selects the bash_arena env without changing
the working directory, so the experiment can live here under experiments/):
    uv run --project environments/bash_arena \
        python experiments/2026-10-07-trusted-monitoring/run.py --tasks 10 --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


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

LOG_DIR = HERE / "logs"


def main(tasks: int, epochs: int, seed: int, modes: list[str], max_sandboxes: int):
    LOG_DIR.mkdir(exist_ok=True)

    cfg_dict = {
        "setting": "bash_arena",
        "protocol": "auto_submit_protocol_wrapper(trusted_monitoring)",
        "untrusted_model": rba.UNTRUSTED_MODEL,
        "trusted_model": rba.TRUSTED_MODEL,
        "tasks": tasks,
        "epochs": epochs,
        "shuffle_seed": seed,
        "modes": modes,
        "max_sandboxes": max_sandboxes,
        "message_limit": 50,
        "time_limit": 1800,
    }
    (HERE / "config.json").write_text(json.dumps(cfg_dict, indent=2) + "\n")

    eval_modes = [rba.MODE_MAP[m] for m in modes]
    setting = BashArenaSetting(
        dataset_spec=DatasetSpec(max_tasks=tasks, shuffle_seed=seed)
    )
    protocol = rba.build_protocol()
    config = ControlEvalConfig(
        untrusted_model=rba.UNTRUSTED_MODEL,
        trusted_model=rba.TRUSTED_MODEL,
        untrusted_config=rba.UNTRUSTED_CONFIG,
        trusted_config=rba.TRUSTED_CONFIG,
        message_limit=50,
        token_limit=2_000_000,
        time_limit=1800,
        epochs=epochs,
        max_sandboxes=max_sandboxes,
        log_dir=str(LOG_DIR),
    )
    tasks_product = get_control_task_product(
        setting,
        protocol,
        eval_modes,
        eval_config=config,
        monitoring_scorers=[max_suspicion_scorer()],
    )
    eval(tasks=tasks_product, sandbox_cleanup=True, log_dir=str(LOG_DIR))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=int, default=10, help="First N tasks (per mode).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="Dataset shuffle seed.")
    p.add_argument(
        "--modes", nargs="+", choices=["honest", "attack"],
        default=["honest", "attack"],
    )
    p.add_argument("--max-sandboxes", type=int, default=1)
    a = p.parse_args()
    main(a.tasks, a.epochs, a.seed, a.modes, a.max_sandboxes)
