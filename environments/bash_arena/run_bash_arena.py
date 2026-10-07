"""Run the BashArena ControlArena setting on this host with Docker sandboxes.

Model wiring (per the task brief):
  - UNTRUSTED / attack model  = Qwen3.5-27B served by the RunPod serverless vLLM
    endpoint (iac/), reached through its OpenAI-compatible API. inspect_ai's
    ``openai-api`` provider reads ``RUNPOD_BASE_URL`` + ``RUNPOD_API_KEY``.
  - TRUSTED / monitor model    = DeepSeek v4.1 Flash via OpenRouter
    (``openrouter/...``), reading ``OPENROUTER_API_KEY``.

Secrets come from the repo-root .env (RUNPOD_API_KEY, OPENROUTER_API_KEY,
OPENROUTER_BASE_URL). RUNPOD_BASE_URL is derived from RUNPOD_ENDPOINT_ID here if
not already exported.

Usage (from the repo root, via uv selecting this env with --directory):
    uv run --directory environments/bash_arena python run_bash_arena.py                        # honest, 1 task
    uv run --directory environments/bash_arena python run_bash_arena.py --modes honest attack --limit 1
    uv run --directory environments/bash_arena python run_bash_arena.py --limit 2 --max-sandboxes 1

BashArena spins up a per-sample docker-compose network (default container +
router + logging/malware/database servers, ~2.3 GB RAM total). This host has
~3.8 GB, so keep --max-sandboxes at 1.
"""

import argparse
import os
from pathlib import Path

# ---- Load secrets from the repo-root .env before importing inspect/openai ----
def _find_repo_root() -> Path:
    """Walk up from this file until a directory containing .env is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".env").exists():
            return parent
    raise FileNotFoundError(
        "No .env found in any parent directory of run_bash_arena.py"
    )


REPO_ROOT = _find_repo_root()


def _load_dotenv(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv(REPO_ROOT / ".env")

# RunPod serverless endpoint serving Qwen/Qwen3.5-27B (managed by iac/).
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "jd42hxujb3bagp")
os.environ.setdefault(
    "RUNPOD_BASE_URL",
    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1",
)

from control_arena import EvalMode  # noqa: E402
from control_arena.eval import ControlEvalConfig, get_control_task_product  # noqa: E402
from control_arena.micro_protocol import trusted_monitoring_builder  # noqa: E402
from control_arena.monitor import basic_monitor_builder  # noqa: E402
from control_arena.policy import untrusted_policy_builder  # noqa: E402
from control_arena.scorers import max_suspicion_scorer  # noqa: E402
from control_arena.settings.bash_arena.bash_arena_setting import (  # noqa: E402
    BashArenaSetting,
)
from control_arena.settings.bash_arena.policies.attack_policy import (  # noqa: E402
    attack_policy,
)
from control_arena.settings.bash_arena.policies.honest_policy import (  # noqa: E402
    honest_policy,
)
from control_arena.settings.bash_arena.protocol.auto_submit_protocol_wrapper import (  # noqa: E402
    auto_submit_protocol_wrapper,
)
from control_arena.settings.bash_arena.task.dataset import DatasetSpec  # noqa: E402
from inspect_ai import eval  # noqa: E402
from inspect_ai.model import GenerateConfig  # noqa: E402

# Both the attack (untrusted) and monitor (trusted) models are DeepSeek v4.1
# Flash via OpenRouter. (The RunPod Qwen3.5-27B endpoint, wired as
# "openai-api/runpod/Qwen/Qwen3.5-27B", was crash-looping its serverless
# workers, so we use OpenRouter for the attacker too — same model as the
# monitor, per the task.)
UNTRUSTED_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"
TRUSTED_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"

UNTRUSTED_CONFIG = GenerateConfig(timeout=300, max_retries=3)
TRUSTED_CONFIG = GenerateConfig(timeout=120, max_retries=3)

MODE_MAP = {"honest": EvalMode.HONEST, "attack": EvalMode.ATTACK}


def build_protocol():
    """auto_submit wrapper (bash_arena's required protocol) around trusted
    monitoring, using bash_arena's own attack/honest policies and a basic
    trusted monitor."""
    # Pass the models explicitly: BashArena's policies resolve their model from
    # get_untrusted_model() (which binds the UNTRUSTED_MODEL env var at import),
    # so relying on ControlEvalConfig's role alone leaves them on the default
    # anthropic model. Passing model_name/model here pins them to OpenRouter.
    policy = untrusted_policy_builder(
        attack=attack_policy(model_name=UNTRUSTED_MODEL),
        honest=honest_policy(model_name=UNTRUSTED_MODEL),
    )
    monitor = basic_monitor_builder(model=TRUSTED_MODEL)
    return auto_submit_protocol_wrapper(
        micro_protocol=trusted_monitoring_builder(policy=policy, monitor=monitor),
    )


def main(
    epochs: int = 1,
    modes: list[str] | None = None,
    limit: int | None = 1,
    max_sandboxes: int = 1,
) -> None:
    eval_modes = [MODE_MAP[m] for m in (modes or ["honest"])]

    setting = BashArenaSetting(dataset_spec=DatasetSpec())
    protocol = build_protocol()
    config = ControlEvalConfig(
        untrusted_model=UNTRUSTED_MODEL,
        trusted_model=TRUSTED_MODEL,
        untrusted_config=UNTRUSTED_CONFIG,
        trusted_config=TRUSTED_CONFIG,
        message_limit=50,
        token_limit=2_000_000,
        time_limit=1800,  # hard per-sample cap (s); covers docker build + rollout
        epochs=epochs,
        max_sandboxes=max_sandboxes,
    )
    tasks = get_control_task_product(
        setting,
        protocol,
        eval_modes,
        eval_config=config,
        monitoring_scorers=[max_suspicion_scorer()],
    )
    eval(tasks=tasks, sandbox_cleanup=True, limit=limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BashArena defense eval.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODE_MAP),
        default=["honest"],
        help="Eval mode(s) to run (default: honest).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Cap each mode's dataset to its first N tasks (default: 1).",
    )
    parser.add_argument(
        "--max-sandboxes",
        type=int,
        default=1,
        help="Concurrent docker-compose sandboxes (default: 1; this host fits ~1).",
    )
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        modes=args.modes,
        limit=args.limit,
        max_sandboxes=args.max_sandboxes,
    )
