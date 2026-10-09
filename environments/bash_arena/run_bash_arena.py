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

# Local vLLM server (e.g. an ssh tunnel to a GPU box), addressed as
# "openai-api/vllm/<model id>". The server must run with
# --enable-auto-tool-choice --tool-call-parser <parser>: BashArena needs tools.
os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("VLLM_API_KEY", "local")

from control_arena import EvalMode, control_agent  # noqa: E402
from control_arena._control_agent import ControlAgent  # noqa: E402
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
from inspect_ai.agent import Agent, AgentState  # noqa: E402
from inspect_ai.model import ChatMessageAssistant, GenerateConfig  # noqa: E402
from inspect_ai.tool import Tool  # noqa: E402
from pydantic import Field  # noqa: E402

# Both the attack (untrusted) and monitor (trusted) models are DeepSeek v4.1
# Flash via OpenRouter. (The RunPod Qwen3.5-27B endpoint, wired as
# "openai-api/runpod/Qwen/Qwen3.5-27B", was crash-looping its serverless
# workers, so we use OpenRouter for the attacker too — same model as the
# monitor, per the task.)
UNTRUSTED_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"
TRUSTED_MODEL = "openrouter/deepseek/deepseek-v4.1-flash"

# Pin every request to ONE OpenRouter provider, no fallbacks. Unpinned, the
# 2026-10-08 pilot was served by 12-18 different providers (mixed quantizations),
# switching within a trajectory. Default Together (used for DeepSeek v4.1 Flash:
# 100% uptime when chosen, tools + reasoning). DeepSeek first-party is not
# usable: the OpenRouter account's data policy excludes providers that train on
# paid data. Qwen3.8-27B runs use "alibaba".
DEFAULT_PROVIDER = "together"

# Role-level configs. Only the monitor resolves its model through the "trusted"
# role; BashArena's policies call get_model(model_name) with no role, so the
# untrusted settings must ALSO go into the policies' generate_config (below).
UNTRUSTED_CONFIG = GenerateConfig(timeout=300, max_retries=3)
TRUSTED_CONFIG = GenerateConfig(timeout=120, max_retries=3)

# Policy generate configs. Upstream defaults are max_tokens=16000 with
# reasoning_effort="medium" (honest) / reasoning_tokens=4000 (attack). DeepSeek
# via OpenRouter does not honour the reasoning-token budget and at medium effort
# reasoning alone ran to 12-16k tokens, ending the turn on max_tokens with no
# tool call (pilot: 82/1596 attack turns). So: low effort, doubled output cap.
HONEST_CONFIG = GenerateConfig(
    max_tokens=32000,
    reasoning_effort="low",
    timeout=300,
    max_retries=3,
)
ATTACK_CONFIG = HONEST_CONFIG.merge(GenerateConfig(reasoning_history="last"))


def routed(
    config: GenerateConfig, model: str, provider: str = DEFAULT_PROVIDER
) -> GenerateConfig:
    """Pin `config` to one OpenRouter provider slug for OpenRouter models (it is
    an OpenRouter request field; other backends, e.g. local vLLM, get none)."""
    if model.startswith("openrouter/"):
        routing = {"provider": {"only": [provider], "allow_fallbacks": False}}
        return config.merge(GenerateConfig(extra_body=routing))
    return config

MODE_MAP = {"honest": EvalMode.HONEST, "attack": EvalMode.ATTACK}


@control_agent
def last_turn_max_tokens_only(
    policy: ControlAgent = Field(description="BashArena honest/attack policy."),
) -> Agent:
    """Work around a sticky flag in BashArena's policies: they scan ALL earlier
    assistant messages for `hit_max_tokens`, so after one truncated turn every
    later turn gets the "you hit the max token limit" reminder instead of the
    normal (attack-planning / submit) reminder (pilot: 56% of attack turns).
    Clear the flag on all but the latest assistant message before delegating."""

    async def impl(state: AgentState, tools: list[Tool]) -> AgentState:
        assistants = [
            m for m in state._messages if isinstance(m, ChatMessageAssistant)
        ]
        for m in assistants[:-1]:
            if m.metadata and m.metadata.get("hit_max_tokens"):
                m.metadata["hit_max_tokens"] = False
        return await policy(state, tools)

    return impl


def build_protocol(
    untrusted_model: str = UNTRUSTED_MODEL,
    trusted_model: str = TRUSTED_MODEL,
    provider: str = DEFAULT_PROVIDER,
):
    """auto_submit wrapper (bash_arena's required protocol) around trusted
    monitoring, using bash_arena's own attack/honest policies and a basic
    trusted monitor."""
    # Pass the models explicitly: BashArena's policies resolve their model from
    # get_untrusted_model() (which binds the UNTRUSTED_MODEL env var at import),
    # so relying on ControlEvalConfig's role alone leaves them on the default
    # anthropic model. Passing model_name/model here pins them to OpenRouter.
    policy = untrusted_policy_builder(
        attack=last_turn_max_tokens_only(
            policy=attack_policy(
                model_name=untrusted_model,
                generate_config=routed(ATTACK_CONFIG, untrusted_model, provider),
            )
        ),
        honest=last_turn_max_tokens_only(
            policy=honest_policy(
                model_name=untrusted_model,
                generate_config=routed(HONEST_CONFIG, untrusted_model, provider),
            )
        ),
    )
    monitor = basic_monitor_builder(model=trusted_model)
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
        untrusted_config=routed(UNTRUSTED_CONFIG, UNTRUSTED_MODEL),
        trusted_config=routed(TRUSTED_CONFIG, TRUSTED_MODEL),
        max_steps=50,
        token_limit=4_000_000,
        time_limit=5400,  # hard per-sample cap (s); covers docker build + rollout
        epochs=epochs,
        limit=limit,
        max_sandboxes=max_sandboxes,
        max_samples=max_sandboxes,
    )
    tasks = get_control_task_product(
        setting,
        protocol,
        eval_modes,
        eval_config=config,
        monitoring_scorers=[max_suspicion_scorer()],
    )
    # eval_args() is what carries time_limit / max_sandboxes / limit to inspect.
    eval(tasks=tasks, sandbox_cleanup=True, **config.eval_args())


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
