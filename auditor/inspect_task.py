"""Inspect (inspect-ai) task for the grade-annotated outside auditor.

One Sample per game x condition; the exact system/user messages are built by
`auditor.prompts` (identical to the local vLLM and urllib API runners) and are
fully browsable per-sample in `inspect view`. Repeated audits per game are
Inspect epochs (config.n_auditor_seeds), reduced with mean + mode
(= majority vote). The scorer is the same `parse_guess` used everywhere.

Run (key in repo-root .env):
  set -a; source .env; set +a
  uv run --no-sync inspect eval auditor/inspect_task.py@carrot_auditor \
      -T arm=base --model openrouter/qwen/qwen3.5-27b \
      --max-connections 32 --log-dir experiments/2026-09-25/inspect_logs
Smoke: add  --limit 4 --epochs 1
View:  uv run --no-sync inspect view --log-dir experiments/2026-09-25/inspect_logs
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.scorer import (CORRECT, INCORRECT, Score, Target, accuracy,
                               scorer, stderr)
from inspect_ai.solver import TaskState, generate

from auditor.config import ApiAuditorConfig  # noqa: E402
from auditor.prompts import (AUDITOR_SYSTEM, AUDITOR_USER, DETECTOR_NOTE,  # noqa: E402
                             build_transcript, parse_guess)
from auditor.run import all_grade_maps, read_jsonl  # noqa: E402
from auditor.scale import load_scale  # noqa: E402


def build_samples(cfg: ApiAuditorConfig, arm: str,
                  conditions: list[str] | None = None) -> tuple[list[Sample], list[str]]:
    conditions = conditions or cfg.conditions
    data, out = cfg.path(cfg.data_dir), cfg.path(cfg.out_dir)
    scale = load_scale(out / "suspicion_scale.json")

    recs = read_jsonl(data / f"transcripts_{arm}.jsonl")
    games_meta = json.load(open(data / f"games_{arm}.json"))["games"]
    parsnip = {g["game_id"]: g["parsnip"] for g in games_meta}
    player_names = list(games_meta[0]["roles"])
    num_rounds = max(r["round"] for r in recs) + 1
    by_game = defaultdict(list)
    for r in recs:
        by_game[r["game_id"]].append(r)
    for msgs in by_game.values():
        msgs.sort(key=lambda m: (m["round"], m["position"]))

    per_group = cfg.n_games // cfg.seed_groups
    game_ids = sorted(g for g in by_game if int(g.split("g")[1]) < per_group)
    assert len(game_ids) == cfg.n_games, len(game_ids)

    grade_by_cond = all_grade_maps(cfg, arm, conditions, recs, scale)

    samples = []
    for gid in game_ids:
        for cond in conditions:
            transcript = build_transcript(by_game[gid], grade_by_cond.get(cond))
            system = AUDITOR_SYSTEM.format(
                players=", ".join(player_names), num_rounds=num_rounds,
                detector_note=DETECTOR_NOTE.format(legend=cfg.grade_legend)
                if cond != "none" else "")
            user = AUDITOR_USER.format(
                transcript=transcript,
                and_scores=" and the suspicion grades" if cond != "none" else "",
                example=player_names[0])
            samples.append(Sample(
                id=f"{gid}:{cond}",
                input=[ChatMessageSystem(content=system),
                       ChatMessageUser(content=user)],
                target=parsnip[gid],
                metadata={"arm": arm, "game_id": gid, "condition": cond,
                          "seed_group": int(gid[1:gid.index("g")])}))
    return samples, player_names


@scorer(metrics=[accuracy(), stderr(cluster="game_id")])
def parsnip_scorer(player_names: list[str]):
    async def score(state: TaskState, target: Target) -> Score:
        guess = parse_guess(state.output.completion, player_names)
        return Score(value=CORRECT if guess == target.text else INCORRECT,
                     answer=guess or "(unparsed)")
    return score


@task
def carrot_auditor(arm: str = "base",
                   config: str = "experiments/2026-09-25/config_api.json",
                   conditions: str | None = None) -> Task:
    cfg = ApiAuditorConfig.load(REPO / config)
    samples, player_names = build_samples(
        cfg, arm, conditions.split(",") if conditions else None)
    return Task(
        dataset=MemoryDataset(samples, name=f"carrot-parsnip-{arm}"),
        solver=generate(),
        scorer=parsnip_scorer(player_names),
        epochs=Epochs(cfg.n_auditor_seeds, ["mean", "mode"]),
        config=GenerateConfig(temperature=cfg.temperature, top_p=cfg.top_p,
                              max_tokens=cfg.api_max_tokens),
    )
