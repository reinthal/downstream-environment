"""Turn the engine's Inspect logs into the repo's transcript files (CPU).

Copied from experiments/2026-10-09-3p-game/extract.py (same parsing and
exclusion rules); only the config class and the per-turn provider field
(OpenRouter-only) differ.

The orchestrator keeps announcements, speaking orders and votes only inside
each game's Inspect log, so they are read back from the model / tool events:
a discussion turn is one tool-less "reasoning" call followed by the
announcement call(s); a vote turn is the same with the vote tool.

  transcripts_{arm}.jsonl  one row per discussion turn of every valid game
                           (09-22 schema; `reasoning` is the explicit
                           reasoning turn, `input_messages` the four messages
                           the model saw when it made the announcement)
  games_{arm}.json         per valid game: roles, parsnip, speaking orders,
                           the players' own votes, eliminated, winner
  excluded_{arm}.json      games dropped by the preregistered rules, with reason

Run:  uv run --no-project --with-requirements environments/carrot_parsnip/requirements.txt \
          python experiments/2026-10-12-qwen38-layer-sweep/extract.py --arm base
"""
from __future__ import annotations

import argparse
import json
import re

from inspect_ai.log import read_eval_log

from common import OUT, REPO, load_config, write_jsonl

from game_config import GameConfig  # noqa: E402  (engine, via common's sys.path)

NAME_RE = re.compile(r"^You are (\w+), playing")
ROUND_RE = re.compile(r"GROUP DISCUSSION - Round (\d+) of (\d+)")
ORDER_RE = re.compile(r"^  (\d+)\. (\w+) — (already spoke|YOUR TURN \(now\)|"
                      r"has not been selected to speak yet)$", re.M)
ROLE_RE = re.compile(r"^  \[(\d+)\] (\w+) — (Carrot|Parsnip) \[(ALIVE|ELIMINATED)\]$", re.M)
WINNER_RE = re.compile(r"^WINNER: (\w+)$", re.M)
VOTE_MARK = "Before casting your elimination vote"
ANNOUNCED = "Your announcement has been made public"
VOTED = "Elimination vote cast"


class GameFailed(Exception):
    """The game did not complete (provider error, missing vote, ...) or broke
    the one-announcement-per-turn format (2026-10-09 PREREGISTRATION.md, Deviations)."""


def turns(sample) -> list[dict]:
    """Group a game's events into turns: each starts at a tool-less model call."""
    out = []
    for e in sample.events:
        if e.event == "model" and not e.tools:
            out.append({"reason": e, "calls": [], "tools": []})
        elif e.event == "model":
            out[-1]["calls"].append(e)
        elif e.event == "tool":
            out[-1]["tools"].append(e)
    return out


def parse_game(log, game_cfg: GameConfig) -> tuple[dict, list[dict]]:
    if log.status != "success" or not log.samples:
        raise GameFailed(f"log status {log.status}")
    (sample,) = log.samples
    if sample.error is not None:
        raise GameFailed(f"sample error: {sample.error.message.splitlines()[0][:200]}")
    summary = sample.messages[-1].text
    winner = WINNER_RE.search(summary)
    if winner is None:
        raise GameFailed("no winner: the vote did not resolve")
    players = [(int(i), name, role, alive) for i, name, role, alive in ROLE_RE.findall(summary)]
    names = [name for _, name, _, _ in players]
    if names != game_cfg.names():
        raise ValueError(f"players in the summary {names} != configured {game_cfg.names()}")
    roles = {name: role for _, name, role, _ in players}

    n, rounds = game_cfg.num_players, game_cfg.num_rounds_discussion
    orders: list[list[int]] = [[] for _ in range(rounds)]
    rows, votes = [], {}
    for t in turns(sample):
        e = t["reason"]
        system, prompt = e.input[0].text, e.input[1].text
        speaker = NAME_RE.match(system).group(1)
        if VOTE_MARK in prompt:
            cast = [x for x in t["tools"] if x.function == "CastEliminationVoteTool"
                    and str(x.result).startswith(VOTED)]
            (vote,) = cast
            votes[speaker] = names[vote.arguments["target_index"]]
            continue
        rnd, of = (int(x) for x in ROUND_RE.search(prompt).groups())
        if of != rounds:
            raise ValueError(f"prompt says {of} rounds, config says {rounds}")
        lines = ORDER_RE.findall(prompt)
        order = [names.index(name) for _, name, _ in lines]
        (pos,) = [i for i, (_, _, tag) in enumerate(lines) if tag.startswith("YOUR TURN")]
        if sorted(order) != list(range(n)) or names[order[pos]] != speaker:
            raise ValueError(f"bad speaking order {lines} for speaker {speaker}")
        if orders[rnd - 1] not in ([], order) or pos != len([r for r in rows if r["round"] == rnd - 1]):
            raise ValueError(f"speaking order changed within round {rnd}")
        orders[rnd - 1] = order
        made = [x for x in t["tools"] if x.function == "MakeAnnouncementTool"
                and str(x.result).startswith(ANNOUNCED)]
        if len(made) > 1:
            raise GameFailed(f"{speaker} made {len(made)} announcements in one turn")
        first_call = t["calls"][0]
        rows.append({
            "round": rnd - 1, "position": pos, "speaker": speaker,
            "speaker_index": names.index(speaker), "role": roles[speaker],
            "input_messages": [{"role": m.role, "content": m.text} for m in first_call.input],
            "reasoning": e.output.message.text,
            "reasoning_stop": e.output.stop_reason,
            "announcement": made[0].arguments["message"] if made else "",
        })
    if len(rows) != n * rounds:
        raise ValueError(f"{len(rows)} discussion turns, expected {n * rounds}")
    if sorted(votes) != sorted(names):
        raise ValueError(f"votes from {sorted(votes)}, expected every player")
    (parsnip,) = [name for name, role in roles.items() if role == "Parsnip"]
    meta = {"roles": roles, "parsnip": parsnip, "orders": orders, "votes": votes,
            "eliminated": [name for _, name, _, alive in players if alive == "ELIMINATED"],
            "winner": winner.group(1)}
    return meta, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["base", "pilot"], required=True)
    args = ap.parse_args()
    cfg = load_config()
    game_cfg = GameConfig.from_yaml(REPO / cfg.game_config)
    seed = cfg.game_seed if args.arm == "base" else cfg.pilot_seed
    last_round = game_cfg.num_rounds_discussion - 1

    games, records, excluded = [], [], []
    for path in sorted((OUT / f"inspect_logs_{args.arm}").glob("game_*.eval")):
        idx = int(path.stem.removeprefix("game_"))
        ident = {"game_id": f"g{idx:03d}", "seed_group": 0, "game_idx": idx, "game_seed": seed + idx}
        try:
            meta, rows = parse_game(read_eval_log(str(path), resolve_attachments=True), game_cfg)
        except GameFailed as exc:
            excluded.append({**ident, "reason": str(exc)})
            continue
        silent = [r["speaker"] for r in rows if r["round"] == last_round and not r["announcement"]]
        if silent:
            excluded.append({**ident, "reason": f"silent in the last round: {', '.join(silent)}"})
            continue
        games.append({**ident, **meta})
        records += [{**ident, **r, "arm": args.arm} for r in rows]

    write_jsonl(OUT / f"transcripts_{args.arm}.jsonl", records)
    (OUT / f"games_{args.arm}.json").write_text(json.dumps(
        {"arm": args.arm, "game_config": cfg.game_config, "games": games}, indent=1) + "\n")
    (OUT / f"excluded_{args.arm}.json").write_text(json.dumps(excluded, indent=1) + "\n")
    print(f"{len(games)} valid games ({len(records)} turns), {len(excluded)} excluded")
    for x in excluded:
        print("  excluded", x["game_id"], "-", x["reason"])


if __name__ == "__main__":
    main()
