"""Turn the engine's Inspect logs into the repo's transcript files (CPU).

Adapted from experiments/2026-10-09-3p-game/extract.py: same parsing, plus
the all-Carrot arms (no Parsnip) and the preregistered exclusions — a game is
dropped when it failed, did not resolve, a player posted two announcements in
one turn, or a player was silent in ANY round. ``count_valid`` is what
play_games.py uses to decide how many replacement games to play.

  transcripts_{tag}.jsonl  one row per discussion turn of every valid game
  games_{tag}.json         per valid game: roles, parsnip (None in _cc arms),
                           speaking orders, votes, eliminated, winner
  excluded_{tag}.json      dropped games with the reason

Run:  uv run --no-sync python experiments/2026-10-12-fibs-probes/extract.py --arm 3p [--pilot]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from inspect_ai.log import read_eval_log

from common import OUT, REPO, load_config, tag, write_jsonl

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
    """The game is excluded by the preregistered rules (PREREGISTRATION.md §4)."""


def turns(sample) -> list[dict]:
    """Group a game's events into turns. A turn starts at a model call whose
    input is just [system, prompt]: the explicit reasoning call in two_call
    mode, the single acting call in single_call mode (later calls of the same
    turn are retries with more messages)."""
    out = []
    for e in sample.events:
        if e.event == "model" and len(e.input) == 2 and (
                not out or e.input[1].text != out[-1]["first"].input[1].text):
            # a re-sampled single_call turn (thinking cut off) repeats the same
            # two messages verbatim and stays in its turn; a new turn never does
            out.append({"first": e, "calls": [e], "tools": []})
        elif e.event == "model":
            out[-1]["calls"].append(e)
        elif e.event == "tool":
            out[-1]["tools"].append(e)
    return out


def reasoning_blocks(message) -> str:
    content = message.content
    if not isinstance(content, list):
        return ""
    return "\n".join(c.reasoning for c in content if type(c).__name__ == "ContentReasoning")


def recorded_tools(event) -> list[dict] | None:
    """The tool schemas exactly as sent to the provider, when Inspect kept
    the raw request (it keeps it for the first few calls of a game only)."""
    if event.call is None or "tools" not in event.call.request:
        return None
    return event.call.request["tools"]


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
    parsnips = [name for name, role in roles.items() if role == "Parsnip"]
    if game_cfg.include_parsnip and len(parsnips) != game_cfg.parsnip.count:
        raise ValueError(f"{len(parsnips)} Parsnips in the summary, config says {game_cfg.parsnip.count}")
    if not game_cfg.include_parsnip and parsnips:
        raise ValueError(f"all-Carrot arm but the summary lists Parsnips {parsnips}")

    n, rounds = game_cfg.num_players, game_cfg.num_rounds_discussion
    single = game_cfg.turn_mode == "single_call"
    # the discussion tools' schema as the provider rendered it: identical for
    # every discussion call of the game, recorded for the first ones only
    canonical: list[dict] | None = None
    orders: list[list[int]] = [[] for _ in range(rounds)]
    rows, votes = [], {}
    for t in turns(sample):
        e = t["first"]
        if single != bool(e.tools):
            raise ValueError(f"turn_mode={game_cfg.turn_mode} but the first call of a turn "
                             f"{'has' if e.tools else 'has no'} tools")
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
        (pos,) = [i for i, (_, _, t_) in enumerate(lines) if t_.startswith("YOUR TURN")]
        if sorted(order) != list(range(n)) or names[order[pos]] != speaker:
            raise ValueError(f"bad speaking order {lines} for speaker {speaker}")
        if orders[rnd - 1] not in ([], order) or pos != len([r for r in rows if r["round"] == rnd - 1]):
            raise ValueError(f"speaking order changed within round {rnd}")
        orders[rnd - 1] = order
        made = [x for x in t["tools"] if x.function == "MakeAnnouncementTool"
                and str(x.result).startswith(ANNOUNCED)]
        if len(made) > 1:
            raise GameFailed(f"{speaker} made {len(made)} announcements in one turn")
        # the acting call: in two_call mode the second call; in single_call mode
        # the first call, or the retry whose tool call succeeded
        acting = t["calls"][0] if single else t["calls"][1]
        if made:
            with_call = [c for c in t["calls"] if any(tc.id == made[0].id for tc in (c.output.message.tool_calls or []))]
            if len(with_call) != 1:
                raise ValueError(f"{speaker}: {len(with_call)} model calls carry the successful announcement")
            acting = with_call[0]
        tool_call = next((tc for tc in (acting.output.message.tool_calls or []) if tc.id == made[0].id), None) if made else None
        sent = recorded_tools(acting)
        if sent is not None:
            if canonical is not None and sent != canonical:
                raise ValueError("the discussion tools' schema changed within the game")
            canonical = sent
        if canonical is None:
            raise ValueError("no recorded tools request before the first unrecorded call")
        if [x["function"]["name"] for x in canonical] != [x.name for x in acting.tools]:
            raise ValueError(f"tool names {[x.name for x in acting.tools]} differ from the recorded schema")
        rows.append({
            "round": rnd - 1, "position": pos, "speaker": speaker,
            "speaker_index": names.index(speaker), "role": roles[speaker],
            "turn_mode": game_cfg.turn_mode,
            "n_calls": len(t["calls"]),  # > 1: re-sampled (thinking cut off) or tool-call retries
            "input_messages": [{"role": m.role, "content": m.text} for m in acting.input],
            "tools": canonical,
            # single_call: the model's own thinking before the tool call; two_call: the explicit reply
            "reasoning": reasoning_blocks(acting.output.message) if single else e.output.message.text,
            "reasoning_stop": acting.output.stop_reason if single else e.output.stop_reason,
            "reasoning_tokens": acting.output.usage.reasoning_tokens if acting.output.usage else None,
            "acting_text": acting.output.message.text,
            "tool_call": {"name": tool_call.function, "arguments": dict(tool_call.arguments)} if tool_call else None,
            "tool_result": str(made[0].result) if made else None,
            "announcement": made[0].arguments["message"] if made else "",
            "provider": e.call.response.get("provider") if e.call is not None else None,
        })
    if len(rows) != n * rounds:
        raise ValueError(f"{len(rows)} discussion turns, expected {n * rounds}")
    if sorted(votes) != sorted(names):
        raise ValueError(f"votes from {sorted(votes)}, expected every player")
    silent = sorted({r["speaker"] for r in rows if not r["announcement"]})
    if silent:
        raise GameFailed(f"silent in a round: {', '.join(silent)}")
    meta = {"roles": roles, "parsnip": parsnips[0] if parsnips else None, "orders": orders,
            "votes": votes,
            "eliminated": [name for _, name, _, alive in players if alive == "ELIMINATED"],
            "winner": winner.group(1)}
    return meta, rows


def game_logs(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("game_*.eval"))


def count_valid(paths: list[Path], game_cfg: GameConfig) -> int:
    ok = 0
    for path in paths:
        try:
            parse_game(read_eval_log(str(path), resolve_attachments=True), game_cfg)
            ok += 1
        except GameFailed:
            pass
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    t = tag(args.arm, args.pilot)
    game_cfg = GameConfig.from_yaml(REPO / cfg.game_configs[args.arm])
    seed = cfg.pilot_seed if args.pilot else cfg.arm_seeds[args.arm]

    games, records, excluded = [], [], []
    for path in game_logs(OUT / f"inspect_logs_{t}"):
        idx = int(path.stem.removeprefix("game_"))
        ident = {"game_id": f"g{idx:03d}", "seed_group": 0, "game_idx": idx, "game_seed": seed + idx}
        try:
            meta, rows = parse_game(read_eval_log(str(path), resolve_attachments=True), game_cfg)
        except GameFailed as exc:
            excluded.append({**ident, "reason": str(exc)})
            continue
        games.append({**ident, **meta})
        records += [{**ident, **r, "arm": t} for r in rows]

    write_jsonl(OUT / f"transcripts_{t}.jsonl", records)
    (OUT / f"games_{t}.json").write_text(json.dumps(
        {"arm": t, "game_config": cfg.game_configs[args.arm], "games": games}, indent=1) + "\n")
    (OUT / f"excluded_{t}.json").write_text(json.dumps(excluded, indent=1) + "\n")
    print(f"{len(games)} valid games ({len(records)} turns), {len(excluded)} excluded")
    for x in excluded:
        print("  excluded", x["game_id"], "-", x["reason"])


if __name__ == "__main__":
    main()
