"""Run the outside auditor with calibrated 1-7 suspicion grades.

All run parameters come from an experiment folder's config.json
(:class:`auditor.config.AuditorConfig`): where the transcripts/scores live
(`data_dir`), where outputs and `suspicion_scale.json` live (`out_dir`),
generation settings, conditions, span, and the grade legend. This module is
pure code — nothing experiment-specific is hardcoded; player names and the
round count are read from the game data.

Conditions: "none" shows no annotations; conditions in `summaries` show a
per-game detector summary after the transcript (no per-message grades);
"<cond>_avg" shows <cond>'s grades
plus each player's mean grade below the transcript; "gold" grades every Parsnip
message 7 and every Carrot message 1; judge conditions ("godmode" and any in
`judge_grades`) show a text judge's own 1-7 ratings; every other condition
name must have cutpoints in suspicion_scale.json and a `<cond>_<span>` (or
`rand_<span>` for "random") column in scores_{arm}.csv.

Run (vLLM env):
  CUDA_VISIBLE_DEVICES=... PATH=$PWD/.venv-vllm/bin:$PATH \
      .venv-vllm/bin/python -m auditor.run \
      --config experiments/<date>/config.json --arm base
Smoke: add  --games 2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from auditor.config import AuditorConfig  # noqa: E402
from auditor.prompts import (ANSWER_CUE, AUDITOR_SYSTEM, AUDITOR_USER,  # noqa: E402
                             DETECTOR_NOTE, build_transcript, parse_guess, score_rows_csv,
                             score_table)
from auditor.scale import grade, load_scale  # noqa: E402
from rollouts.generation import CLOSE_CUE, make_backend, split_think  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score_column(cond: str, span: str) -> str:
    return f"{'rand' if cond == 'random' else cond}_{span}"


def build_grade_maps(score_rows: list[dict], scale: dict,
                     conditions: list[str]) -> dict[str, dict]:
    """condition -> {(game_id, round, position) -> 1..7}. Detectors with
    per-seed-group cutpoint lists (e.g. "random") are graded per group."""
    span, cuts = scale["span"], scale["cutpoints"]
    out: dict[str, dict] = {}
    for cond in conditions:
        if cond == "none":
            continue
        col = score_column(cond, span)
        out[cond] = {
            (row["game_id"], int(row["round"]), int(row["position"])):
                grade(float(row[col]),
                      cuts[cond][int(row["seed_group"])]
                      if isinstance(cuts[cond][0], list) else cuts[cond])
            for row in score_rows}
    return out


def judge_grade_map(path: Path) -> dict:
    """(game_id, round, position) -> a text judge's own 1-7 score;
    unparseable ratings fall back to 4 (unsure). No calibration — the judge
    already speaks the auditor's scale."""
    gm = {}
    for line in open(path):
        r = json.loads(line)
        gm[(r["game_id"], r["round"], r["position"])] = r["judge_score"] or 4
    return gm


AVG = "_avg"      # condition suffix: same grades + per-player mean grade summary


def all_grade_maps(cfg: AuditorConfig, arm: str, conditions: list[str],
                   recs: list[dict], scale: dict) -> dict[str, dict]:
    """condition -> grade map for every annotated condition (probe, judge,
    gold); "none" gets no entry. "<cond>_avg" shares <cond>'s map."""
    asked = [c for c in conditions if c not in cfg.summaries and c not in cfg.raw_scores]
    conditions = list(dict.fromkeys(c.removesuffix(AVG) for c in asked))
    judges = {"godmode": str(cfg.path(cfg.out_dir) / "judge_godmode_{arm}.jsonl"),
              **cfg.judge_grades}
    probe_conds = [c for c in conditions if c not in judges and c not in ("none", "gold")]
    maps: dict[str, dict] = {}
    if probe_conds:
        with open(cfg.path(cfg.data_dir) / f"scores_{arm}.csv") as f:
            maps.update(build_grade_maps(list(csv.DictReader(f)), scale, probe_conds))
    if "gold" in conditions:
        maps["gold"] = {(r["game_id"], r["round"], r["position"]):
                        7 if r["role"] == "Parsnip" else 1 for r in recs}
    for c in conditions:
        if c in judges:
            maps[c] = judge_grade_map(cfg.path(judges[c].format(arm=arm)))
    return {c: maps[c.removesuffix(AVG)] for c in asked if c.removesuffix(AVG) in maps}


def prepare(cfg: AuditorConfig, arm: str, conditions: list[str], games: int = 0,
            per_group: int = 0) -> SimpleNamespace:
    """Load the games and build every (game, condition) auditor conversation.
    Shared by the local runner (main) and the API runner (auditor.api_run), so
    both auditors see byte-identical prompts."""
    data, out = cfg.path(cfg.data_dir), cfg.path(cfg.out_dir)
    scale = load_scale(out / "suspicion_scale.json")
    assert scale["span"] == cfg.span, (scale["span"], cfg.span)

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
    game_ids = sorted(g for g in by_game
                      if not per_group
                      or int(g.split("g")[1]) < per_group)[: games or None]

    grade_by_cond = all_grade_maps(cfg, arm, conditions, recs, scale)

    jobs = [(gid, cond) for gid in game_ids for cond in conditions]
    print(f"{len(jobs)} auditor calls ({len(game_ids)} games x {conditions})", flush=True)

    summaries = {c: json.loads(cfg.path(p.format(arm=arm)).read_text())
                 for c, p in cfg.summaries.items() if c in conditions}

    raw_scores = {c: json.loads(cfg.path(p.format(arm=arm)).read_text())
                  for c, p in cfg.raw_scores.items() if c in conditions}

    def build_conv(gid: str, cond: str) -> list[dict]:
        if cond in raw_scores:
            tool = cond in cfg.tool_conditions
            transcript = build_transcript(by_game[gid], None) + "\n\n" + score_table(raw_scores[cond][gid])
            note = cfg.raw_note + (" " + cfg.tool_note if tool else "")
            system = AUDITOR_SYSTEM.format(players=", ".join(player_names), num_rounds=num_rounds,
                                           detector_note="\n\n" + note)
            user = AUDITOR_USER.format(transcript=transcript, and_scores=" and the detector scores",
                                       example=player_names[0])
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if cond in summaries:
            transcript = build_transcript(by_game[gid], None) + "\n\n" + summaries[cond][gid]
            system = AUDITOR_SYSTEM.format(players=", ".join(player_names), num_rounds=num_rounds,
                                           detector_note="\n\n" + cfg.summary_note)
            user = AUDITOR_USER.format(transcript=transcript, and_scores=" and the detector summary",
                                       example=player_names[0])
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        transcript = build_transcript(by_game[gid], grade_by_cond.get(cond),
                                      summary=cond.endswith(AVG))
        system = AUDITOR_SYSTEM.format(
            players=", ".join(player_names), num_rounds=num_rounds,
            detector_note=DETECTOR_NOTE.format(legend=cfg.grade_legend)
            if cond != "none" else "")
        user = AUDITOR_USER.format(
            transcript=transcript,
            and_scores=" and the suspicion grades" if cond != "none" else "",
            example=player_names[0])
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    convs = [build_conv(gid, cond) for gid, cond in jobs]
    return SimpleNamespace(jobs=jobs, convs=convs, game_ids=game_ids, parsnip=parsnip,
                           player_names=player_names, raw_scores=raw_scores)


def main(argv: list[str] | None = None, default_config: str | Path | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=default_config, required=default_config is None,
                    help="path to an AuditorConfig config.json")
    ap.add_argument("--arm", default="base",
                    help="reads transcripts_/games_/scores_{arm} from data_dir")
    ap.add_argument("--backend", choices=["vllm", "hf"], default=None)
    ap.add_argument("--conditions", type=str, default=None)
    ap.add_argument("--games", type=int, default=0, help="limit games (smoke test)")
    ap.add_argument("--per-group", type=int, default=0,
                    help="use games with game_idx < N per seed group (the "
                         "balanced subset the API runs use, e.g. 10 -> 50 games)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="audits per game x condition; >1 writes "
                         "auditor_seeds_{arm}.jsonl with an auditor_seed field")
    ap.add_argument("--chunk", type=int, default=None, help="HF batch size")
    args = ap.parse_args(argv)

    cfg = AuditorConfig.load(args.config)
    data, out = cfg.path(cfg.data_dir), cfg.path(cfg.out_dir)
    conditions = args.conditions.split(",") if args.conditions else cfg.conditions
    backend_kind = args.backend or cfg.gen_backend
    smoke = bool(args.games)
    if not smoke:
        cfg.save(out)

    P = prepare(cfg, args.arm, conditions, args.games, args.per_group)
    jobs, convs, game_ids, parsnip = P.jobs, P.convs, P.game_ids, P.parsnip
    player_names, raw_scores = P.player_names, P.raw_scores
    backend = make_backend(backend_kind, cfg.model_id,
                           **({"chunk": args.chunk or cfg.gen_chunk}
                              if backend_kind == "hf" else
                              {"tensor_parallel_size": cfg.tensor_parallel_size,
                               "max_model_len": cfg.max_model_len}))
    backend.force_close_tokens = cfg.force_close_tokens

    if smoke:
        print("=== sample system prompt (last job) ===", flush=True)
        print(convs[-1][0]["content"], flush=True)
        print("=== sample user prompt (last job, first 3000 chars) ===", flush=True)
        print(convs[-1][1]["content"][:3000], flush=True)

    multi = args.seeds > 1
    out_path = out / (f"auditor_seeds_{args.arm}{'_smoke' if smoke else ''}.jsonl"
                      if multi else
                      f"auditor_{args.arm}{'_smoke' if smoke else ''}.jsonl")
    done_seeds = 0
    if multi and not smoke and out_path.exists():
        done_seeds = sum(1 for _ in open(out_path)) // len(jobs)
        print(f"resume: {done_seeds}/{args.seeds} seed passes done", flush=True)

    def follow_up(raws: list[str], idxs: list[int], seed: int) -> dict[int, str | None]:
        """Answer provenance per index: "tag" (the auditor wrote one),
        "followup" (tag completed after ANSWER_CUE; raws updated in place),
        or None (still unanswered)."""
        src = {i: "tag" if parse_guess(raws[i], player_names) else None for i in idxs}
        todo = [i for i in idxs if src[i] is None]
        if cfg.answer_followup and todo:
            conts = backend.complete([backend.render(convs[i]) + raws[i] + ANSWER_CUE
                                      for i in todo], 10, seed=seed + 30_000)
            for i, c in zip(todo, conts):
                raws[i] = raws[i] + ANSWER_CUE + c
                src[i] = "followup" if parse_guess(raws[i], player_names) else None
        return src

    totals = defaultdict(int)
    with open(out_path, "a" if done_seeds else "w") as f:
        for si in range(done_seeds, args.seeds):
            t0 = time.time()
            seed = cfg.seed + 100 * si            # matches the API seed scheme
            tool_idx = [i for i, (_, c) in enumerate(jobs) if c in cfg.tool_conditions]
            plain_idx = [i for i in range(len(jobs)) if i not in set(tool_idx)]
            raws = [""] * len(jobs)
            for i, r in zip(plain_idx, backend.generate([convs[i] for i in plain_idx],
                                                        cfg.max_new_tokens, seed=seed)):
                raws[i] = r
            source = follow_up(raws, plain_idx, seed)
            tool_logs = {}
            if tool_idx:
                from auditor.tools import run_tool_audits
                results = run_tool_audits(
                    backend, [convs[i] for i in tool_idx],
                    [{"scores.csv": score_rows_csv(raw_scores[jobs[i][1]][jobs[i][0]])} for i in tool_idx],
                    max_new_tokens=cfg.max_new_tokens, max_calls=cfg.max_tool_calls, seed=seed,
                    player_names=player_names, force_close_tokens=cfg.force_close_tokens,
                    answer_followup=cfg.answer_followup)
                for i, (r, log) in zip(tool_idx, results):
                    raws[i], tool_logs[i] = r, log
                    source[i] = (None if parse_guess(r, player_names) is None else
                                 "followup" if log.get("answer_followup") else "tag")
            retry_idx = [i for i in plain_idx if parse_guess(raws[i], player_names) is None]
            if retry_idx:
                retry_raws = backend.generate([convs[i] for i in retry_idx],
                                              cfg.max_new_tokens,
                                              seed=seed + 20_000)
                for i, raw in zip(retry_idx, retry_raws):
                    raws[i] = raw
                source.update(follow_up(raws, retry_idx, seed + 20_000))
            n_correct = defaultdict(int)
            for i, ((gid, cond), raw) in enumerate(zip(jobs, raws)):
                guess = parse_guess(raw, player_names)
                reasoning, public = split_think(raw)
                correct = guess == parsnip[gid]
                n_correct[cond] += correct
                totals[cond] += correct
                f.write(json.dumps({
                    "arm": args.arm, "game_id": gid, "condition": cond,
                    "auditor_seed": si, "guess": guess, "parsnip": parsnip[gid],
                    "correct": bool(correct), "parse_failed": guess is None,
                    "answer_source": source[i], "retried": i in retry_idx,
                    "force_closed": CLOSE_CUE in raw, "tool": tool_logs.get(i),
                    "reasoning": reasoning, "public": public}) + "\n")
            f.flush()
            print(f"seed {si}: "
                  + "  ".join(f"{c} {n_correct[c]}/{len(game_ids)}" for c in conditions)
                  + f"  ({time.time() - t0:.0f}s, {backend.n_force_closed} force-closed)",
                  flush=True)
    n_done = (args.seeds - done_seeds) * len(game_ids)
    for cond in conditions:
        print(f"{args.arm}/{cond}: {totals[cond]}/{n_done} correct this run", flush=True)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
