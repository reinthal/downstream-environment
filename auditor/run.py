"""Run the outside auditor with calibrated 1-7 suspicion grades.

All run parameters come from an experiment folder's config.json
(:class:`auditor.config.AuditorConfig`): where the transcripts/scores live
(`data_dir`), where outputs and `suspicion_scale.json` live (`out_dir`),
generation settings, conditions, span, and the grade legend. This module is
pure code — nothing experiment-specific is hardcoded; player names and the
round count are read from the game data.

Conditions: "none" shows no annotations; every other condition name must
have cutpoints in suspicion_scale.json and a `<cond>_<span>` (or
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

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from auditor.config import AuditorConfig  # noqa: E402
from auditor.prompts import (AUDITOR_SYSTEM, AUDITOR_USER, DETECTOR_NOTE,  # noqa: E402
                             build_transcript, parse_guess)
from auditor.scale import grade, load_scale  # noqa: E402
from rollouts.generation import make_backend, split_think  # noqa: E402


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


def godmode_grade_map(out_dir: Path, arm: str) -> dict:
    """(game_id, round, position) -> the god-mode judge's own 1-7 score
    (judge_godmode_{arm}.jsonl); unparseable ratings fall back to 4 (unsure).
    No calibration — the detector already speaks the auditor's scale."""
    gm = {}
    for line in open(out_dir / f"judge_godmode_{arm}.jsonl"):
        r = json.loads(line)
        gm[(r["game_id"], r["round"], r["position"])] = r["judge_score"] or 4
    return gm


def main(argv: list[str] | None = None, default_config: str | Path | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=default_config, required=default_config is None,
                    help="path to an AuditorConfig config.json")
    ap.add_argument("--arm", choices=["base", "steered"], default="base")
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

    scale = load_scale(out / "suspicion_scale.json")
    assert scale["span"] == cfg.span, (scale["span"], cfg.span)

    recs = read_jsonl(data / f"transcripts_{args.arm}.jsonl")
    games_meta = json.load(open(data / f"games_{args.arm}.json"))["games"]
    parsnip = {g["game_id"]: g["parsnip"] for g in games_meta}
    player_names = list(games_meta[0]["roles"])
    num_rounds = max(r["round"] for r in recs) + 1
    by_game = defaultdict(list)
    for r in recs:
        by_game[r["game_id"]].append(r)
    for msgs in by_game.values():
        msgs.sort(key=lambda m: (m["round"], m["position"]))
    game_ids = sorted(g for g in by_game
                      if not args.per_group
                      or int(g.split("g")[1]) < args.per_group)[: args.games or None]

    with open(data / f"scores_{args.arm}.csv") as f:
        score_rows = list(csv.DictReader(f))
    grade_by_cond = build_grade_maps(
        score_rows, scale, [c for c in conditions if c != "godmode"])
    if "godmode" in conditions:
        grade_by_cond["godmode"] = godmode_grade_map(out, args.arm)

    jobs = [(gid, cond) for gid in game_ids for cond in conditions]
    print(f"{len(jobs)} auditor calls ({len(game_ids)} games x {conditions})", flush=True)

    backend = make_backend(backend_kind, cfg.model_id,
                           **({"chunk": args.chunk or cfg.gen_chunk}
                              if backend_kind == "hf" else {}))

    def build_conv(gid: str, cond: str) -> list[dict]:
        transcript = build_transcript(by_game[gid], grade_by_cond.get(cond))
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

    totals = defaultdict(int)
    with open(out_path, "a" if done_seeds else "w") as f:
        for si in range(done_seeds, args.seeds):
            t0 = time.time()
            seed = cfg.seed + 100 * si            # matches the API seed scheme
            raws = backend.generate(convs, cfg.max_new_tokens, seed=seed)
            retry_idx = [i for i, raw in enumerate(raws)
                         if parse_guess(raw, player_names) is None]
            if retry_idx:
                retry_raws = backend.generate([convs[i] for i in retry_idx],
                                              cfg.max_new_tokens,
                                              seed=seed + 20_000)
                for i, raw in zip(retry_idx, retry_raws):
                    raws[i] = raw
            n_correct = defaultdict(int)
            for (gid, cond), raw in zip(jobs, raws):
                guess = parse_guess(raw, player_names)
                reasoning, public = split_think(raw)
                correct = guess == parsnip[gid]
                n_correct[cond] += correct
                totals[cond] += correct
                f.write(json.dumps({
                    "arm": args.arm, "game_id": gid, "condition": cond,
                    "auditor_seed": si, "guess": guess, "parsnip": parsnip[gid],
                    "correct": bool(correct), "parse_failed": guess is None,
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
