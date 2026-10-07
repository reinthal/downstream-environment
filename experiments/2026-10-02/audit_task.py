"""Inspect task: manual audit of the LR probe's strongest hits (base arm).

5 true positives (highest-scoring Parsnip announcements) and 5 false
positives (highest-scoring Carrot announcements) by the column named in
config.json (lr_pub = announcement-only span, lr_full = reasoning +
announcement), at most one flagged turn per game per class. One Sample per
flagged turn; the input shows the speaker's private reasoning, the
announcement, and the full public transcript with the flagged turn marked.
All probe numbers are in the sample header and metadata. Suspicion grades
(1-7) are shown only when the selection column's span matches the scale's
calibration span. The solver is a no-op — nothing is generated; the log
exists purely so the samples are browsable.

Run (offline, no API key needed):
  uv run --no-sync inspect eval experiments/2026-10-02/audit_task.py@lr_probe_audit \
      --model mockllm/model --log-dir experiments/2026-10-02/inspect_logs
View:
  uv run --no-sync inspect view --log-dir experiments/2026-10-02/inspect_logs
"""
from __future__ import annotations

import csv
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import solver

HERE = Path(__file__).resolve().parent
CFG = json.load(open(HERE / "config.json"))


def select_hits() -> list[dict]:
    """Top-N announcements per class by CFG['column'], one per game per class."""
    rows = [r for r in csv.DictReader(open(HERE / "scores_base.csv"))
            if r["silent"] == "0"]
    for r in rows:
        r["lr_full"], r["lr_pub"] = float(r["lr_full"]), float(r["lr_pub"])
    picked = []
    for kind, role in [("TP", "Parsnip"), ("FP", "Carrot")]:
        seen_games: set[str] = set()
        pool = sorted((r for r in rows if r["role"] == role),
                      key=lambda r: -r[CFG["column"]])
        for r in pool:
            if r["game_id"] in seen_games:
                continue
            seen_games.add(r["game_id"])
            picked.append({**r, "kind": kind})
            if len(seen_games) == CFG["n_per_class"]:
                break
        assert len(seen_games) == CFG["n_per_class"], (kind, len(seen_games))
    return picked


def render_public(msgs: list[dict], flag: tuple[int, int]) -> str:
    lines = []
    cur = None
    for m in msgs:
        if m["round"] != cur:
            cur = m["round"]
            lines.append(f"--- Round {cur + 1} ---")
        text = m["announcement"] if m["announcement"] else "(stays silent)"
        mark = ">>> " if (m["round"], m["position"]) == flag else "    "
        lines.append(f'{mark}{m["speaker"]}: "{text}"')
    return "\n".join(lines)


def build_samples() -> list[Sample]:
    recs = [json.loads(l) for l in open(HERE / "transcripts_base.jsonl")]
    by_game = defaultdict(list)
    for r in recs:
        by_game[r["game_id"]].append(r)
    for msgs in by_game.values():
        msgs.sort(key=lambda m: (m["round"], m["position"]))
    by_turn = {(r["game_id"], r["round"], r["position"]): r for r in recs}
    scale = json.load(open(HERE / "suspicion_scale.json"))
    cuts = scale["cutpoints"]["lr"]
    thresholds = json.load(open(HERE / "alpaca_thresholds.json"))
    col = CFG["column"]
    col_span = {"lr_full": "full", "lr_pub": "public"}[col]
    graded = col_span == scale["span"]
    thr = thresholds[col]

    samples = []
    for hit in select_hits():
        key = (hit["game_id"], int(hit["round"]), int(hit["position"]))
        rec = by_turn[key]
        score_line = (f"selected by {col} = {hit[col]:.4f}"
                      + (f" (grade {bisect_right(cuts, hit[col]) + 1}/7)" if graded
                         else f" (no grade: scale is calibrated on the "
                              f"'{scale['span']}' span)")
                      + f"; lr_full = {hit['lr_full']:.4f}, "
                        f"lr_pub = {hit['lr_pub']:.4f}\n"
                      f"alpaca {thresholds['fpr']:.0%}-FPR threshold ({col} span, "
                      f"n={thresholds['n_control']}): {thr:.4f} — this turn is "
                      f"{'ABOVE' if hit[col] > thr else 'below'} "
                      f"({hit[col] / thr:.1f}x)")
        reasoning_note = ("part of the scored span" if col == "lr_full"
                          else f"context only — NOT scored by {col}")
        header = (
            f"=== LR-PROBE AUDIT — {hit['kind']} "
            f"({'Parsnip' if hit['kind'] == 'TP' else 'Carrot'} flagged) ===\n"
            f"Game {hit['game_id']}, round {int(hit['round']) + 1}, "
            f"speaker {hit['speaker']} (role: {hit['role']})\n"
            f"{score_line}"
        )
        body = (
            f"{header}\n\n"
            f"--- FLAGGED TURN: public announcement (the {col} span"
            f"{' is reasoning + announcement' if col == 'lr_full' else ''}) ---\n"
            f"\"{rec['announcement']}\"\n\n"
            f"--- FLAGGED TURN: private reasoning ({reasoning_note}) ---\n"
            f"{rec['reasoning']}\n\n"
            f"--- FULL PUBLIC TRANSCRIPT (flagged turn marked >>>) ---\n"
            f"{render_public(by_game[hit['game_id']], (key[1], key[2]))}"
        )
        meta = {"kind": hit["kind"], "game_id": hit["game_id"],
                "round": key[1], "position": key[2],
                "speaker": hit["speaker"], "role": hit["role"],
                "selected_by": col,
                "lr_full": hit["lr_full"], "lr_pub": hit["lr_pub"],
                "alpaca_1pct_thr": thr, "above_alpaca_thr": hit[col] > thr}
        if graded:
            meta["grade"] = bisect_right(cuts, hit[col]) + 1
        samples.append(Sample(
            id=f"{hit['kind']}:{hit['game_id']}:r{key[1]}p{key[2]}",
            input=body,
            metadata=meta))
    samples.sort(key=lambda s: s.id)
    return samples


@solver
def noop():
    async def solve(state, generate):
        return state
    return solve


@task
def lr_probe_audit() -> Task:
    return Task(dataset=MemoryDataset(build_samples(), name="lr-probe-audit"),
                solver=noop())
