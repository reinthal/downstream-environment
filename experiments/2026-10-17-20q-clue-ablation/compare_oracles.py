"""EXPLORATORY (researcher-requested after the conclusion): the posterior
entropy analysis with a second scorer — Qwen3.5-9B on local vLLM (the GPU
box; not OpenRouter), one prompt per (question|clue, concept) pair, p(yes)
from the first-token logprobs — compared with Jev.

    # 1. on the GPU box (.venv-vllm): score_vllm.py -> qwen_vllm_scores.jsonl
    # 2. here (env venv), no model calls:
    uv run --project environments/twenty_questions python experiments/2026-10-17-20q-clue-ablation/compare_oracles.py

Writes entropy_qwen_<cell>.json, compare_oracles.json.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import statistics

from common import ENV, OUT, REPO, AblationConfig, cell, slug

from twenty_questions.concepts import ConceptSet
from twenty_questions.entropy import EntropyConfig, analyse, game_observations

QWEN_CACHE = OUT / "qwen_vllm_scores.jsonl"
JEV_CACHE = OUT / "jev_scores.jsonl"
QWEN = dict(backend="precomputed", model="Qwen/Qwen3.5-9B", include_clues=True)


def cell_games(model, arm):
    return json.loads((OUT / f"games_{cell(model, arm, False)}.json").read_text())["games"]


def bits_by_kind(res, games):
    """soft and hard bits per observation kind; requested clues kept apart."""
    out = {"question": [], "clue": [], "question_hard": [], "clue_hard": []}
    for game, g in zip(games, res["games"]):
        requested = {e["turn"] for e in game["exchanges"] if e["kind"] == "clue_request"}
        for prev, s in zip(g["steps"], g["steps"][1:]):
            if s["kind"] == "clue" and s["turn"] in requested:
                continue
            out[s["kind"]].append(s["bits"])
            if prev["n_consistent"] and s["n_consistent"]:
                out[s["kind"] + "_hard"].append(math.log2(prev["n_consistent"] / s["n_consistent"]))
    return out


def jev_p(concepts):
    jev = {}
    for line in JEV_CACHE.read_text().splitlines():
        r = json.loads(line)
        jev[(r.get("kind", "question"), r["question"], r["concept"])] = r["p_yes"]
    return jev


def main():
    cfg = AblationConfig.load()
    base = EntropyConfig.from_yaml(ENV / "configs" / "entropy.yaml")
    qcfg = dataclasses.replace(base, **QWEN)
    every = [g for m in cfg.models for a in cfg.arms for g in cell_games(m, a)]
    asyncio.run(analyse(qcfg, every, QWEN_CACHE))     # raises if any pair is missing or unanswered

    # ── pairwise agreement with Jev (every pair both scored) ──
    concepts = ConceptSet.load(base.concept_path())
    jev = jev_p(concepts)
    qwen = {}
    for line in QWEN_CACHE.read_text().splitlines():
        r = json.loads(line)
        qwen[(r["kind"], r["text"], r["concept"])] = r["p_yes"]
    agree = {}
    for kind in ("question", "clue"):
        keys = [k for k in qwen if k[0] == kind and k in jev]
        a = [qwen[k] for k in keys]; b = [jev[k] for k in keys]
        ma, mb = statistics.mean(a), statistics.mean(b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        r = cov / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        agree[kind] = {"n_pairs": len(keys), "pearson_r": r,
                       "hard_call_agreement": sum((x > 0.5) == (y > 0.5) for x, y in zip(a, b)) / len(keys),
                       "mean_p_yes_qwen": ma, "mean_p_yes_jev": mb}

    # ── contradicting the suspect's answer for the true secret ──
    contra = {"qwen": 0, "jev": 0, "n": 0}
    for g in every:
        for e in g["exchanges"]:
            if e["kind"] != "question":
                continue
            yes = e["answer"] == " yes"
            contra["n"] += 1
            contra["qwen"] += (qwen[("question", e["text"], g["secret"])] > 0.5) != yes
            contra["jev"] += (jev[("question", e["text"], g["secret"])] > 0.5) != yes

    # ── per cell: bits per question / clue under both scorers ──
    jcfg = dataclasses.replace(base, include_clues=True)
    cells = []
    for m in cfg.models:
        for a in cfg.arms:
            games = cell_games(m, a)
            rq = asyncio.run(analyse(qcfg, games, QWEN_CACHE))
            (OUT / f"entropy_qwen_{cell(m, a, False)}.json").write_text(json.dumps(rq, indent=1) + "\n")
            rj = asyncio.run(analyse(jcfg, games, JEV_CACHE))
            bq, bj = bits_by_kind(rq, games), bits_by_kind(rj, games)
            mean = lambda xs: statistics.mean(xs) if xs else None
            cells.append({"model": m, "arm": a,
                          **{f"qwen_{k}": mean(v) for k, v in bq.items()},
                          **{f"jev_{k}": mean(v) for k, v in bj.items()},
                          "qwen_final_H": mean([g["steps"][-1]["H"] for g in rq["games"]]),
                          "jev_final_H": mean([g["steps"][-1]["H"] for g in rj["games"]]),
                          "qwen_secret_consistent": sum(bool(g["steps"][-1].get("secret_consistent")) for g in rq["games"]),
                          "jev_secret_consistent": sum(bool(g["steps"][-1].get("secret_consistent")) for g in rj["games"])})
    pooled = {}
    for scorer in ("qwen", "jev"):
        for a in cfg.arms:
            for k in ("question", "clue", "question_hard", "clue_hard"):
                vals = []
                for m in cfg.models:
                    games = cell_games(m, a)
                    path = OUT / (f"entropy_qwen_{cell(m, a, False)}.json")
                    res = json.loads(path.read_text()) if scorer == "qwen" else asyncio.run(analyse(jcfg, games, JEV_CACHE))
                    vals += bits_by_kind(res, games)[k]
                pooled[f"{scorer}_{a}_{k}"] = {"n": len(vals), "mean": statistics.mean(vals) if vals else None,
                                               "median": statistics.median(vals) if vals else None}
    result = {"qwen_config": dataclasses.asdict(qcfg), "agreement": agree, "contradicts_suspect": contra,
              "cells": cells, "pooled": pooled}
    (OUT / "compare_oracles.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(agree, indent=1))
    print(f"contradicts the suspect's answer for the true secret: Qwen {contra['qwen']}/{contra['n']}, "
          f"Jev {contra['jev']}/{contra['n']}")
    print(f"{'model':20} {'arm':9} {'q bits qwen/jev':>16} {'clue bits qwen/jev':>19} {'hard q qwen/jev':>16} {'hard clue qwen/jev':>19} {'final H qwen/jev':>17}")
    f = lambda x: "  -  " if x is None else f"{x:5.2f}"
    for c in cells:
        print(f"{slug(c['model']):20} {c['arm']:9} {f(c['qwen_question'])}/{f(c['jev_question']):>9} "
              f"{f(c['qwen_clue'])}/{f(c['jev_clue']):>12} {f(c['qwen_question_hard'])}/{f(c['jev_question_hard']):>9} "
              f"{f(c['qwen_clue_hard'])}/{f(c['jev_clue_hard']):>12} {f(c['qwen_final_H'])}/{f(c['jev_final_H']):>10}")
    for k, v in pooled.items():
        print(f"pooled {k:28} n={v['n']:4} mean={v['mean'] if v['mean'] is None else round(v['mean'], 3)} "
              f"median={v['median'] if v['median'] is None else round(v['median'], 3)}")


if __name__ == "__main__":
    main()
