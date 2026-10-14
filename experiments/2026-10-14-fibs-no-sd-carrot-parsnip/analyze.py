"""Preregistered analysis (PREREGISTRATION.md §3-4) -> results.json.

Primary: gap = mean AUROC of the random_drop linear probes minus mean AUROC
of the no_sd linear probes (seeds averaged), announcement-level AUROC of the
contrast score (Parsnip = 1), 3p and 5p pooled; 95% percentile interval over
cfg.n_boot game-bootstrap draws (games resampled within arm, every probe
scored on the same draw).
  Test 1: supported if the upper bound < cfg.delta, refuted if the lower
          bound >= cfg.delta, else inconclusive.
  Test 2: supported if the lower bound > 0, refuted if the upper bound < 0,
          else inconclusive.
The positive control (every probe's fibs validation AUROC >=
cfg.pc_min_val_auroc, train_results.json) must have passed.

Secondary: the same per arm and for MLP; mean-max game accuracy; flag rates
at the alpaca 1%-FPR cutpoint. Published probes (2026-10-12) as reference.

Run:  uv run --no-sync python experiments/2026-10-14-fibs-no-sd-carrot-parsnip/analyze.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

from common import OUT, REPO, load_config


def auroc_cols(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC of every column of S (n, k) against y (ties count half)."""
    r = rankdata(S, axis=0)
    n1 = y.sum()
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def interval(v: np.ndarray) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def verdicts(lo: float, hi: float, delta: float) -> dict:
    return {"test1_gap_below_delta": "supported" if hi < delta else "refuted" if lo >= delta else "inconclusive",
            "test2_gap_above_zero": "supported" if lo > 0 else "refuted" if hi < 0 else "inconclusive"}


def load(arm: str) -> list[dict]:
    rows = list(csv.DictReader(open(OUT / f"scores_{arm}.csv")))
    for r in rows:
        r["arm"] = arm
    return rows


def main():
    cfg = load_config()
    tr = json.loads((OUT / "train_results.json").read_text())
    names = {(c, s, a): f"{c}_s{s}_{a}" for c in cfg.conditions for s in cfg.train_seeds for a in cfg.archs}
    missing = [f"{c}_s{s}/{a}" for c, s, a in names if f"{c}_s{s}/{a}" not in tr]
    if missing:
        raise ValueError(f"no training result for {missing}")
    failed = [k for k, v in tr.items() if not v["positive_control_pass"]]
    if failed:
        raise RuntimeError(f"positive control failed for {failed}: no analysis")

    rows = [r for arm in cfg.main_arms for r in load(arm)]
    cols = list(names.values()) + [f"published_{a}" for a in cfg.published_archs]
    S = np.array([[float(r[c]) for c in cols] for r in rows])
    y = np.array([r["role"] == "Parsnip" for r in rows], dtype=int)
    arm = np.array([r["arm"] for r in rows])
    gid = np.array([f"{r['arm']}/{r['game_id']}" for r in rows])
    ci = {c: j for j, c in enumerate(cols)}

    def cond_mean(A: np.ndarray, c: str, a: str) -> np.ndarray:      # A (..., k) -> (...)
        return A[..., [ci[names[(c, s, a)]] for s in cfg.train_seeds]].mean(axis=-1)

    def block(sel: np.ndarray, draws: list[np.ndarray]) -> dict:
        """Point and bootstrap AUROCs of every column on the rows ``sel``."""
        point = auroc_cols(S[sel], y[sel])
        boot = np.array([auroc_cols(S[d], y[d]) for d in draws])
        out = {"n": int(sel.sum()) if sel.dtype == bool else len(sel), "n_pos": int(y[sel].sum())}
        for a in cfg.archs:
            nd, rd = cond_mean(point, "no_sd", a), cond_mean(point, "random_drop", a)
            gap = cond_mean(boot, "random_drop", a) - cond_mean(boot, "no_sd", a)
            lo, hi = interval(gap)
            out[a] = {
                "no_sd": {"mean_auroc": float(nd), "ci95_game": interval(cond_mean(boot, "no_sd", a)),
                          "per_seed": {str(s): float(point[ci[names[("no_sd", s, a)]]]) for s in cfg.train_seeds}},
                "random_drop": {"mean_auroc": float(rd), "ci95_game": interval(cond_mean(boot, "random_drop", a)),
                                "per_seed": {str(s): float(point[ci[names[("random_drop", s, a)]]])
                                             for s in cfg.train_seeds}},
                "gap": {"value": float(rd - nd), "ci95_game": [lo, hi], **verdicts(lo, hi, cfg.delta)}}
            if f"published_{a}" in ci:
                out[a]["published_reference"] = {"auroc": float(point[ci[f"published_{a}"]]),
                                                 "ci95_game": interval(boot[:, ci[f"published_{a}"]])}
        return out

    rng = np.random.default_rng(cfg.boot_seed)
    idx_by_game = {g: np.flatnonzero(gid == g) for g in np.unique(gid)}
    games_by_arm = {a: sorted({g for g in idx_by_game if g.startswith(a + "/")}) for a in cfg.main_arms}
    draws_by_arm = {a: [] for a in cfg.main_arms}
    pooled_draws = []
    for _ in range(cfg.n_boot):
        parts = {}
        for a, gs in games_by_arm.items():
            pick = rng.choice(len(gs), len(gs), replace=True)
            parts[a] = np.concatenate([idx_by_game[gs[k]] for k in pick])
            draws_by_arm[a].append(parts[a])
        pooled_draws.append(np.concatenate(list(parts.values())))

    res = {"n_boot": cfg.n_boot, "delta": cfg.delta, "primary_arch": cfg.primary_arch,
           "positive_control": {k: v["val_auroc"] for k, v in sorted(tr.items())}}
    pooled = block(np.arange(len(y)), pooled_draws)
    res["primary"] = {"pooled": pooled[cfg.primary_arch], "n": pooled["n"], "n_pos": pooled["n_pos"],
                      "n_games": len(idx_by_game)}

    sec = {"per_arm": {}, "mlp_pooled": {a: pooled[a] for a in cfg.archs if a != cfg.primary_arch}}
    for a in cfg.main_arms:
        # per-arm draws: the arm's own part of each pooled draw, indices into the full arrays
        sel = np.flatnonzero(arm == a)
        sec["per_arm"][a] = block(sel, draws_by_arm[a])

    games = {a: {g["game_id"]: g for g in json.load(open(REPO / cfg.cp_dir / f"games_{a}.json"))["games"]}
             for a in cfg.main_arms}
    sec["mean_max"] = {}
    for a in cfg.main_arms:
        sub = [r for r in rows if r["arm"] == a]
        acc = {}
        for c in cols:
            by_game = defaultdict(lambda: defaultdict(list))
            for r in sub:
                by_game[r["game_id"]][r["speaker"]].append(float(r[c]))
            acc[c] = float(np.mean([max(d, key=lambda p: np.mean(d[p])) == games[a][g]["parsnip"]
                                    for g, d in by_game.items()]))
        sec["mean_max"][a] = {"per_probe": acc, **{
            f"{c}_{ar}_mean": float(np.mean([acc[names[(c, s, ar)]] for s in cfg.train_seeds]))
            for c in cfg.conditions for ar in cfg.archs}}

    alp = list(csv.DictReader(open(OUT / "scores_alpaca.csv")))
    cc = [r for a in cfg.cc_arms for r in load(a)]
    sec["alpaca_1pct_fpr"] = {}
    for c in cols:
        cut = float(np.quantile([float(r[c]) for r in alp], cfg.fpr_quantile))
        sec["alpaca_1pct_fpr"][c] = {
            "cutpoint": cut,
            "parsnip": float(np.mean([float(r[c]) > cut for r in rows if r["role"] == "Parsnip"])),
            "carrot": float(np.mean([float(r[c]) > cut for r in rows if r["role"] == "Carrot"])),
            "all_carrot_games": float(np.mean([float(r[c]) > cut for r in cc]))}
    res["secondary"] = sec
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")

    p = res["primary"]["pooled"]
    print(f"primary ({cfg.primary_arch}, pooled, {res['primary']['n']} announcements): "
          f"no_sd {p['no_sd']['mean_auroc']:.4f} {p['no_sd']['ci95_game']}, "
          f"random_drop {p['random_drop']['mean_auroc']:.4f} {p['random_drop']['ci95_game']}")
    print(f"  gap {p['gap']['value']:.4f} [{p['gap']['ci95_game'][0]:.4f}, {p['gap']['ci95_game'][1]:.4f}]  "
          f"Test 1: {p['gap']['test1_gap_below_delta']}  Test 2: {p['gap']['test2_gap_above_zero']}")
    for a in cfg.archs:
        g = pooled[a]["gap"]
        print(f"  {a}: gap {g['value']:.4f} {g['ci95_game']}; published {pooled[a]['published_reference']['auroc']:.4f}")


if __name__ == "__main__":
    main()
