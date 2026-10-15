"""Preregistered analysis for the AIS-404 pre-onset ablation
(PREREGISTRATION.md §4), unmodified by results.

Per condition: the median probe score (per architecture) with a 95%
task-clustered bootstrap CI. Primary (linear): does the ``swap`` median shift
to the honest median? read with the control-framing pattern per the §2
decision procedure. mlp corroborating; efc/axial reported when present.

Conditions are re-collected locally (``scores_main_<condition>.jsonl``):
  honest  := orig_honest        pre_onset := orig_preonset
  swap, ctrl_neutral, ctrl_good, ctrl_evil
``scores_main`` (the Modal main run) is loaded only for the cross-pipeline
agreement check and the post_onset reference band.

Run (.venv):  uv run --no-sync python experiments/2026-10-14-preonset-ablation/analyze.py --tag main
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C  # noqa: E402  (run from the experiment dir or via -m)

BASELINE = {"honest": "orig_honest", "pre_onset": "orig_preonset"}
ORDER = ["honest", "pre_onset", "swap", "ctrl_neutral", "ctrl_good", "ctrl_evil"]


def task_of(uid: str) -> str:
    return uid.split("-")[0]


def load_condition(tag: str, condition: str) -> list[dict]:
    path = C.OUT / f"scores_{tag}_{condition}.jsonl"
    return [r for r in C.read_jsonl(path) if not r["dropped"]]


def clustered_median_ci(values, clusters, n_boot, seed):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    ids = np.unique(clusters)
    members = {c: np.nonzero(clusters == c)[0] for c in ids}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([members[c] for c in draw])
        out.append(np.median(values[idx]))
    return float(np.median(values)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def clustered_gap_ci(va, ta, vb, tb, n_boot, seed):
    """median(A) - median(B), task-clustered over the union of tasks; within a
    resampled task each condition uses its own rows (A and B are different
    trajectories, shared only by task)."""
    rng = np.random.default_rng(seed)
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    ma = {c: np.nonzero(ta == c)[0] for c in np.unique(ta)}
    mb = {c: np.nonzero(tb == c)[0] for c in np.unique(tb)}
    ids = np.array(sorted(set(ma) & set(mb)))
    point = float(np.median(va) - np.median(vb))
    out = []
    for _ in range(n_boot):
        draw = rng.choice(ids, size=len(ids), replace=True)
        ia = np.concatenate([ma[c] for c in draw])
        ib = np.concatenate([mb[c] for c in draw])
        out.append(np.median(va[ia]) - np.median(vb[ib]))
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--archs", default=",".join(C.ARCHS_READY))
    args = ap.parse_args()
    archs = args.archs.split(",")
    cfg = C.CFG

    data = {}
    for cond in ORDER:
        src = BASELINE.get(cond, cond)
        data[cond] = load_condition(args.tag, src)

    res = {"tag": args.tag, "archs": archs, "n_boot": cfg.n_boot,
           "n": {c: len(rows) for c, rows in data.items()}, "by_arch": {}}

    for arch in archs:
        key = f"{arch}_score"
        per_cond = {}
        for cond in ORDER:
            rows = data[cond]
            v = [r[key] for r in rows]
            t = np.array([task_of(r["uid"]) for r in rows])
            m, lo, hi = clustered_median_ci(v, t, cfg.n_boot, cfg.boot_seed)
            per_cond[cond] = {"n": len(v), "median": m, "ci95": [lo, hi],
                              "q10": float(np.quantile(v, 0.1)), "q90": float(np.quantile(v, 0.9))}

        def vt(cond):
            rows = data[cond]
            return [r[key] for r in rows], np.array([task_of(r["uid"]) for r in rows])

        vh, th = vt("honest"); vp, tp = vt("pre_onset"); vs, ts = vt("swap")
        gap = {}
        gp = clustered_gap_ci(vs, ts, vh, th, cfg.n_boot, cfg.boot_seed)
        gap["swap_minus_honest"] = {"point": gp[0], "ci95": [gp[1], gp[2]]}
        gp = clustered_gap_ci(vs, ts, vp, tp, cfg.n_boot, cfg.boot_seed)
        gap["swap_minus_preonset"] = {"point": gp[0], "ci95": [gp[1], gp[2]]}
        ve, te = vt("ctrl_evil"); vg, tg = vt("ctrl_good"); vn, tn = vt("ctrl_neutral")
        for name, (va, ta, vb, tb) in {
            "evil_minus_good": (ve, te, vg, tg),
            "evil_minus_honest": (ve, te, vh, th),
            "evil_minus_neutral": (ve, te, vn, tn),
            "good_minus_honest": (vg, tg, vh, th),
            "neutral_minus_honest": (vn, tn, vh, th),
        }.items():
            gp = clustered_gap_ci(va, ta, vb, tb, cfg.n_boot, cfg.boot_seed)
            gap[name] = {"point": gp[0], "ci95": [gp[1], gp[2]]}

        # swap retained fraction (secondary): numerator bootstrapped, denominator fixed
        orig_gap = per_cond["pre_onset"]["median"] - per_cond["honest"]["median"]
        retained = {"orig_gap": orig_gap,
                    "point": (per_cond["swap"]["median"] - per_cond["honest"]["median"]) / orig_gap,
                    "ci95": [gap["swap_minus_honest"]["ci95"][0] / orig_gap,
                             gap["swap_minus_honest"]["ci95"][1] / orig_gap]}

        res["by_arch"][arch] = {"per_condition": per_cond, "gaps": gap, "retained_fraction": retained}

    # confirmatory decision (linear), §2: swap shifts to honest AND the control pattern
    lin = res["by_arch"]["linear"]
    swap_ci = lin["per_condition"]["swap"]["ci95"]
    hon_med = lin["per_condition"]["honest"]["median"]
    pre_med = lin["per_condition"]["pre_onset"]["median"]
    swap_at_honest = swap_ci[0] <= hon_med <= swap_ci[1] and not (swap_ci[0] <= pre_med <= swap_ci[1])
    evil_elevated = lin["gaps"]["evil_minus_good"]["ci95"][0] > 0 and lin["gaps"]["evil_minus_honest"]["ci95"][0] > 0
    controls_congruent = (lin["gaps"]["evil_minus_honest"]["ci95"][0] <= 0 and
                          lin["gaps"]["good_minus_honest"]["ci95"][0] <= 0 and
                          lin["gaps"]["neutral_minus_honest"]["ci95"][0] <= 0)
    res["decision"] = {
        "swap_shifts_to_honest": bool(swap_at_honest),
        "evil_elevated_vs_good_and_honest": bool(evil_elevated),
        "controls_all_near_honest": bool(controls_congruent),
        "abandon_H1": bool(swap_at_honest and controls_congruent and not evil_elevated),
        "confidence_in_H1_rises": bool(evil_elevated),
        "note": "conclusion sentences deferred to post-experiment (researcher)",
    }

    (C.OUT / f"results_{args.tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"n": res["n"], "linear_medians": {c: round(v["median"], 4)
                      for c, v in res["by_arch"]["linear"]["per_condition"].items()},
                      "decision": res["decision"]}, indent=1))


if __name__ == "__main__":
    main()
