"""EXPLORATORY: is the role axis there when the Parsnip does not lie?

Carrot-Parsnip PC1 (contrast feature, pooled 3p + 5p, Parsnip end positive,
as in pca.py) per layer, with the Parsnip announcements split by the oracle
judge's honesty grade (``cfg.src_dir``/judge_{arm}.jsonl, 2026-10-12: local
Qwen3.8-27B, role + private reasoning visible, Apollo 1-7 scale): honest 1-2,
ambiguous 3-5, lie 6-7. Position on the axis: 0 = Carrot mean, 1 = Parsnip
mean.

  results_judge.json

Run:  uv run --no-sync python experiments/2026-10-16-random-direction-pca/judge_split.py
"""
from __future__ import annotations

import json

import numpy as np

from pca import CACHE, OUT, REPO, auroc_cols, feature, load_config

GROUPS = {"honest": (1, 2), "ambiguous": (3, 5), "lie": (6, 7)}


def main():
    cfg = load_config()
    info = json.loads((CACHE / "index.json").read_text())
    grade = {}
    for a in cfg.main_arms:
        for line in open(REPO / cfg.src_dir / f"judge_{a}.jsonl"):
            r = json.loads(line)
            grade[a, r["game_id"], r["round"], r["position"]] = r["grade"]
    meta = [(a, m) for a in cfg.main_arms for m in info["sets"][a]]
    g = np.array([grade[a, m["game_id"], m["round"], m["position"]] for a, m in meta])
    par = np.array([m["role"] == "Parsnip" for _, m in meta])
    grp = {"Carrot": ~par, **{f"Parsnip {k}": par & (g >= lo) & (g <= hi) for k, (lo, hi) in GROUPS.items()}}
    if sum(int(v.sum()) for v in grp.values()) != len(g):
        raise ValueError("a grade outside 1-7")
    xs = {a: np.load(CACHE / f"{a}.npy", mmap_mode="r") for a in cfg.main_arms}

    def A(s, pos, neg):
        sel = pos | neg
        return float(auroc_cols(s[sel][:, None], pos[sel])[0])

    res = {"n": {k: int(v.sum()) for k, v in grp.items()}, "by_layer": {}}
    print(res["n"])
    for li, L in enumerate(info["layers"]):
        F = np.concatenate([feature(xs[a], "contrast", li, info["prefills"], info["lie_prefill"]) for a in cfg.main_arms])
        d = F[par].mean(0) - F[~par].mean(0)
        v = np.linalg.svd(F - F.mean(0), full_matrices=False)[2][0]
        v = v if v @ d > 0 else -v
        s = F @ v
        t = (s - s[~par].mean()) / (s[par].mean() - s[~par].mean())
        res["by_layer"][str(L)] = {
            "axis_position_mean": {k: float(t[m].mean()) for k, m in grp.items()},
            "auroc_honest_parsnip_vs_carrot": A(s, grp["Parsnip honest"], ~par),
            "auroc_lie_vs_honest_parsnip": A(s, grp["Parsnip lie"], grp["Parsnip honest"]),
            "auroc_lie_vs_ambiguous_parsnip": A(s, grp["Parsnip lie"], grp["Parsnip ambiguous"]),
        }
        r = res["by_layer"][str(L)]
        print(f"L{L}: " + "  ".join(f"{k}={x:.2f}" for k, x in r["axis_position_mean"].items()) +
              f" | honest P vs C {r['auroc_honest_parsnip_vs_carrot']:.3f}  lie vs honest P "
              f"{r['auroc_lie_vs_honest_parsnip']:.3f}  lie vs ambiguous P {r['auroc_lie_vs_ambiguous_parsnip']:.3f}")
    (OUT / "results_judge.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
