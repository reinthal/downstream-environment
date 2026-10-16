"""Calibrate the detectors on the calibration games only (CPU, .venv).

  stats        per message kind, mean / s.d. of l32, l48, rand (and each
               mix's z) over the calibration messages of that kind — the
               affine rescale to layer-48 units for that kind
  random dir.  cfg.n_random_candidates unit directions in layer 48's
               standardized space (probe mu / sd), seed cfg.random_seed;
               kept: the one closest to uninformative on calibration
               announcements, |AUROC - 0.5| + |top-1 - 0.2| smallest
               (single late-layer random directions range 0.05-0.93,
               2026-10-12, so an unchecked draw could be a hidden probe)
  mix alpha    per cfg.mix_targets: alpha on a 0.005 grid whose probe-alone
               top-1 (calibration announcements) is closest to the target
  pools        layer-48 calibration scores by (role, kind) for oracle / inv

Also reports every detector's message AUROC and probe-alone top-1 on the
calibration games and on the passive-replay games (top-1 = the player with
the highest mean score over their announcements is the Parsnip).

Output: calibration.json, random_direction.npy, calibration_report.json

Run:  .venv/bin/python experiments/2026-10-16-auditor-uplift/calibrate.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from common import OUT, REPO, load_config, read_jsonl
from detectors import KIND

from linear_probes.probes import LinearProbe  # noqa: E402


def load(cfg):
    rows, l32, l48, act = [], [], [], []
    for k in range(cfg.collect_shards):
        z = np.load(OUT / f"activations_shard{k}.npz")
        rows += read_jsonl(OUT / f"rows_shard{k}.jsonl")
        l32.append(z["l32"]), l48.append(z["l48"]), act.append(z["act48"].astype(np.float32))
    return rows, np.concatenate(l32), np.concatenate(l48), np.concatenate(act)


def top1(rows, s, mask) -> float:
    per, parsnip = defaultdict(lambda: defaultdict(list)), {}
    for r, v, m in zip(rows, s, mask):
        if m:
            per[(r["set"], r["game_id"])][r["speaker"]].append(v)
        if r["role"] == "Parsnip":
            parsnip[(r["set"], r["game_id"])] = r["speaker"]
    return float(np.mean([max(p, key=lambda k: np.mean(p[k])) == parsnip[g] for g, p in per.items()]))


def auroc(rows, s, mask) -> float:
    y = np.array([r["role"] == "Parsnip" for r in rows])[mask]
    return float(roc_auc_score(y, np.asarray(s)[mask]))


def main():
    cfg = load_config()
    rows, l32, l48, act = load(cfg)
    calib = np.array([r["part"] == "calib" for r in rows])
    ann = np.array([r["kind"] == "announcement" for r in rows])
    p48 = LinearProbe.load(str(REPO / cfg.probe_paths["l48"]))
    Z = (act - p48.mu) / p48.sd

    rng = np.random.default_rng(cfg.random_seed)
    U = rng.standard_normal((cfg.n_random_candidates, Z.shape[1]))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    cand = []
    for k, u in enumerate(U):
        s = Z @ u
        a, t = auroc(rows, s, calib & ann), top1(rows, s, calib & ann)
        cand.append((abs(a - 0.5) + abs(t - 0.2), k, a, t))
    _, kbest, a_best, t_best = min(cand)
    u = U[kbest]
    rand = Z @ u
    np.save(OUT / "random_direction.npy", u)

    kinds = np.array([KIND[r["kind"]] for r in rows])

    def ms(x, kind):
        m = calib & (kinds == kind)
        return [float(np.mean(x[m])), float(np.std(x[m]))]

    def zk(x, name):                       # per-kind standardization with the stored stats
        out = np.empty_like(x)
        for kind in ("announcement", "answer"):
            m, s = stats[kind][name]
            out[kinds == kind] = (x[kinds == kind] - m) / s
        return out
    stats = {kind: {"l32": ms(l32, kind), "l48": ms(l48, kind), "rand": ms(rand, kind)}
             for kind in ("announcement", "answer")}
    z48, zr = zk(l48, "l48"), zk(rand, "rand")

    alphas, grid = {}, np.round(np.arange(0, 1.0001, 0.005), 3)
    curve = {float(a): top1(rows, a * z48 + (1 - a) * zr, calib & ann) for a in grid}
    for name, target in cfg.mix_targets.items():
        best = min(abs(t - target) for t in curve.values())
        hits = [a for a, t in curve.items() if abs(t - target) == best]
        alphas[name] = float(np.median(hits))
        for kind in ("announcement", "answer"):
            stats[kind][f"{name}_z"] = ms(alphas[name] * z48 + (1 - alphas[name]) * zr, kind)

    pools = {role: {kind: [float(v) for r, v, c in zip(rows, l48, calib)
                           if c and r["role"] == role and KIND[r["kind"]] == kind]
                    for kind in ("announcement", "answer")} for role in ("Parsnip", "Carrot")}
    cal = {"stats": stats, "mix_alpha": alphas, "random_candidate": int(kbest), "pools": pools,
           "n_calib_messages": int(calib.sum()), "calib_sets": cfg.calib_sets, "calib_games": cfg.calib_games}
    (OUT / "calibration.json").write_text(json.dumps(cal) + "\n")

    # report, through the same scoring code the games use
    from detectors import Detectors
    D = Detectors()
    seeds = {}
    src = REPO / cfg.source_dir
    for s in {r["set"] for r in rows}:
        for g in json.loads((src / f"games_{s}.json").read_text())["games"]:
            if "roles" in g:
                seeds[(s, g["game_id"])] = (g["seed"], g["roles"])
    report = {"random_direction": {"candidate": int(kbest), "calib_auroc_ann": a_best, "calib_top1_ann": t_best},
              "mix_alpha": alphas, "mix_curve_top1": curve}
    for det in cfg.passive_detectors[1:]:
        vals = []
        for r, a32, a48, ar in zip(rows, l32, l48, rand):
            seed, roles = seeds[(r["set"], r["game_id"])]
            vals.append(D.score(det, {"l32": float(a32), "l48": float(a48), "rand": float(ar)},
                                role=r["role"], kind=r["kind"], speaker=r["speaker"], decoy=D.decoy(seed, roles),
                                key=f"{seed}|{r['kind']}|{r['round']}|{r['speaker']}|{r['position']}"))
        vals = np.array(vals)
        report[det] = {part: {"auroc_ann": round(auroc(rows, vals, m & ann), 3),
                              "auroc_ans": round(auroc(rows, vals, m & ~ann), 3) if (m & ~ann).any() else None,
                              "top1_ann": round(top1(rows, vals, m & ann), 3),
                              "mean_ann": round(float(vals[m & ann].mean()), 2),
                              "sd_ann": round(float(vals[m & ann].std()), 2),
                              "mean_ans": round(float(vals[m & ~ann].mean()), 2) if (m & ~ann).any() else None}
                       for part, m in (("calib", calib), ("passive", ~calib))}
        print(det, json.dumps(report[det]), flush=True)
    (OUT / "calibration_report.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
