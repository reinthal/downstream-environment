"""EXPLORATORY: does the Carrot-Parsnip role axis detect lies on FIBSv1?

Directions fitted on the pooled 3p + 5p announcements (``acts_readtok/``, as in
pca.py), per layer and per Carrot-Parsnip feature (``cfg.features``):
  ``pc1``  the first principal component, sign fixed so the Parsnip mean is
           positive (the one bit of supervision);
  ``dom``  the Parsnip - Carrot mean difference (unit).
They are applied, unchanged, to the FIBSv1 validation hold-out of the stored
Qwen3.5-27B read tokens (``cfg.fibs_acts_dir``: the token before the answer's
``<|im_end|>``, the fibs linear probe's read; one completion per row, no
prefill contrast). Score = projection, higher = deceptive.

Reported per (feature, direction, layer): AUROC against
  ``followup_deceptive`` (the FIBS label: the follow-up answer is false),
  ``base_deceptive``     (the original response was deceptive: deception in
                          the context, the role-like signal),
and the label AUROC within each ``base_deceptive`` stratum. Baselines on the
same rows: the trained fibs linear weight per layer and the full ensemble,
``cfg.n_random`` random directions, and FIBS's own PC1 (sign fixed by the
label) and diff-of-means (in-sample ceiling), plus the FIBS variance
decomposition (is the label the dominant axis there?).

  results_fibs.json, figures/fibs_transfer.svg

Run:  uv run --no-sync python experiments/2026-10-16-random-direction-pca/fibs_transfer.py
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from pca import CACHE, OUT, REPO, auroc_cols, feature, load_config, unit_dirs


def directions(F: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    Fc = F - F.mean(0)
    delta = F[y].mean(0) - F[~y].mean(0)
    v = np.linalg.svd(Fc, full_matrices=False)[2][0]
    v = v if v @ delta > 0 else -v
    return {"pc1": v, "dom": delta / np.linalg.norm(delta)}


def decomposition(F: np.ndarray, y: np.ndarray) -> dict:
    Fc = F - F.mean(0)
    m1, m0 = F[y].mean(0), F[~y].mean(0)
    delta = m1 - m0; d2 = float(delta @ delta)
    Wc = np.concatenate([F[y] - m1, F[~y] - m0])
    _, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    lam = S ** 2 / len(F)
    p = float(y.mean())
    return {"delta_sq_over_tr_within": d2 / float((Wc ** 2).sum(1).mean()),
            "label_share_of_total_variance": p * (1 - p) * d2 / float(lam.sum()),
            "pc1_explained_var": float(lam[0] / lam.sum()),
            "delta_share_pc1": float((Vt[0] @ delta) ** 2 / d2),
            "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum())}


def main():
    cfg = load_config()
    from fibs_probes import load_read_token, load_rows

    root = REPO / cfg.fibs_acts_dir
    rows = [r for r in load_rows(root) if r["split"] == "val"]
    extra = pq.read_table(root.parent / "rows.parquet", columns=["row", "base_deceptive", "followup_deceptive"])
    base = {int(r): b for r, b in zip(extra["row"].to_pylist(), extra["base_deceptive"].to_pylist())}
    fu = {int(r): b for r, b in zip(extra["row"].to_pylist(), extra["followup_deceptive"].to_pylist())}
    for r in rows:
        if str(fu[r["i"]]) != str(bool(r["label"])):
            raise ValueError(f"row {r['i']}: followup_deceptive {fu[r['i']]} != label {r['label']}")
    y = np.array([r["label"] == 1 for r in rows])
    yb = np.array([str(base[r["i"]]) == "True" for r in rows])
    print(f"FIBS val: {len(rows)} rows, {y.sum()} false answers, {yb.sum()} with a deceptive original response",
          flush=True)
    X = load_read_token(root, rows).float().numpy()          # (N, 6, d)

    info = json.loads((CACHE / "index.json").read_text())
    layers, prefills, lie = info["layers"], info["prefills"], info["lie_prefill"]
    probe = np.load(CACHE / "linear_probe.npz")
    if list(probe["layers"]) != layers:
        raise ValueError("probe layers differ from the cache's")
    xs = {a: np.load(CACHE / f"{a}.npy", mmap_mode="r") for a in cfg.main_arms}
    ycp = np.array([m["role"] == "Parsnip" for a in cfg.main_arms for m in info["sets"][a]])

    def aucs(s: np.ndarray) -> dict:
        a = lambda sel, lab: float(auroc_cols(s[sel][:, None], lab[sel])[0])
        allr = np.ones(len(s), bool)
        return {"label": a(allr, y), "base_deceptive": a(allr, yb),
                "label_within_base_honest": a(~yb, y), "label_within_base_deceptive": a(yb, y)}

    res = {"n_val": len(rows), "n_false": int(y.sum()), "n_base_deceptive": int(yb.sum()),
           "crosstab": {f"base_{int(b)}_label_{int(l)}": int(((yb == b) & (y == l)).sum())
                        for b in (0, 1) for l in (0, 1)},
           "layers": layers, "cp_directions": {}, "baselines": {}}

    # trained fibs ensemble on these rows (positive control: the card reports 0.972 on its validation rows)
    W, b, scale = probe["W"], probe["b"], probe["scale"]
    logits = np.einsum("nld,ld->nl", X / scale[None, :, None], W) + b
    res["baselines"]["fibs_linear_ensemble"] = aucs((1 / (1 + np.exp(-logits))).mean(1))
    print(f"fibs linear ensemble: {res['baselines']['fibs_linear_ensemble']}", flush=True)

    for li, L in enumerate(layers):
        F = X[:, li].astype(np.float64)
        bl = {"trained_fibs_weight": aucs(F @ W[li]),
              "fibs_own": {k: aucs(F @ v) for k, v in directions(F, y).items()},
              "fibs_decomposition": decomposition(F, y)}
        A = auroc_cols(F @ unit_dirs(cfg.n_random, F.shape[1], cfg.random_seed, L).T, y)
        bl["random"] = {"median_abs_dev": float(np.median(np.abs(A - 0.5))),
                        "frac_extreme": float(((A > cfg.extreme_auroc) | (A < 1 - cfg.extreme_auroc)).mean()),
                        "max_folded": float(np.max(np.maximum(A, 1 - A)))}
        res["baselines"][str(L)] = bl
        for fname in cfg.features:
            Fcp = np.concatenate([feature(xs[a], fname, li, prefills, lie) for a in cfg.main_arms])
            for dname, v in directions(Fcp, ycp).items():
                r = aucs(F @ v)
                r["cos_with_fibs_dom"] = float(v @ directions(F, y)["dom"])
                r["cos_with_trained_w"] = float(v @ W[li] / np.linalg.norm(W[li]))
                res["cp_directions"].setdefault(fname, {}).setdefault(dname, {})[str(L)] = r
        c = res["cp_directions"]["contrast"]["pc1"][str(L)]
        print(f"L{L}: CP-PC1(contrast) label={c['label']:.3f} base={c['base_deceptive']:.3f} "
              f"within base honest={c['label_within_base_honest']:.3f} within base deceptive={c['label_within_base_deceptive']:.3f} | "
              f"trained w label={bl['trained_fibs_weight']['label']:.3f} | FIBS own PC1 {bl['fibs_own']['pc1']['label']:.3f} "
              f"DoM {bl['fibs_own']['dom']['label']:.3f} | random median|dev| {bl['random']['median_abs_dev']:.3f} | "
              f"FIBS |D|^2/trW {bl['fibs_decomposition']['delta_sq_over_tr_within']:.2f} D-in-PC1 {bl['fibs_decomposition']['delta_share_pc1']:.2f}",
              flush=True)
    (OUT / "results_fibs.json").write_text(json.dumps(res, indent=1) + "\n")
    figure(cfg, res)
    print("wrote results_fibs.json, figures/fibs_transfer.svg")


def figure(cfg, res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pca import COL, INK, _style

    layers = res["layers"]
    targets = [("label", "FIBS label (follow-up answer is false)"),
               ("base_deceptive", "original response was deceptive")]
    lines = [("trained fibs linear weight", lambda L: res["baselines"][str(L)]["trained_fibs_weight"], COL["Carrot"], "-"),
             ("FIBS own PC1 (+ sign)", lambda L: res["baselines"][str(L)]["fibs_own"]["pc1"], COL["all-Carrot games"], "-"),
             ("Carrot-Parsnip PC1, contrast", lambda L: res["cp_directions"]["contrast"]["pc1"][str(L)], COL["Parsnip"], "-"),
             ('Carrot-Parsnip PC1, "Yes." alone', lambda L: res["cp_directions"]["Yes."]["pc1"][str(L)], COL["Parsnip"], "--"),
             ('Carrot-Parsnip PC1, "No." alone', lambda L: res["cp_directions"]["No."]["pc1"][str(L)], COL["Parsnip"], ":")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, layout="constrained")
    for ax, (key, title) in zip(axes, targets):
        for lab, get, c, ls in lines:
            ax.plot(layers, [get(L)[key] for L in layers], color=c, ls=ls, lw=2, marker="o", ms=5, label=lab)
        ax.axhline(0.5, color=INK, lw=0.8, ls=":")
        ax.set_ylim(0, 1); ax.set_xticks(layers)
        ax.set_xlabel("layer"); ax.set_title(f"AUROC vs {title}", fontsize=10)
        _style(ax)
    axes[0].set_ylabel(f"AUROC, FIBS validation ({res['n_val']} rows)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Directions fitted on Carrot-Parsnip, applied unchanged to FIBSv1 validation read tokens", fontsize=11)
    fig.savefig(OUT / "figures" / "fibs_transfer.svg"); plt.close(fig)


if __name__ == "__main__":
    main()
