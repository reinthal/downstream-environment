"""EXPLORATORY (not preregistered): why do random directions separate Parsnip
from Carrot announcements? PCA of the fibs linear probe's read-token
activations of 2026-10-12-fibs-probes (stored in ``cfg.src_dir``/acts_*).

Stages (both ``.venv``, CPU only):

``extract``  the read token (the one before ``<|im_end|>`` of the prefilled
  answer, as the fibs linear probe reads it) at layers 19-58, both prefills,
  for the main arms, the all-Carrot arms and the alpaca control, plus the
  trained linear probe's weights -> ``acts_readtok/`` (gitignored).
``analyze``  per feature (``cfg.features``: the "Yes." - "No." contrast and
  each prefill alone) and layer, on the pooled main arms (PCA fitted there,
  label = speaker is the Parsnip):
  - variance decomposition: total / within-class trace, the role mean shift
    Delta, the share of all variance that is the role shift;
  - PCA spectrum, participation ratio, per-PC share of |Delta|^2, per-PC role
    AUROC, 5p-vs-3p AUROC and |corr| with the round (confound check);
  - ``cfg.n_random`` random unit directions: empirical AUROC vs the Gaussian
    prediction Phi(d'/sqrt 2), d' = u.Delta / sqrt(u' S_within u);
  - the same after projecting out the top-k PCs (``cfg.ablate_k``), with the
    trained probe's and the refitted diff-of-means direction's AUROC;
  - where the all-Carrot games and alpaca sit on the role axis (0 = Carrot
    mean, 1 = Parsnip mean of the main arms);
  - the 10 matched random seeds of 2026-10-12 (random_probe.direction):
    per-layer d' next to their ensemble AUROC.
  -> ``results.json``, ``figures/*.svg``.

Run:  uv run --no-sync python experiments/2026-10-16-random-direction-pca/pca.py --stage extract
      uv run --no-sync python experiments/2026-10-16-random-direction-pca/pca.py --stage analyze
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
CACHE = OUT / "acts_readtok"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from linear_probes.config import ExperimentConfig  # noqa: E402

META_KEYS = ("game_id", "round", "position", "speaker", "role")


@dataclass
class PcaConfig(ExperimentConfig):
    src_dir: str = ""                   # experiment whose stored activations / probe read are analysed
    main_arms: list[str] = field(default_factory=list)   # pooled for the PCA fit; label = Parsnip
    cc_arms: list[str] = field(default_factory=list)     # all-Carrot games, projected only
    alpaca: bool = True                 # alpaca control (acts_alpaca), projected only
    features: list[str] = field(default_factory=list)    # "contrast" (lie prefill - other) and/or a prefill
    n_pcs: int = 20                     # PCs reported / drawn
    n_random: int = 2000                # random unit directions per (feature, layer)
    random_seed: int = 0
    ablate_k: list[int] = field(default_factory=list)    # top-k PCs projected out
    extreme_auroc: float = 0.8          # "extreme" random direction: AUROC > this or < 1 - this
    scatter_feature: str = "contrast"   # feature drawn in the figures
    fibs_acts_dir: str = ""             # FIBSv1 read-token store (fibs_probes layout), fibs_transfer.py


def load_config() -> PcaConfig:
    return PcaConfig.load(OUT / "config.json")


def use_src(cfg: PcaConfig) -> None:
    """Put the source experiment's modules (common, random_probe, ...) on sys.path."""
    p = str(REPO / cfg.src_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


# ───────────────────────────── extract ─────────────────────────────

def stage_extract(cfg: PcaConfig) -> None:
    use_src(cfg)
    import torch
    from common import OUT as SRC, load_config as src_config, probe_dir
    from probe_inference import load_probe
    from probe_inference.score import build_read_mask
    from random_probe import read_tokens

    scfg = src_config()
    probe = load_probe(probe_dir(scfg, "linear"))
    if tuple(probe.layers) != tuple(scfg.probe_layers):
        raise ValueError(f"linear probe layers {probe.layers} != source config {scfg.probe_layers}")
    mods = [probe.modules[L] for L in scfg.probe_layers]
    if any(m.normalize_input != "unit_norm" or m.nhead != 1 for m in mods):
        raise ValueError("the linear probe is not a single-head unit_norm probe")
    CACHE.mkdir(exist_ok=True)
    np.savez(CACHE / "linear_probe.npz",
             W=np.stack([m.linear.weight.detach().numpy()[0] for m in mods]),
             b=np.array([float(m.linear.bias.detach()) for m in mods]),
             scale=np.array([float(m.input_scale) for m in mods]),
             layers=np.array(scfg.probe_layers))
    info = {"layers": scfg.probe_layers, "prefills": scfg.prefills, "lie_prefill": scfg.lie_prefill,
            "read_window": probe.read_window, "sets": {}}

    for arm in (*cfg.main_arms, *cfg.cc_arms):
        recs, x = read_tokens(scfg, arm, probe.read_window, probe.d_model)
        np.save(CACHE / f"{arm}.npy", x)
        info["sets"][arm] = [{k: r[k] for k in META_KEYS} for r in recs]
        print(f"{arm}: {x.shape}", flush=True)

    if cfg.alpaca:
        parts = sorted((SRC / "acts_alpaca").glob("part*.pt"))
        rows, meta = [], []
        for part in parts:
            d = torch.load(part, weights_only=True)
            for idx in sorted(d):
                per = []
                for prefill in scfg.prefills:
                    e = d[idx][prefill]
                    mask = build_read_mask(probe.read_window, e["prompt_mask"], e["completion_mask"], torch.tensor(0))
                    if int(mask.sum()) != 1:
                        raise ValueError(f"alpaca {idx}: read mask selects {int(mask.sum())} tokens")
                    per.append(e["acts"][:, mask].squeeze(1).float().numpy())
                rows.append(np.stack(per))
                meta.append({"idx": idx, "role": "alpaca"})
        x = np.stack(rows).astype(np.float32)
        np.save(CACHE / "alpaca.npy", x)
        info["sets"]["alpaca"] = meta
        print(f"alpaca: {x.shape}", flush=True)
    (CACHE / "index.json").write_text(json.dumps(info) + "\n")
    print(f"wrote {CACHE}")


# ───────────────────────────── analyze ─────────────────────────────

def auroc_cols(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC (positive = y True) of every column of S, ties averaged."""
    from scipy.stats import rankdata
    r = rankdata(S, axis=0)
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def feature(x: np.ndarray, name: str, li: int, prefills: list[str], lie: str) -> np.ndarray:
    if name == "contrast":
        (other,) = [p for p in prefills if p != lie]
        return x[:, prefills.index(lie), li].astype(np.float64) - x[:, prefills.index(other), li]
    return x[:, prefills.index(name), li].astype(np.float64)


def unit_dirs(n: int, d: int, seed: int, layer: int) -> np.ndarray:
    U = np.random.default_rng([seed, layer]).standard_normal((n, d))
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def random_stats(F: np.ndarray, y: np.ndarray, U: np.ndarray, extreme: float) -> dict:
    a = auroc_cols(F @ U.T, y)
    fold = np.abs(a - 0.5)
    return {"auroc": a, "median_abs_dev": float(np.median(fold)),
            "frac_extreme": float(((a > extreme) | (a < 1 - extreme)).mean())}


def analyse(F, y, is_second_arm, rnd, w, U, cfg) -> tuple[dict, dict]:
    """One (feature, layer). F (N, D) pooled main arms. Returns (results, arrays for figures)."""
    from scipy.stats import norm
    N, D = F.shape
    mu = F.mean(0); Fc = F - mu
    m1, m0 = F[y].mean(0), F[~y].mean(0)
    delta = m1 - m0; d2 = float(delta @ delta); dhat = delta / np.sqrt(d2)
    Wc = np.concatenate([F[y] - m1, F[~y] - m0])
    tr_total = float((Fc ** 2).sum(1).mean()); tr_within = float((Wc ** 2).sum(1).mean())
    p = float(y.mean())
    _, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    lam = S ** 2 / N
    # orient each PC so the Parsnip mean is on the positive side
    sgn = np.sign(Vt @ delta); sgn[sgn == 0] = 1; Vt = Vt * sgn[:, None]
    K = cfg.n_pcs
    pcs = Fc @ Vt[:K].T
    round_corr = [abs(float(np.corrcoef(pcs[:, k], rnd)[0, 1])) for k in range(K)]

    proj_d = F @ dhat
    sd_w = float(np.sqrt(((Wc @ dhat) ** 2).mean()))

    rs = random_stats(F, y, U, cfg.extreme_auroc)
    uWu = ((Wc @ U.T) ** 2).mean(0)
    uTu = ((Fc @ U.T) ** 2).mean(0)
    dprime = (U @ delta) / np.sqrt(uWu)
    pred = norm.cdf(dprime / np.sqrt(2))
    pc1_share_u = (U @ Vt[0]) ** 2 * lam[0] / uTu         # share of a random direction's variance from PC1
    role_share_u = (U @ dhat) ** 2 * p * (1 - p) * d2 / uTu   # share of it that is the role mean shift

    res = {
        "n": N, "d": D, "frac_parsnip": p,
        "tr_total": tr_total, "tr_within": tr_within,
        "delta_sq_over_tr_within": d2 / tr_within,
        "role_share_of_total_variance": p * (1 - p) * d2 / tr_total,
        "role_axis_var_share": float(((Fc @ dhat) ** 2).mean() / tr_total),   # incl. within-class spread along it
        "dprime_role_axis": float((proj_d[y].mean() - proj_d[~y].mean()) / sd_w),
        "auroc_role_axis_insample": float(auroc_cols(proj_d[:, None], y)[0]),
        "auroc_trained_probe_layer": float(auroc_cols((F @ w)[:, None], y)[0]),
        "cos_trained_w_role_axis": float(w @ dhat / np.linalg.norm(w)),
        "cos_trained_w_pc1": float(abs(w @ Vt[0]) / np.linalg.norm(w)),
        "cos_role_axis_pc1": float(abs(dhat @ Vt[0])),
        "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
        "explained_var_ratio": (lam[:K] / lam.sum()).tolist(),
        "delta_share_per_pc": ((Vt[:K] @ delta) ** 2 / d2).tolist(),
        "delta_share_cum_top": {str(k): float(((Vt[:k] @ delta) ** 2).sum() / d2) for k in (1, 2, 5, 10, 20, 50, 100)},
        "pc_role_auroc": auroc_cols(pcs, y).tolist(),
        "pc_5p_vs_3p_auroc": auroc_cols(pcs, is_second_arm).tolist(),
        "pc_abs_corr_round": round_corr,
        "random": {
            "n": len(U), "median_abs_dev": rs["median_abs_dev"], "frac_extreme": rs["frac_extreme"],
            "pred_median_abs_dev": float(np.median(np.abs(pred - 0.5))),
            "pred_frac_extreme": float(((pred > cfg.extreme_auroc) | (pred < 1 - cfg.extreme_auroc)).mean()),
            "corr_pred_emp": float(np.corrcoef(pred, rs["auroc"])[0, 1]),
            "max_abs_err_pred_emp": float(np.abs(pred - rs["auroc"]).max()),
            "rms_dprime": float(np.sqrt((dprime ** 2).mean())),
            "rms_dprime_theory": float(np.sqrt(d2 / tr_within)),  # isotropic-direction expectation
            "median_pc1_share_of_u_variance": float(np.median(pc1_share_u)),
            "median_role_share_of_u_variance": float(np.median(role_share_u)),
            "corr_abs_dev_vs_role_share": float(np.corrcoef(np.abs(rs["auroc"] - 0.5), role_share_u)[0, 1]),
        },
        "ablate": {},
    }
    arrays = {"rand_auroc": rs["auroc"], "rand_pred": pred, "Vt": Vt[:2], "mu": mu, "m0": m0, "delta": delta,
              "tr_total": tr_total, "pcs": pcs[:, :2], "rand_ablate": {}}
    for k in cfg.ablate_k:
        V = Vt[:k]
        Fa = F - (Fc @ V.T) @ V
        ra = random_stats(Fa, y, U, cfg.extreme_auroc)
        da = Fa[y].mean(0) - Fa[~y].mean(0)
        Wa = np.concatenate([Fa[y] - Fa[y].mean(0), Fa[~y] - Fa[~y].mean(0)])
        res["ablate"][str(k)] = {
            "delta_sq_remaining": float(da @ da / d2),
            "delta_sq_over_tr_within": float(da @ da / (Wa ** 2).sum(1).mean()),
            "random_median_abs_dev": ra["median_abs_dev"], "random_frac_extreme": ra["frac_extreme"],
            "auroc_trained_probe_layer": float(auroc_cols((Fa @ w)[:, None], y)[0]),
            "auroc_role_axis_refit_insample": float(auroc_cols((Fa @ (da / np.linalg.norm(da)))[:, None], y)[0]),
        }
        arrays["rand_ablate"][k] = ra["auroc"]
    return res, arrays


def place(F_set: np.ndarray, a: dict) -> dict:
    """Position on the role axis (0 = main Carrot mean, 1 = main Parsnip mean) and on PC1, PC2 (units of the
    main arms' total std)."""
    t = (F_set - a["m0"]) @ a["delta"] / (a["delta"] @ a["delta"])
    pc = (F_set - a["mu"]) @ a["Vt"].T / np.sqrt(a["tr_total"])
    return {"role_axis_mean": float(t.mean()), "role_axis_sd": float(t.std()),
            "role_axis_q05_q95": [float(np.quantile(t, 0.05)), float(np.quantile(t, 0.95))],
            "_pc": pc}


def stage_analyze(cfg: PcaConfig) -> None:
    use_src(cfg)
    from random_probe import direction

    info = json.loads((CACHE / "index.json").read_text())
    layers, prefills, lie = info["layers"], info["prefills"], info["lie_prefill"]
    probe = np.load(CACHE / "linear_probe.npz")
    xs = {s: np.load(CACHE / f"{s}.npy", mmap_mode="r") for s in info["sets"]}
    main_meta = [m for a in cfg.main_arms for m in info["sets"][a]]
    y = np.array([m["role"] == "Parsnip" for m in main_meta])
    second = np.concatenate([np.full(len(info["sets"][a]), i == 1) for i, a in enumerate(cfg.main_arms)])
    rnd = np.array([m["round"] for m in main_meta], dtype=float)
    holdout = [*cfg.cc_arms, *(["alpaca"] if cfg.alpaca else [])]
    for s in holdout:
        if s in cfg.cc_arms and any(m["role"] != "Carrot" for m in info["sets"][s]):
            raise ValueError(f"{s}: an all-Carrot arm has a non-Carrot speaker")

    results = {"layers": layers, "features": cfg.features, "main_arms": cfg.main_arms,
               "n_parsnip": int(y.sum()), "n_carrot": int((~y).sum()), "by_feature": {}}
    figdata = {}
    for fname in cfg.features:
        results["by_feature"][fname] = {}
        for li, L in enumerate(layers):
            F = np.concatenate([feature(xs[a], fname, li, prefills, lie) for a in cfg.main_arms])
            U = unit_dirs(cfg.n_random, F.shape[1], cfg.random_seed, L)
            res, arr = analyse(F, y, second, rnd, probe["W"][li].astype(np.float64), U, cfg)
            res["holdout_role_axis"] = {}
            pcs_hold = {}
            for name, sel in (("main Carrot", ~y), ("main Parsnip", y)):
                q = place(F[sel], arr); q.pop("_pc"); res["holdout_role_axis"][name] = q
            for s in holdout:
                q = place(feature(xs[s], fname, li, prefills, lie), arr)
                pcs_hold[s] = q.pop("_pc"); res["holdout_role_axis"][s] = q
            results["by_feature"][fname][str(L)] = res
            if fname == cfg.scatter_feature:
                arr["pcs_hold"] = pcs_hold
                figdata[L] = arr
            r = res["random"]
            print(f"{fname:>8} L{L}: |D|^2/trW={res['delta_sq_over_tr_within']:.2f} role share={res['role_share_of_total_variance']:.2f} "
                  f"EVR1={res['explained_var_ratio'][0]:.2f} D-in-PC1={res['delta_share_per_pc'][0]:.2f} "
                  f"PR={res['participation_ratio']:.1f} | rand extreme={r['frac_extreme']:.2f} (pred {r['pred_frac_extreme']:.2f}, "
                  f"corr {r['corr_pred_emp']:.3f}) | ablate1 extreme={res['ablate'][str(cfg.ablate_k[0])]['random_frac_extreme']:.2f} "
                  f"probe {res['auroc_trained_probe_layer']:.3f}->{res['ablate'][str(cfg.ablate_k[0])]['auroc_trained_probe_layer']:.3f}",
                  flush=True)

    # the matched random seeds of the source experiment, per-layer d' on its contrast
    src_rand = json.loads((REPO / cfg.src_dir / "results_random.json").read_text())["auroc"]["rand"]["per_seed"]
    seeds = {}
    for seed in src_rand:
        per = {}
        for li, L in enumerate(layers):
            F = np.concatenate([feature(xs[a], "contrast", li, prefills, lie) for a in cfg.main_arms])
            u = direction(int(seed), L, F.shape[1])
            s = F @ u
            sd = np.sqrt(0.5 * (s[y].var() + s[~y].var()))
            per[str(L)] = float((s[y].mean() - s[~y].mean()) / sd)
        seeds[seed] = {"ensemble_auroc_pooled": src_rand[seed]["pooled"], "dprime_per_layer": per}
        print(f"seed {seed}: ensemble {src_rand[seed]['pooled']:.3f}  d' " +
              " ".join(f"{per[str(L)]:+.2f}" for L in layers))
    results["source_random_seeds"] = seeds
    (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    figures(cfg, results, figdata, layers)
    print("wrote results.json, figures/")


# ───────────────────────────── figures ─────────────────────────────

# categorical order (dataviz reference palette); the colour follows the set
COL = {"Carrot": "#2a78d6", "Parsnip": "#eb6834", "all-Carrot games": "#1baf7a", "alpaca": "#eda100"}
GRID, INK = "#e4e3df", "#52514e"


def _style(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figures(cfg: PcaConfig, results: dict, figdata: dict, layers: list[int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (OUT / "figures").mkdir(exist_ok=True)
    fr = results["by_feature"][cfg.scatter_feature]
    K = cfg.n_pcs
    n = len(layers)
    tag = '"Yes." − "No." contrast' if cfg.scatter_feature == "contrast" else cfg.scatter_feature

    # 1. spectrum: variance per PC vs role shift per PC
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4), sharex=True, sharey="row", layout="constrained")
    for j, L in enumerate(layers):
        r = fr[str(L)]
        ks = np.arange(1, K + 1)
        axes[0, j].bar(ks, r["explained_var_ratio"], color=COL["Carrot"], width=0.75)
        axes[1, j].bar(ks, r["delta_share_per_pc"], color=COL["Parsnip"], width=0.75)
        axes[0, j].set_title(f"layer {L}\nrole = {100 * r['role_share_of_total_variance']:.0f}% of all variance", fontsize=9)
        for ax in axes[:, j]:
            _style(ax)
        axes[1, j].set_xlabel("principal component")
    axes[0, 0].set_ylabel("share of total variance")
    axes[1, 0].set_ylabel("share of |Parsnip − Carrot mean|²")
    axes[0, 0].set_ylim(0, 1); axes[1, 0].set_ylim(0, 1)
    fig.suptitle(f"PCA of the read-token {tag}, 3p + 5p pooled: variance per PC (top) and role shift per PC (bottom)",
                 fontsize=11)
    fig.savefig(OUT / "figures" / "spectrum.svg"); plt.close(fig)

    # 2. PC1 / PC2 scatter, every set, identical axes
    pts = []
    for L in layers:
        a = figdata[L]
        pts.append(a["pcs"] / np.sqrt(a["tr_total"]))
        pts += list(a["pcs_hold"].values())
    allp = np.concatenate(pts)
    lo, hi = np.quantile(allp, 0.002, axis=0), np.quantile(allp, 0.998, axis=0)
    pad = 0.05 * (hi - lo)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 8.4), sharex=True, sharey=True,
                             layout="constrained")
    y = None
    for ax, L in zip(axes.flat, layers):
        a = figdata[L]
        pc = a["pcs"] / np.sqrt(a["tr_total"])
        if y is None:
            y = np.array([m == "Parsnip" for m in _main_roles(cfg)])
        for s, p in a["pcs_hold"].items():
            lab = "alpaca" if s == "alpaca" else "all-Carrot games"
            ax.scatter(p[:, 0], p[:, 1], s=4, alpha=0.35, color=COL[lab], lw=0, label=lab, zorder=1)
        ax.scatter(pc[~y, 0], pc[~y, 1], s=4, alpha=0.5, color=COL["Carrot"], lw=0, label="Carrot", zorder=2)
        ax.scatter(pc[y, 0], pc[y, 1], s=4, alpha=0.6, color=COL["Parsnip"], lw=0, label="Parsnip", zorder=3)
        r = fr[str(L)]
        ax.set_title(f"layer {L}: PC1 {100 * r['explained_var_ratio'][0]:.0f}% var, "
                     f"{100 * r['delta_share_per_pc'][0]:.0f}% of role shift", fontsize=9)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xlabel("PC1 (units of total std)"); ax.set_ylabel("PC2 (units of total std)")
        _style(ax)
    h, l = axes.flat[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    order = [k for k in ("Carrot", "Parsnip", "all-Carrot games", "alpaca") if k in uniq]
    fig.legend([uniq[k] for k in order], order, loc="outside lower center", ncol=len(order), markerscale=4,
               frameon=False)
    fig.suptitle(f"Read-token {tag} on the PCs of the 3p + 5p games (fitted on main arms only)", fontsize=11)
    fig.savefig(OUT / "figures" / "pc_scatter.svg"); plt.close(fig)

    # 3. random-direction AUROC distributions, before / after projecting out top-k PCs
    ks = cfg.ablate_k[:2]
    bins = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 7.2), sharex=True, sharey=True,
                             layout="constrained")
    series = [("as is", None, COL["Carrot"])] + [(f"top {k} PC{'s' if k > 1 else ''} removed", k, c)
                                               for k, c in zip(ks, (COL["Parsnip"], COL["all-Carrot games"]))]
    for ax, L in zip(axes.flat, layers):
        a = figdata[L]
        for lab, k, c in series:
            v = a["rand_auroc"] if k is None else a["rand_ablate"][k]
            ax.hist(v, bins=bins, histtype="step", lw=2, color=c, label=lab)
        ax.axvline(0.5, color=INK, lw=0.8, ls=":")
        ax.axvline(fr[str(L)]["auroc_trained_probe_layer"], color=INK, lw=1.2, ls="--",
                   label="trained fibs linear weight (this layer)")
        ax.set_title(f"layer {L}", fontsize=10)
        ax.set_xlabel("AUROC, Parsnip vs Carrot"); ax.set_ylabel(f"random directions (of {cfg.n_random})")
        _style(ax)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(l), frameon=False)
    fig.suptitle(f"AUROC of {cfg.n_random} random unit directions on the read-token {tag}", fontsize=11)
    fig.savefig(OUT / "figures" / "random_auroc.svg"); plt.close(fig)

    # 4. Gaussian prediction vs empirical AUROC
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.0 * ((n + 1) // 2), 8.2), sharex=True, sharey=True,
                             layout="constrained")
    for ax, L in zip(axes.flat, layers):
        a = figdata[L]
        ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls=":")
        ax.scatter(a["rand_pred"], a["rand_auroc"], s=4, alpha=0.4, color=COL["Carrot"], lw=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(f"layer {L}: r = {fr[str(L)]['random']['corr_pred_emp']:.3f}", fontsize=10)
        ax.set_xlabel("predicted Φ(d′/√2), d′ = u·Δ / √(uᵀΣ_w u)"); ax.set_ylabel("empirical AUROC")
        _style(ax)
    fig.suptitle(f"Random directions: mean-shift prediction vs measured AUROC ({tag})", fontsize=11)
    fig.savefig(OUT / "figures" / "random_predicted.svg"); plt.close(fig)


def _main_roles(cfg: PcaConfig) -> list[str]:
    info = json.loads((CACHE / "index.json").read_text())
    return [m["role"] for a in cfg.main_arms for m in info["sets"][a]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "analyze"], required=True)
    args = ap.parse_args()
    cfg = load_config()
    cfg.save()
    {"extract": stage_extract, "analyze": stage_analyze}[args.stage](cfg)


if __name__ == "__main__":
    main()
