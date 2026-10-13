"""EXPLORATORY (not preregistered): random-direction control for the fibs
linear ensemble, on the saved window activations (CPU, no subject model).

``rand``  (matched, the one reported): the published linear ensemble with
  every layer's weight vector replaced by a seeded random direction of the
  same norm. Everything else is the trained probe's own: the read token (the
  one before ``<|im_end|>``), the per-layer input scale, the bias, the
  sigmoid, the mean over the six layers, and the contrast "Yes." - "No.".
``randraw`` (the plain definition, for comparison): mean over the six layers
  of the raw projection onto the same unit directions, same token, same
  contrast. No per-layer scale, so the late layers (larger norms) dominate.

The direction for (seed, layer) is ``direction(seed, layer, d)``;
apollo_probe.py uses the same function at layer 16. Before any random score
is written, the vectorised read is checked against scores_{arm}.csv with the
trained weights (it must reproduce ``linear_contrast``).

  random_{arm}.csv      game_id, round, position, speaker, role,
                        rand_s{seed}, randraw_s{seed} per seed (contrasts)
  results_random.json   AUROC (Parsnip vs Carrot, as computed, no sign flip)
                        per seed for 3p / 5p / pooled, and min/median/max

Run:  uv run --no-sync python experiments/2026-10-12-fibs-probes/random_probe.py
"""
from __future__ import annotations

import csv
import json

import numpy as np
import torch

from analyze import MAIN, auroc
from common import OUT, load_config, probe_dir, read_jsonl

from probe_inference import load_probe  # noqa: E402  (vendor package, via common's sys.path)
from probe_inference.score import build_read_mask  # noqa: E402

KEY = ("game_id", "round", "position", "speaker", "role")
VARIANTS = ("rand", "randraw")


def direction(seed: int, layer: int, d: int) -> np.ndarray:
    """Unit vector, isotropic Gaussian direction, seeded by (seed, layer)."""
    v = np.random.default_rng([seed, layer]).standard_normal(d)
    return v / np.linalg.norm(v)


def read_tokens(cfg, arm: str, read_window: str, d_model: int) -> tuple[list[dict], np.ndarray]:
    """(transcript rows, (N, n_prefills, n_layers, d) float32 activations at
    the linear probe's read token), in transcript order."""
    index = json.loads((OUT / f"acts_{arm}" / "index.json").read_text())
    if index["layers"] != cfg.probe_layers or index["prefills"] != cfg.prefills:
        raise ValueError("acts index does not match config.json")
    if any(e["dropped"] for e in index["turns"]):
        raise ValueError(f"{arm}: dropped turns in the acts index; the row set would differ from scores_{arm}.csv")
    recs = read_jsonl(OUT / f"transcripts_{arm}.jsonl")
    out = np.empty((len(recs), len(cfg.prefills), len(cfg.probe_layers), d_model), dtype=np.float32)
    game, loaded = None, None
    for i, r in enumerate(recs):
        if r["game_id"] != loaded:
            game, loaded = torch.load(OUT / f"acts_{arm}" / f"{r['game_id']}.pt", weights_only=True), r["game_id"]
        for j, prefill in enumerate(cfg.prefills):
            e = game[f"r{r['round']}p{r['position']}"][prefill]
            mask = build_read_mask(read_window, e["prompt_mask"], e["completion_mask"], torch.tensor(0))
            if int(mask.sum()) != 1:
                raise ValueError(f"{arm} {r['game_id']}: read mask selects {int(mask.sum())} tokens")
            out[i, j] = e["acts"][:, mask].squeeze(1).float().numpy()
    return recs, out


def ensemble(x: np.ndarray, w: np.ndarray, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """probe_inference's linear read: x (N, P, L, d) float32, w (L, d),
    b / scale (L,) -> (N, P) mean over layers of sigmoid(w . x / scale + b)."""
    logits = np.einsum("npld,ld->npl", x / scale[None, None, :, None].astype(np.float32), w.astype(np.float32),
                       dtype=np.float32) + b.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).mean(-1)


def main():
    cfg = load_config()
    probe = load_probe(probe_dir(cfg, "linear"))
    if tuple(probe.layers) != tuple(cfg.probe_layers) or probe.used_layers != probe.layers:
        raise ValueError(f"linear probe layers {probe.layers} / used {probe.used_layers} != config {cfg.probe_layers}")
    mods = [probe.modules[L] for L in cfg.probe_layers]
    if any(m.normalize_input != "unit_norm" or m.nhead != 1 for m in mods):
        raise ValueError("the linear probe is not a single-head unit_norm probe; ensemble() does not implement it")
    W = np.stack([m.linear.weight.detach().numpy()[0] for m in mods])
    b = np.array([float(m.linear.bias.detach()) for m in mods])
    scale = np.array([float(m.input_scale) for m in mods])
    lie = cfg.prefills.index(cfg.lie_prefill)
    (other,) = [j for j in range(len(cfg.prefills)) if j != lie]
    units = {s: np.stack([direction(s, L, W.shape[1]) for L in cfg.probe_layers]) for s in cfg.random_seeds}

    res = {"definition": {"rand": "linear ensemble with each layer's weight replaced by a random direction of the "
                                  "same norm (same read token, input scale, bias, sigmoid, layer mean); Yes. - No.",
                          "randraw": "mean over layers of the raw projection onto the same unit directions; Yes. - No."},
           "seeds": cfg.random_seeds, "layers": cfg.probe_layers, "auroc": {v: {} for v in VARIANTS}}
    scores, labels = {v: {} for v in VARIANTS}, {}
    for arm in MAIN:
        recs, x = read_tokens(cfg, arm, probe.read_window, probe.d_model)
        ref = list(csv.DictReader(open(OUT / f"scores_{arm}.csv")))
        if [(r["game_id"], str(r["round"]), str(r["position"])) for r in recs] != \
                [(r["game_id"], r["round"], r["position"]) for r in ref]:
            raise ValueError(f"{arm}: transcript rows and scores_{arm}.csv rows differ")
        s = ensemble(x, W, b, scale)
        err = np.abs((s[:, lie] - s[:, other]) - np.array([float(r["linear_contrast"]) for r in ref])).max()
        print(f"{arm}: {len(recs)} announcements; trained weights reproduce linear_contrast to {err:.2e}")
        if err > 1e-5:
            raise ValueError(f"{arm}: the vectorised read does not reproduce scores_{arm}.csv (max error {err})")
        labels[arm] = np.array([r["role"] == "Parsnip" for r in recs], dtype=int)
        for seed, u in units.items():
            p = ensemble(x, u * np.linalg.norm(W, axis=1, keepdims=True), b, scale)
            scores["rand"][arm, seed] = p[:, lie] - p[:, other]
            raw = np.einsum("npld,ld->np", x.astype(np.float64), u) / len(cfg.probe_layers)
            scores["randraw"][arm, seed] = raw[:, lie] - raw[:, other]
        with open(OUT / f"random_{arm}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([*KEY, *(f"{v}_s{s_}" for v in VARIANTS for s_ in cfg.random_seeds)])
            for i, r in enumerate(recs):
                w.writerow([*(r[k] for k in KEY),
                            *(f"{scores[v][arm, s_][i]:.6f}" for v in VARIANTS for s_ in cfg.random_seeds)])

    for v in VARIANTS:
        per_seed = {}
        for seed in cfg.random_seeds:
            per_seed[str(seed)] = {arm: auroc(scores[v][arm, seed], labels[arm]) for arm in MAIN}
            per_seed[str(seed)]["pooled"] = auroc(np.concatenate([scores[v][arm, seed] for arm in MAIN]),
                                                  np.concatenate([labels[arm] for arm in MAIN]))
        res["auroc"][v]["per_seed"] = per_seed
        res["auroc"][v]["summary"] = {
            k: {"min": float(np.min(a)), "median": float(np.median(a)), "max": float(np.max(a))}
            for k in (*MAIN, "pooled") for a in [[per_seed[str(s_)][k] for s_ in cfg.random_seeds]]}
        print(f"\n{v}: seed  " + "  ".join(f"{k:>6}" for k in (*MAIN, "pooled")))
        for seed in cfg.random_seeds:
            print(f"      {seed:>4}  " + "  ".join(f"{per_seed[str(seed)][k]:6.3f}" for k in (*MAIN, "pooled")))
        for stat in ("min", "median", "max"):
            print(f"    {stat:>6}  " + "  ".join(f"{res['auroc'][v]['summary'][k][stat]:6.3f}" for k in (*MAIN, "pooled")))
    (OUT / "results_random.json").write_text(json.dumps(res, indent=1) + "\n")
    print("\nwrote random_{3p,5p}.csv, results_random.json")


if __name__ == "__main__":
    main()
