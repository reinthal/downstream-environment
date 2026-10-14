"""Train the fibs linear / MLP ensembles for every (condition, seed) on the
Qwen3.5-27B FIBSv1 read-token store, with the recipe of
experiments/2026-10-14-fibs-linear-no-sd.

Validation hold-out (shared by all probes): ``split_validation`` over the
rows outside ``cfg.exclude_sources`` (seed ``cfg.val_seed``). Training pool
per (condition, seed):
  no_sd        the remaining rows outside ``cfg.exclude_sources``
  random_drop  all remaining rows (every family) minus ``cfg.n_drop`` drawn
               uniformly with ``random.Random(seed)``
then ``balance_within_family(seed)``; the seed also sets init and order.

  probes_main/{name}/{arch}/   probe_inference directory
  train/{name}_{arch}.json     training record
  train/{name}_split.parquet   i -> split (train / val / balanced_out / dropped)
  train_results.json           per probe: fibs validation AUROC and the
                               exclude_sources rows' AUROC (held out for
                               no_sd), both through load_probe; positive control

Run:  CUDA_VISIBLE_DEVICES=6 uv run --no-sync python experiments/2026-10-14-fibs-no-sd-carrot-parsnip/train.py [--seeds 0]
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from common import OUT, REPO, load_config

from fibs_probes import (auroc, balance_within_family, load_read_token, load_rows, save_probe,  # noqa: E402
                         split_validation, train_layer_ensemble)
from probe_inference import load_probe  # noqa: E402


def score(probe, x: torch.Tensor) -> np.ndarray:
    """Ensemble score of a saved probe on (N, L, d) read-token activations."""
    return torch.cat([probe.score_batch(x[s:s + 4096, :, None].float(), torch.ones(len(x[s:s + 4096]), 1, dtype=torch.bool))
                      for s in range(0, len(x), 4096)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", help="subset of cfg.train_seeds (default: all)")
    args = ap.parse_args()
    cfg = load_config()
    cfg.save()
    seeds = args.seeds if args.seeds is not None else cfg.train_seeds
    if set(seeds) - set(cfg.train_seeds):
        raise ValueError(f"seeds {seeds} not in config train_seeds {cfg.train_seeds}")
    acts_dir, device = REPO / cfg.acts_dir, torch.device("cuda")

    rows = load_rows(acts_dir)
    unknown = set(cfg.exclude_sources) - {r["source"] for r in rows}
    if unknown:
        raise ValueError(f"exclude_sources not in FIBSv1: {sorted(unknown)}")
    excluded = [r for r in rows if r["source"] in cfg.exclude_sources]
    if len(excluded) != cfg.n_drop:
        raise ValueError(f"n_drop {cfg.n_drop} != {len(excluded)} rows of {cfg.exclude_sources}")
    kept = [r for r in rows if r["source"] not in cfg.exclude_sources]
    split_validation(kept, cfg.n_val, cfg.val_seed)
    val_ids = {r["i"] for r in kept if r["split"] == "val"}
    val_pairs = {r["pair_id"] for r in kept if r["split"] == "val" and r["pair_id"]}
    if any(r["pair_id"] in val_pairs for r in excluded):
        raise ValueError("an excluded-family row shares a pair_id with the validation hold-out")
    print(f"{len(rows)} rows; {len(excluded)} in {cfg.exclude_sources}; {len(val_ids)} validation")

    pos = {r["i"]: j for j, r in enumerate(rows)}
    x_all = load_read_token(acts_dir, rows)                    # (N, L, d) bf16, rows order
    y_all = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
    va = [r for r in rows if r["i"] in val_ids]
    va_idx = torch.tensor([pos[r["i"]] for r in va])
    ex_idx = torch.tensor([pos[r["i"]] for r in excluded])
    x_val, y_val = x_all[va_idx].to(device), y_all[va_idx].to(device)

    (OUT / "train").mkdir(exist_ok=True)
    res_path = OUT / "train_results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for cond in cfg.conditions:
        for seed in seeds:
            name = f"{cond}_s{seed}"
            pool = [dict(r, split="train") for r in rows if r["i"] not in val_ids]
            if cond == "no_sd":
                pool = [r for r in pool if r["source"] not in cfg.exclude_sources]
                dropped = []
            elif cond == "random_drop":
                drop = set(random.Random(seed).sample([r["i"] for r in pool], cfg.n_drop))
                dropped = [r for r in pool if r["i"] in drop]
                pool = [r for r in pool if r["i"] not in drop]
            else:
                raise ValueError(cond)
            tr = balance_within_family(pool, seed)
            split = {r["i"]: "balanced_out" for r in pool}
            split.update({r["i"]: "train" for r in tr})
            split.update({r["i"]: "dropped" for r in dropped})
            split.update({i: "val" for i in val_ids})
            if cond == "no_sd":
                split.update({r["i"]: "dropped" for r in excluded})
            pq.write_table(pa.table({"i": list(split), "split": list(split.values())}),
                           OUT / "train" / f"{name}_split.parquet")
            tr_idx = torch.tensor([pos[r["i"]] for r in tr])
            x_tr, y_tr = x_all[tr_idx].to(device), y_all[tr_idx].to(device)
            print(f"== {name}: {len(tr)} training rows ({len(dropped) or len(excluded)} dropped before balance)")

            for arch in cfg.archs:
                model, used, record = train_layer_ensemble(
                    arch, cfg.probe_layers, x_tr, y_tr, x_val, y_val, cfg.train_hparams[arch],
                    batch_size=cfg.train_batch_size, warmup_steps=cfg.warmup_steps, grad_clip=cfg.grad_clip_adamw,
                    scale_rows=cfg.scale_rows, layer_loss_cut=cfg.layer_loss_cut, seed=seed)
                pdir = OUT / "probes_main" / name / arch
                save_probe(arch, model, cfg.probe_layers, used, pdir, {
                    "model": cfg.model_id, "model_revision": cfg.model_revision,
                    "trained_by": "experiments/2026-10-14-fibs-no-sd-carrot-parsnip/train.py (fibs_probes)",
                    "fibs": {"repo": cfg.fibs_repo, "revision": cfg.fibs_revision},
                    "condition": cond, "seed": seed, "exclude_sources": cfg.exclude_sources, "n_drop": cfg.n_drop})
                (OUT / "train" / f"{name}_{arch}.json").write_text(json.dumps(record, indent=1) + "\n")

                probe = load_probe(pdir)
                s_val = score(probe, x_all[va_idx])
                s_ex = score(probe, x_all[ex_idx])
                val_auc = auroc(s_val, y_all[va_idx].numpy())
                if abs(val_auc - record["val"]["auroc"]) > 1e-3:
                    raise ValueError(f"{name}/{arch}: saved probe val AUROC {val_auc:.4f} != trained "
                                     f"{record['val']['auroc']:.4f}")
                src = np.array([r["source"] for r in excluded])
                y_ex = y_all[ex_idx].numpy()
                results[f"{name}/{arch}"] = {
                    "condition": cond, "seed": seed, "arch": arch, "n_train": len(tr), "used_layers": used,
                    "val_auroc": val_auc, "excluded_auroc": auroc(s_ex, y_ex),
                    "excluded_by_source": {k: auroc(s_ex[src == k], y_ex[src == k]) for k in cfg.exclude_sources},
                    "positive_control_pass": val_auc >= cfg.pc_min_val_auroc}
                print(f"   {arch}: val {val_auc:.4f}, {'/'.join(cfg.exclude_sources)} {results[f'{name}/{arch}']['excluded_auroc']:.4f}")
                res_path.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
                if val_auc < cfg.pc_min_val_auroc:
                    raise RuntimeError(f"positive control failed: {name}/{arch} val AUROC {val_auc:.4f} "
                                       f"< {cfg.pc_min_val_auroc}")
            del x_tr, y_tr
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
