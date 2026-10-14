"""Train the fibs linear ensemble for Qwen3.5-27B on FIBSv1 without the
social-deduction families (``cfg.exclude_sources``), from the read-token
activations of experiments/2026-10-13-fibs-acts (``cfg.acts_dir``, the
read-token store layout of ``fibs_probes.data``).

  probes/{arch}/          probe_inference directory (load_probe reads it back)
  {arch}_train.json       training record (checkpoint, used layers, curve)
  results.json            AUROC of the saved probe, through load_probe: the
                          validation hold-out (overall, per source) and the
                          held-out social-deduction rows (overall, per source)
  split.parquet           i -> split (train / val / balanced_out / excluded)

Run:  CUDA_VISIBLE_DEVICES=1 uv run --no-sync python experiments/2026-10-14-fibs-linear-no-sd/run.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fibs_probes import (auroc, balance_within_family, load_read_token, load_rows, save_probe, split_validation,
                         train_layer_ensemble)
from linear_probes.config import ExperimentConfig
from probe_inference import load_probe

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]


@dataclass
class FibsTrainConfig(ExperimentConfig):
    model_revision: str = ""
    acts_dir: str = ""                      # a read-token store (rows.jsonl + acts_{k}.pt)
    probe_layers: list[int] = field(default_factory=list)
    exclude_sources: list[str] = field(default_factory=list)   # FIBSv1 families left out of train and val
    archs: list[str] = field(default_factory=list)
    n_val: int = 5000
    val_seed: int = 0
    train_seed: int = 0
    train_hparams: dict = field(default_factory=dict)
    train_batch_size: int = 32
    warmup_steps: int = 20
    grad_clip_adamw: float = 1.0
    scale_rows: int = 512
    layer_loss_cut: float = 0.6
    fibs_repo: str = ""
    fibs_revision: str = ""


def score(probe, x: torch.Tensor) -> np.ndarray:
    """Ensemble score of the saved probe on (N, L, d) activations in
    ``probe.layers`` order (one read token per row)."""
    return torch.cat([probe.score_batch(x[s:s + 4096, :, None].float(), torch.ones(len(x[s:s + 4096]), 1, dtype=torch.bool))
                      for s in range(0, len(x), 4096)]).numpy()


def by_source(s: np.ndarray, y: np.ndarray, src: np.ndarray) -> dict:
    return {k: {"n": int((src == k).sum()), "auroc": auroc(s[src == k], y[src == k])}
            for k in sorted(set(src)) if 0 < y[src == k].sum() < (src == k).sum()}


def main():
    cfg = FibsTrainConfig.load(OUT / "config.json")
    cfg.save()
    acts_dir, layers, device = REPO / cfg.acts_dir, cfg.probe_layers, torch.device("cuda")
    rows = load_rows(acts_dir)
    unknown = set(cfg.exclude_sources) - {r["source"] for r in rows}
    if unknown:
        raise ValueError(f"exclude_sources not in FIBSv1: {sorted(unknown)}")
    excluded = [r for r in rows if r["source"] in cfg.exclude_sources]
    rows = [r for r in rows if r["source"] not in cfg.exclude_sources]
    print(f"{len(rows)} rows kept, {len(excluded)} excluded ({', '.join(cfg.exclude_sources)})")
    split_validation(rows, cfg.n_val, cfg.val_seed)      # over the kept rows, not the store's split
    used_rows = balance_within_family(rows, cfg.train_seed)

    split = {r["i"]: "balanced_out" for r in rows}
    split.update({r["i"]: r["split"] for r in used_rows})
    split.update({r["i"]: "excluded" for r in excluded})
    pq.write_table(pa.table({"i": list(split), "split": list(split.values())}), OUT / "split.parquet")

    tr = [r for r in used_rows if r["split"] == "train"]
    va = [r for r in used_rows if r["split"] == "val"]
    x = {k: load_read_token(acts_dir, rs) for k, rs in (("train", tr), ("val", va), ("excluded", excluded))}
    y = {k: torch.tensor([r["label"] for r in rs], dtype=torch.float32)
         for k, rs in (("train", tr), ("val", va), ("excluded", excluded))}

    results = {}
    for arch in cfg.archs:
        model, used, record = train_layer_ensemble(
            arch, layers, x["train"].to(device), y["train"].to(device), x["val"].to(device), y["val"].to(device),
            cfg.train_hparams[arch], batch_size=cfg.train_batch_size, warmup_steps=cfg.warmup_steps,
            grad_clip=cfg.grad_clip_adamw, scale_rows=cfg.scale_rows, layer_loss_cut=cfg.layer_loss_cut,
            seed=cfg.train_seed)
        save_probe(arch, model, layers, used, OUT / "probes" / arch, {
            "model": cfg.model_id, "model_revision": cfg.model_revision,
            "trained_by": "experiments/2026-10-14-fibs-linear-no-sd/run.py (fibs_probes)",
            "fibs": {"repo": cfg.fibs_repo, "revision": cfg.fibs_revision, "exclude_sources": cfg.exclude_sources}})
        (OUT / f"{arch}_train.json").write_text(json.dumps(record, indent=1) + "\n")

        probe = load_probe(OUT / "probes" / arch)
        res = {"used_layers": used}
        for k, rs in (("val", va), ("excluded", excluded)):
            s, yy, src = score(probe, x[k]), y[k].numpy(), np.array([r["source"] for r in rs])
            res[k] = {"n": len(yy), "auroc": auroc(s, yy), "by_source": by_source(s, yy, src)}
        if abs(res["val"]["auroc"] - record["val"]["auroc"]) > 1e-3:
            raise ValueError(f"saved probe scores val AUROC {res['val']['auroc']:.4f}, "
                             f"trained model {record['val']['auroc']:.4f}: the save/load round trip differs")
        results[arch] = res
        print(f"{arch} (saved probe): val AUROC {res['val']['auroc']:.4f}; held-out social deduction "
              f"{res['excluded']['auroc']:.4f} " + ", ".join(f"{k} {v['auroc']:.4f}"
                                                            for k, v in res["excluded"]["by_source"].items()))
    (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
