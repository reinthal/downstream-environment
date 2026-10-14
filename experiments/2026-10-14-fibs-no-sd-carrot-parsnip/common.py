"""Shared config and paths for 2026-10-14-fibs-no-sd-carrot-parsnip.

Probe conditions (``cfg.conditions``), each trained with every seed of
``cfg.train_seeds`` and every arch of ``cfg.archs``:
  no_sd        FIBSv1 without ``cfg.exclude_sources`` (train and validation)
  random_drop  FIBSv1 without ``cfg.n_drop`` random non-validation rows
Both share one validation hold-out. A probe is named ``{condition}_s{seed}``;
its directory is ``probes_main/{name}/{arch}`` (probe_inference format).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import fibs_probes  # noqa: E402,F401  (puts the vendored probe_inference on sys.path)
from linear_probes.config import ExperimentConfig  # noqa: E402


@dataclass
class NoSdConfig(ExperimentConfig):
    model_revision: str = ""
    acts_dir: str = ""                      # FIBSv1 read-token store (rows.jsonl + acts_{k}.pt)
    probe_layers: list[int] = field(default_factory=list)
    exclude_sources: list[str] = field(default_factory=list)   # left out of train and val in no_sd
    n_drop: int = 0                         # random_drop: rows dropped = rows of exclude_sources
    conditions: list[str] = field(default_factory=list)
    train_seeds: list[int] = field(default_factory=list)
    archs: list[str] = field(default_factory=list)
    primary_arch: str = "linear"
    n_val: int = 5000
    val_seed: int = 0
    train_hparams: dict = field(default_factory=dict)
    train_batch_size: int = 32
    warmup_steps: int = 20
    grad_clip_adamw: float = 1.0
    scale_rows: int = 512
    layer_loss_cut: float = 0.6
    fibs_repo: str = ""
    fibs_revision: str = ""
    pc_min_val_auroc: float = 0.95          # positive control: every probe's fibs validation AUROC
    # ── Carrot-Parsnip announcements (stored activations of 2026-10-12) ──
    cp_dir: str = ""
    main_arms: list[str] = field(default_factory=list)
    cc_arms: list[str] = field(default_factory=list)
    pilot_arms: list[str] = field(default_factory=list)
    prefills: list[str] = field(default_factory=list)
    lie_prefill: str = "Yes."               # contrast = score(lie prefill) - score(the other)
    published_archs: list[str] = field(default_factory=list)   # reference columns from cp_dir/scores_{arm}.csv
    # ── analysis ──
    delta: float = 0.10                     # Test 1: gap upper bound < delta
    n_boot: int = 2000
    boot_seed: int = 0
    fpr_quantile: float = 0.99              # alpaca 1%-FPR cutpoint


def load_config() -> NoSdConfig:
    return NoSdConfig.load(OUT / "config.json")


def probe_names(cfg: NoSdConfig) -> list[str]:
    return [f"{c}_s{s}" for c in cfg.conditions for s in cfg.train_seeds]
