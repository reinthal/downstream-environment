"""Shared config and paths for the 2026-10-13 fibs activation collection."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
PROBE_SRC = REPO / "vendor" / "caught-in-the-act-probes" / "src"   # the paper's probe_inference package
for p in (REPO, PROBE_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from linear_probes.config import ExperimentConfig  # noqa: E402

QUADRANTS = ("honest_honest", "honest_deceptive", "deceptive_honest", "deceptive_deceptive")


@dataclass
class FibsActsConfig(ExperimentConfig):
    model_revision: str = "fc05daec18b0a78c049392ed2e771dde82bdf654"  # HF commit the paper's probes were trained on
    fibs_repo: str = "AlignmentResearch/fibs-v1"
    fibs_revision: str = "65dccf12934620bfe01db665ea3d7d0e3c9f1355"
    layers: list[int] = field(default_factory=lambda: [19, 27, 35, 43, 51, 58])  # decoder block outputs stored
    acts_max_len: int = 32768         # rendered rows longer than this are DROPPED, not trimmed (paper B.2)
    acts_batch_tokens: int = 16384    # padded-token budget per forward (bare decoder, 59 blocks on two A40s)
    acts_max_batch: int = 64          # rows per forward: 128 OOMed here at ~200-token rows (GDN state on an A40)
    n_shards: int = 4                 # one worker (two GPUs) per shard; rows are dealt out by length rank
    save_rows: int = 2000             # a chunk file is written once this many rows are collected
    save_tokens: int = 2_000_000      # ... or this many tokens, whichever comes first (long rows come first)
    shard_rows: int = 500             # merge: rows per read-token/acts_{k}.pt (the 3.8 store's shard size)
    n_val: int = 5000                 # merge: validation hold-out over all rows (paper A.6; the 3.8 store's rule)
    val_seed: int = 0
    render_workers: int = 32          # CPU processes of the render stage
    pilot_rows: int = 400             # seeded random rows, plus the longest kept row
    pilot_seed: int = 0
    pilot_check_rows: int = 8         # pilot: rows on which the hook is compared with hidden_states
    # ── check stage: the published linear ensemble on the stored activations ──
    probe_repo: str = "AlignmentResearch/probe-inference-weights"
    probe_revision: str = "05def1c13a70237cda3b717d785813918b8a22fa"   # tag camera-ready
    probe_path: str = "qwen3.5-27b/linear"


def load_config() -> FibsActsConfig:
    return FibsActsConfig.load(OUT / "config.json")


def model_path(cfg: FibsActsConfig) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(cfg.model_id, revision=cfg.model_revision)


def fibs_root(cfg: FibsActsConfig) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(cfg.fibs_repo, repo_type="dataset", revision=cfg.fibs_revision,
                                  allow_patterns=["data/*"])) / "data"


def acts_dir(pilot: bool) -> Path:
    return OUT / ("acts_fibs_pilot" if pilot else "acts_fibs")
