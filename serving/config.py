"""ServingConfig: one experiment's publish + parity-check hyperparameters.

Light module — imports in both `.venv` and `.venv-vllm` (no heavy deps),
like :mod:`linear_probes.config` it extends.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from linear_probes.config import ExperimentConfig


@dataclass
class ServingConfig(ExperimentConfig):
    """Parameters for publishing a probe model and checking serve parity.

    Inherited ``model_id``/``layer`` name the base model and the probe layer;
    ``probe_paths[probe_name]`` (repo-relative) is the probe npz that becomes
    the published ``score`` head.
    """

    probe_name: str = "lr"
    hf_repo_id: str = ""     # target repo, e.g. reinthal/qwen3.5-27b-deception-probe-l16-...
    publish_dir: str = ""    # repo-relative build dir for the checkpoint (gitignored)
    private: bool = False     # create the HF repo public
    pooling: str = "MEAN"    # "MEAN" (span probes: all tokens) | "LAST" (DYL: the final token)

    # parity check (experiments/2026-10-07/run.py)
    n_parity: int = 64
    parity_source: str = "experiments/2026-09-22/transcripts_base.jsonl"
    parity_max_abs_dp: float = 0.05   # max |p_vllm - p_reference| allowed
    parity_min_r: float = 0.995       # min Pearson r on logits

    # distribution comparison: dataset name -> repo-relative jsonl of rollouts
    dist_sources: dict[str, str] = field(default_factory=dict)
    n_dist: int = 500                 # conversations per dataset
