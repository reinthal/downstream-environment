"""Outside-auditor package: identify the Parsnip from public transcripts.

Code lives here; every run's parameters live in an experiment folder's
`config.json` (an :class:`AuditorConfig`). The latest auditor presents probe
read-outs as a calibrated 1-7 suspicion grade (see :mod:`auditor.scale`).

Modules (import-light; heavy deps stay behind the run entry point):
  config   AuditorConfig dataclass (extends ExperimentConfig)
  scale    alpaca-quantile cutpoints: fit / grade / save / load
  prompts  auditor prompt templates, transcript builder, answer parsing
  run      the runnable audit (vLLM env):  python -m auditor.run --config ...
"""
from auditor.config import AuditorConfig  # noqa: F401
from auditor.scale import fit_cutpoints, grade, load_scale, save_scale  # noqa: F401
