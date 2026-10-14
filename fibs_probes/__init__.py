"""Train fibs probes (Hollinsworth et al. 2026, arXiv:2610.12445) on stored
read-token activations over FIBSv1, in the paper's ``probe_inference``
directory format.

  data   rows.parquet + layer_{L}.pt (the layout of
         experiments/2026-10-13-fibs-acts/): row selection, the validation
         hold-out, per-family class balance, activation loading
  train  the per-layer linear / MLP ensembles with the paper's recipe, saved
         so ``probe_inference.load_probe`` reads them back

The paper's ``probe_inference`` package is vendored as a git submodule
(vendor/caught-in-the-act-probes); importing this package puts its ``src`` on
``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROBE_SRC = Path(__file__).resolve().parents[1] / "vendor" / "caught-in-the-act-probes" / "src"
if str(_PROBE_SRC) not in sys.path:
    sys.path.insert(0, str(_PROBE_SRC))

from fibs_probes.data import (balance_within_family, load_layer_acts, load_rows,  # noqa: E402
                              split_validation)
from fibs_probes.train import PER_LAYER, auroc, save_probe, train_layer_ensemble  # noqa: E402

__all__ = ["PER_LAYER", "auroc", "balance_within_family", "load_layer_acts", "load_rows", "save_probe",
           "split_validation", "train_layer_ensemble"]
