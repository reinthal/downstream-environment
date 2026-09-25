"""Experiment-local entry point for the grade-annotated outside auditor.

The auditor itself lives in the `auditor` package (code); everything
run-specific is in this folder's config.json (config). Equivalent to:

  python -m auditor.run --config experiments/2026-09-25/config.json --arm base

Run:  CUDA_VISIBLE_DEVICES=4,5,6,7 PATH=$PWD/.venv-vllm/bin:$PATH \
          .venv-vllm/bin/python experiments/2026-09-25/run_auditor.py --arm base
Smoke: add  --games 2
"""
from __future__ import annotations

import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))

from auditor.run import main  # noqa: E402

if __name__ == "__main__":
    main(default_config=OUT / "config.json")
