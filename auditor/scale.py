"""Calibrated 1-7 suspicion scale.

Per detector, the 6 grade cutpoints are quantiles of that detector's scores
on a benign control set (alpaca rollouts), so a grade means "how rare on
benign text" and detectors on wildly different score ranges become
comparable under one prompt. Fitted once per experiment
(`suspicion_scale.json`); detectors whose directions differ per seed group
(the random control) get one cutpoint list per group.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_cutpoints(control_scores, quantiles) -> list[float]:
    return np.quantile(np.asarray(control_scores), list(quantiles)).tolist()


def grade(score: float, cuts) -> int:
    """Map a score to grade 1..len(cuts)+1. side="right" so a score equal to
    a cutpoint lands in the higher grade, matching the `score >= threshold`
    flagging convention used across the repo."""
    return int(np.searchsorted(np.asarray(cuts), score, side="right")) + 1


def save_scale(path: str | Path, *, span: str, quantiles, cutpoints: dict,
               grade_legend: str, n_control: int) -> Path:
    """cutpoints: detector name -> list of 6 floats, or (per-seed-group
    detectors) -> list of such lists."""
    path = Path(path)
    path.write_text(json.dumps(
        {"span": span, "quantiles": list(quantiles), "n_control": n_control,
         "grade_legend": grade_legend, "cutpoints": cutpoints}, indent=1) + "\n")
    return path


def load_scale(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
