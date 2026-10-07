"""Shared inputs for the detector analysis: game messages and exact per-message
detector scores, all aligned to transcripts_{arm}.jsonl order (2026-09-29).

Detectors (scores as log-odds for probes, 1-7 for the judge):
  fu      follow-up probe (2026-10-01): layer-18 activation on the follow-up "No"
  lr      instructed-pairs LR probe (2026-09-22): layer-16, mean over the full turn
  godmode god-mode text judge (2026-09-25): 1-7 rating with private reasoning
  rand_fu / rand_lr  random directions on the fu / lr activations (null baseline)
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
EXP = OUT.parent
REPO = EXP.parent
E29, E01 = EXP / "2026-09-29", EXP / "2026-10-01"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from linear_probes.probes import LinearProbe  # noqa: E402

ARMS = ("base", "steered", "nothink")


@lru_cache(maxsize=None)
def messages(arm: str) -> list[dict]:
    rows = []
    for line in open(E29 / f"transcripts_{arm}.jsonl"):
        r = json.loads(line)
        rows.append({k: r[k] for k in ("game_id", "seed_group", "round", "position",
                                       "speaker", "role")}
                    | {"ann_len": len(r["announcement"] or ""), "reas_len": len(r["reasoning"] or ""),
                       "silent": not r["announcement"]})
    return rows


@lru_cache(maxsize=None)
def activations(source: str, arm: str) -> np.ndarray:
    if source == "fu":
        return np.load(E01 / f"activations_followup_{arm}.npz")["X"]
    return np.load(E29 / f"message_activations_{arm}.npz")["X_full"]


# god-mode judge outputs with no parsable 1-7 score (8/1500 base, 5/1500
# steered in 2026-09-29) are scored as the scale midpoint
UNPARSED_JUDGE = 4

def _logodds(probe: LinearProbe, X: np.ndarray) -> np.ndarray:
    return ((X - probe.mu) / probe.sd) @ probe.w + probe.b


def scores(detector: str, arm: str, seed: int = 0) -> np.ndarray | None:
    """Per-message score aligned to messages(arm); None if unavailable."""
    if detector == "fu":
        return _logodds(LinearProbe.load(str(REPO / "probes" / "qwen3.5-27b_layer18_followup_logistic_regression.npz")), activations("fu", arm))
    if detector == "lr":
        p = LinearProbe.load(str(REPO / "probes" / "qwen3.5-27b_layer16_logistic_regression.npz"))
        return _logodds(p, activations("lr", arm))
    if detector == "godmode":
        path = E29 / f"judge_full_{arm}.jsonl"
        if not path.exists():
            return None
        by = {}
        for line in open(path):
            r = json.loads(line)
            by[(r["game_id"], r["round"], r["position"])] = UNPARSED_JUDGE if r["judge_score"] is None else r["judge_score"]
        return np.array([by[(m["game_id"], m["round"], m["position"])] for m in messages(arm)],
                        dtype=float)
    if detector.startswith("rand_"):
        X = activations(detector.removeprefix("rand_"), arm)
        # standardize with base-arm statistics so a direction means the same in every arm
        B = activations(detector.removeprefix("rand_"), "base")
        d = np.random.default_rng(seed).standard_normal(X.shape[1])
        return ((X - B.mean(0)) / (B.std(0) + 1e-6)) @ d / np.sqrt(X.shape[1])
    raise ValueError(detector)
