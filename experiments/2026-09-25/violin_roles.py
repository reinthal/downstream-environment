"""Exploratory start of the role-split experiment: one violin plot.

Three groups of LR probe scores (full span): alpaca control, Carrot
messages, Parsnip messages (base arm), with the 1%-FPR-on-alpaca threshold.
Question on deck: are Parsnips more probe-deceptive than Carrots?
Reuses the 2026-09-22 palette/violin helper and existing scores — no GPU.

Run:  uv run --no-sync python experiments/2026-09-25/violin_roles.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
PREV = OUT.parents[0] / "2026-09-22"
sys.path.insert(0, str(PREV))

from analyze_judge import ALPACA, CONTROL_C, DECEPTIVE_C, HONEST_C, violin  # noqa: E402

COL = "lr_full"


def main():
    FIG.mkdir(exist_ok=True)
    with open(PREV / "scores_base.csv") as f:
        scores = list(csv.DictReader(f))
    with open(ALPACA) as f:
        alp = np.array([float(r[COL]) for r in csv.DictReader(f)])

    by_role = {role: np.array([float(s[COL]) for s in scores if s["role"] == role])
               for role in ("Carrot", "Parsnip")}
    thr = float(np.quantile(alp, 0.99))

    groups = [("alpaca control", alp, CONTROL_C),
              ("Carrot", by_role["Carrot"], HONEST_C),
              ("Parsnip", by_role["Parsnip"], DECEPTIVE_C)]
    path = FIG / "violin_roles_base_lr_full.png"
    violin(groups, thr, "LR probe score by role — base arm, full span", path)
    print(f"figure -> {path}")

    print(f"alpaca p99 threshold = {thr:.3f}")
    for name, g, _ in groups:
        print(f"{name:15s} n={len(g):4d}  mean={g.mean():.3f}  "
              f"median={np.median(g):.3f}  frac>=thr={float((g >= thr).mean()):.3f}")


if __name__ == "__main__":
    main()
