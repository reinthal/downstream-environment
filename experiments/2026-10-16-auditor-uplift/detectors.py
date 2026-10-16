"""Detector scores shown to the auditor, all in layer-48 raw units.

From the raw values of one message ({l32, l48, rand}: the two follow-up
probes' log-odds and the activation's projection on the chosen random
direction) and calibration.json (calibrate.py):

  l32     layer-32 probe, affinely rescaled to layer 48's mean / s.d.
  l48     layer-48 probe, as is
  rand    the random direction (checked to be ~uninformative), rescaled
  mixNN   a * z(l48) + (1 - a) * z(rand), rescaled; a set so the probe
          alone names the Parsnip in ~NN% of calibration games ("mix" = 50%)
  oracle  a draw from layer 48's own calibration scores of the message's
          true role and kind (Parsnip pool for the Parsnip, Carrot pool
          for Carrots)
  inv     the same with roles swapped onto a decoy: one Carrot per game
          (seeded) draws from the Parsnip pool, the Parsnip from the Carrot
          pool

z(.) and the rescale are per message kind (announcement / answer): each
detector's calibration mean / s.d. for that kind, mapped onto layer 48's
mean / s.d. for that kind, so every detector shows layer 48's levels for
announcements and for answers. Oracle draws are seeded per message, so a replay of the same
message gets the same score.
"""
from __future__ import annotations

import json
import random

from common import OUT

KIND = {"announcement": "announcement", "answer": "answer", "private_answer": "answer"}


class Detectors:
    def __init__(self, path=OUT / "calibration.json"):
        self.c = json.loads(path.read_text())

    def decoy(self, seed: int, roles: dict[str, str]) -> str:
        carrots = sorted(n for n, r in roles.items() if r == "Carrot")
        return random.Random(f"decoy-{seed}").choice(carrots)

    def z(self, name: str, x: float, kind: str) -> float:
        m, s = self.c["stats"][KIND[kind]][name]
        return (x - m) / s

    def _rescaled(self, z: float, kind: str) -> float:
        m, s = self.c["stats"][KIND[kind]]["l48"]
        return m + s * z

    def score(self, det: str, raw: dict, *, role: str, kind: str, speaker: str, decoy: str, key: str) -> float:
        if det == "l48":
            return raw["l48"]
        if det in ("l32", "rand"):
            return self._rescaled(self.z(det, raw[det], kind), kind)
        if det.startswith("mix"):
            a = self.c["mix_alpha"][det]
            mixed = a * self.z("l48", raw["l48"], kind) + (1 - a) * self.z("rand", raw["rand"], kind)
            return self._rescaled(self.z(f"{det}_z", mixed, kind), kind)
        if det in ("oracle", "inv"):
            shown = role
            if det == "inv":
                shown = "Parsnip" if speaker == decoy else "Carrot"
            pool = self.c["pools"][shown][KIND[kind]]
            return random.Random(f"{det}-{key}").choice(pool)
        raise ValueError(f"unknown detector {det!r}")
