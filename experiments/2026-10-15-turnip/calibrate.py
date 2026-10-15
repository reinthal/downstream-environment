"""Cutpoint for the probe arm: the cfg.fpr_quantile quantile of the live
contrast over every Carrot answer to the Turnip in the interrog arm
(PREREGISTRATION.md §3, "calibrate on carrots ... for interrogate runs").
The pilot writes cutpoint_pilot.json from the interrog pilot (provisional,
pilot only).

Run:  $R python experiments/2026-10-15-turnip/calibrate.py [--pilot]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import OUT, load_config, read_jsonl, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    t = tag("interrog", args.pilot)
    rows = read_jsonl(OUT / f"qa_{t}.jsonl")
    by_role = {r: np.array([q["detector"]["contrast"] for q in rows if q["target_role"] == r])
               for r in ("Carrot", "Parsnip")}
    carrot = by_role["Carrot"]
    if len(carrot) == 0:
        raise SystemExit("no Carrot answers in the interrog arm")
    cut = float(np.quantile(carrot, cfg.fpr_quantile))
    out = {
        "cutpoint": cut, "quantile": cfg.fpr_quantile, "source": f"qa_{t}.jsonl",
        "n_carrot_answers": int(len(carrot)), "n_carrot_above": int((carrot > cut).sum()),
        "n_parsnip_answers": int(len(by_role["Parsnip"])),
        "parsnip_flag_rate": float((by_role["Parsnip"] > cut).mean()) if len(by_role["Parsnip"]) else None,
        "n_games": len({q["game_id"] for q in rows}),
    }
    name = "cutpoint_pilot.json" if args.pilot else "cutpoint.json"
    (OUT / name).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
