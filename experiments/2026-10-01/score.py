"""Apply the follow-up probe(s) to every activation set and calibrate (CPU).

  scores_{base,steered,nothink,noparsnip,instructed}.csv   game turns
        (09-29 key columns + followup_full [+ followup_on_full] [+ variant])
  scores_alpaca.csv                                         control
  suspicion_scale.json   auditor 1-7 grades: alpaca quantiles
        [.25 .50 .75 .90 .97 .99] of the main probe (grade 7 = 1%-FPR threshold)

The column is named *_full so auditor.run's `<cond>_<span>` lookup works with
span "full"; the probe actually reads only the follow-up "No".

Run:  .venv/bin/python experiments/2026-10-01/score.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import GAME_SETS, GAMES  # noqa: E402
from common import OUT, read_jsonl  # noqa: E402

from auditor.config import AuditorConfig  # noqa: E402
from auditor.scale import fit_cutpoints, save_scale  # noqa: E402
from linear_probes.probes import LinearProbe  # noqa: E402

KEYS = ["game_id", "seed_group", "round", "position", "speaker", "role"]


def main():
    cfg = AuditorConfig.load(OUT / "config.json")
    probes = {"followup": LinearProbe.load(str(OUT / "probe_followup_off.npz"))}
    if (OUT / "probe_followup_on.npz").exists():
        probes["followup_on"] = LinearProbe.load(str(OUT / "probe_followup_on.npz"))

    def scores(name):
        X = np.load(OUT / f"activations_followup_{name}.npz")["X"]
        return {p: probe.predict_proba(X) for p, probe in probes.items()}

    alpaca = scores("alpaca")
    with open(OUT / "scores_alpaca.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"{p}_full" for p in probes])
        w.writerows(zip(*(np.round(alpaca[p], 6) for p in probes)))
    cuts = fit_cutpoints(alpaca["followup"], cfg.scale_quantiles)
    save_scale(OUT / "suspicion_scale.json", span=cfg.span, quantiles=cfg.scale_quantiles,
               cutpoints={"followup": cuts}, grade_legend=cfg.grade_legend,
               n_control=len(alpaca["followup"]))
    print("1%-FPR threshold:", round(cuts[-1], 4), flush=True)

    for name in GAME_SETS:
        if not (OUT / f"activations_followup_{name}.npz").exists():
            continue
        recs = read_jsonl(GAMES / f"transcripts_{name}.jsonl")
        s = scores(name)
        extra = ["variant"] if "variant" in recs[0] else []
        with open(OUT / f"scores_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([*KEYS, "silent", *(f"{p}_full" for p in probes), *extra])
            for i, r in enumerate(recs):
                w.writerow([*(r[k] for k in KEYS), int(not r["announcement"]),
                            *(f"{s[p][i]:.6f}" for p in probes), *(r[k] for k in extra)])
        print("wrote", OUT / f"scores_{name}.csv", flush=True)


if __name__ == "__main__":
    main()
