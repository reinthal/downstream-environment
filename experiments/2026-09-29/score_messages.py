"""Score this experiment's new transcripts (noparsnip / nothink / instructed)
with the 2026-09-22 instructed-pairs probes, exactly as 09-22 scored its games:
re-encode the full generation context, mean-pool layer-16 activations over the
full assistant turn and the public (post-</think>) span, apply LR, DoM and the
per-seed-group random direction.

Output: scores_{arm}.csv (09-22 columns, + `variant` for instructed) and
message_activations_{arm}.npz (untracked; used by analyze.py's role probe).

Run:  CUDA_VISIBLE_DEVICES=4 .venv/bin/python experiments/2026-09-29/score_messages.py --arm nothink
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "2026-09-22"))

from auditor.config import AuditorConfig  # noqa: E402
from common import read_jsonl  # noqa: E402
from linear_probes import LMProbe  # noqa: E402
from score_messages import RandomProbe  # noqa: E402  (09-22)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    args = ap.parse_args()
    cfg = AuditorConfig.load(OUT / "config.json")

    recs = read_jsonl(OUT / f"transcripts_{args.arm}.jsonl")
    convs = [list(r["input_messages"]) + [{"role": "assistant", "content": r["announcement"],
                                           "reasoning_content": r["reasoning"]}]
             for r in recs]
    lr = LMProbe.from_config(cfg, probe="lr")
    dom = lr.with_probe(cfg.path(cfg.probe_paths["dom"]))
    acts = lr.collect(convs, spans=("full", "public"))
    X_full, X_pub = acts[:, 0], acts[:, 1]
    np.savez(OUT / f"message_activations_{args.arm}.npz", X_full=X_full, X_pub=X_pub,
             layer=cfg.layer)

    X_train = np.load(cfg.path(cfg.probe_paths["train_activations"]))[f"layer_{cfg.layer}"]
    rand = {g: RandomProbe(g, lr.probe.mu, lr.probe.sd, X_train) for g in range(cfg.seed_groups)}
    cols = {"lr_pub": lr.probe.predict_proba(X_pub), "lr_full": lr.probe.predict_proba(X_full),
            "dom_pub": dom.probe.predict_proba(X_pub), "dom_full": dom.probe.predict_proba(X_full)}
    extra = ["variant"] if "variant" in recs[0] else []
    path = OUT / f"scores_{args.arm}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game_id", "seed_group", "round", "position", "speaker", "role", "silent",
                    *cols, "rand_pub", "rand_full", *extra])
        for i, r in enumerate(recs):
            g = r["seed_group"]
            w.writerow([r["game_id"], g, r["round"], r["position"], r["speaker"], r["role"],
                        int(not r["announcement"]), *(f"{c[i]:.6f}" for c in cols.values()),
                        f"{rand[g].predict_proba(X_pub[i:i + 1])[0]:.6f}",
                        f"{rand[g].predict_proba(X_full[i:i + 1])[0]:.6f}",
                        *(r[k] for k in extra)])
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
