"""Score the saved window activations with the paper's four probes (CPU or
one GPU; no subject model needed) and join the DYL baseline.

  scores_{tag}.csv   one row per announcement: game_id, round, position,
                     speaker, role, {arch}_yes, {arch}_no, {arch}_contrast
                     per arch (contrast = lie prefill minus the other),
                     dyl_logodds
  raw_fu_{tag}.json  the primary probe's contrast in the auditor's raw-score
                     format: {game_id: [{round, position, player, score}]},
                     1-based round / position (auditor.run reads it)

Run:  uv run --no-sync python experiments/2026-10-12-fibs-probes/score_probes.py --arm 3p
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import torch

from common import OUT, load_config, probe_dir, read_jsonl, tag

from probe_inference import load_probe  # noqa: E402  (vendor package, via common's sys.path)
from probe_inference.score import build_read_mask  # noqa: E402


def score_one(probe, entry: dict) -> float:
    acts = entry["acts"]                      # (layers, W, d) bf16
    layers = probe.layers
    mask = build_read_mask(probe.read_window, entry["prompt_mask"], entry["completion_mask"],
                           torch.tensor(0))
    return probe.score({L: acts[i] for i, L in enumerate(layers)}, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    t = tag(args.arm, args.pilot)
    index = json.loads((OUT / f"acts_{t}" / "index.json").read_text())
    if index["layers"] != cfg.probe_layers or index["prefills"] != cfg.prefills:
        raise ValueError("acts index does not match config.json")
    (other,) = [p for p in cfg.prefills if p != cfg.lie_prefill]
    probes = {arch: load_probe(probe_dir(cfg, arch)) for arch in cfg.probe_archs}
    for arch, p in probes.items():
        if tuple(p.layers) != tuple(cfg.probe_layers):
            raise ValueError(f"{arch} probe layers {p.layers} != config {cfg.probe_layers}")

    recs = read_jsonl(OUT / f"transcripts_{t}.jsonl")
    dropped = {(e["game_id"], e["key"]) for e in index["turns"] if e["dropped"]}
    dyl = {}
    dyl_path = OUT / f"dyl_{t}.csv"
    if dyl_path.exists():
        for r in csv.DictReader(open(dyl_path)):
            dyl[(r["game_id"], int(r["round"]), int(r["position"]))] = r["dyl_logodds"]

    rows, raw = [], defaultdict(list)
    cache: dict[str, dict] = {}
    for r in recs:
        key = f"r{r['round']}p{r['position']}"
        if (r["game_id"], key) in dropped:
            continue
        if r["game_id"] not in cache:
            cache = {r["game_id"]: torch.load(OUT / f"acts_{t}" / f"{r['game_id']}.pt", weights_only=True)}
        entries = cache[r["game_id"]][key]
        row = {"game_id": r["game_id"], "round": r["round"], "position": r["position"],
               "speaker": r["speaker"], "role": r["role"]}
        for arch, p in probes.items():
            s = {prefill: score_one(p, entries[prefill]) for prefill in cfg.prefills}
            row[f"{arch}_yes"], row[f"{arch}_no"] = s["Yes."], s["No."]
            row[f"{arch}_contrast"] = s[cfg.lie_prefill] - s[other]
        row["dyl_logodds"] = dyl.get((r["game_id"], r["round"], r["position"]), "")
        rows.append(row)
        raw[r["game_id"]].append({"round": r["round"] + 1, "position": r["position"] + 1,
                                  "player": r["speaker"],
                                  "score": round(row[f"{cfg.primary_arch}_contrast"], 4)})
    with open(OUT / f"scores_{t}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for v in raw.values():
        v.sort(key=lambda x: (x["round"], x["position"]))
    (OUT / f"raw_fu_{t}.json").write_text(json.dumps(raw, indent=1) + "\n")
    print(f"scored {len(rows)} announcements ({len(dropped)} dropped) -> scores_{t}.csv, raw_fu_{t}.json")


if __name__ == "__main__":
    main()
