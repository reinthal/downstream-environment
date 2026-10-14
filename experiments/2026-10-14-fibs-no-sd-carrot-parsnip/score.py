"""Score the stored 2026-10-12 announcement / alpaca activations with every
trained probe (CPU; no subject model).

  scores_{arm}.csv   one row per announcement: game_id, round, position,
                     speaker, role, then {name}_{arch} = contrast (lie
                     prefill minus the other) per probe, and published_{arch}
                     copied from cfg.cp_dir/scores_{arm}.csv (2026-10-12)
  scores_alpaca.csv  idx, {name}_{arch}, published_{arch}

The read is ``score_one`` of experiments/2026-10-12-fibs-probes/score_probes.py.

Run:  uv run --no-sync python experiments/2026-10-14-fibs-no-sd-carrot-parsnip/score.py --arms 5p_pilot --seeds 0
"""
from __future__ import annotations

import argparse
import csv
import json

import torch

from common import OUT, REPO, load_config, probe_names

from probe_inference import load_probe  # noqa: E402
from probe_inference.score import build_read_mask  # noqa: E402


def score_one(probe, entry: dict) -> float:
    acts = entry["acts"]                      # (layers, W, d) bf16
    mask = build_read_mask(probe.read_window, entry["prompt_mask"], entry["completion_mask"], torch.tensor(0))
    return probe.score({L: acts[i] for i, L in enumerate(probe.layers)}, mask)


def contrasts(probes: dict, entries: dict, cfg) -> dict:
    (other,) = [p for p in cfg.prefills if p != cfg.lie_prefill]
    return {k: f"{score_one(p, entries[cfg.lie_prefill]) - score_one(p, entries[other]):.6f}" for k, p in probes.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="game arms and/or 'alpaca'")
    ap.add_argument("--seeds", type=int, nargs="*", help="pilot: subset of cfg.train_seeds (default: all)")
    args = ap.parse_args()
    cfg = load_config()
    if args.seeds is not None:
        cfg.train_seeds = args.seeds
    cp = REPO / cfg.cp_dir
    probes = {f"{n}_{a}": load_probe(OUT / "probes_main" / n / a) for n in probe_names(cfg) for a in cfg.archs}
    for k, p in probes.items():
        if tuple(p.layers) != tuple(cfg.probe_layers):
            raise ValueError(f"{k}: layers {p.layers} != config {cfg.probe_layers}")

    for arm in args.arms:
        if arm == "alpaca":
            pub = {r["idx"]: r for r in csv.DictReader(open(cp / "scores_alpaca.csv"))}
            out = []
            for part in sorted((cp / "acts_alpaca").glob("part*.pt")):
                for idx, entries in torch.load(part, weights_only=True).items():
                    row = {"idx": idx, **contrasts(probes, entries, cfg)}
                    row.update({f"published_{a}": pub[str(idx)][f"{a}_contrast"] for a in cfg.published_archs})
                    out.append(row)
            if len(out) != len(pub):
                raise ValueError(f"alpaca: {len(out)} stored replies, {len(pub)} published scores")
        else:
            index = json.loads((cp / f"acts_{arm}" / "index.json").read_text())
            if index["layers"] != cfg.probe_layers or index["prefills"] != cfg.prefills:
                raise ValueError(f"acts_{arm}/index.json does not match config.json")
            dropped = {(e["game_id"], e["key"]) for e in index["turns"] if e["dropped"]}
            pub = {(r["game_id"], r["round"], r["position"]): r for r in csv.DictReader(open(cp / f"scores_{arm}.csv"))}
            out, cache = [], {}
            with open(cp / f"transcripts_{arm}.jsonl") as f:
                recs = [json.loads(line) for line in f]
            for r in recs:
                key = f"r{r['round']}p{r['position']}"
                if (r["game_id"], key) in dropped:
                    continue
                if r["game_id"] not in cache:
                    cache = {r["game_id"]: torch.load(cp / f"acts_{arm}" / f"{r['game_id']}.pt", weights_only=True)}
                row = {"game_id": r["game_id"], "round": r["round"], "position": r["position"],
                       "speaker": r["speaker"], "role": r["role"],
                       **contrasts(probes, cache[r["game_id"]][key], cfg)}
                ref = pub[(r["game_id"], str(r["round"]), str(r["position"]))]
                row.update({f"published_{a}": ref[f"{a}_contrast"] for a in cfg.published_archs})
                out.append(row)
            if len(out) != len(pub):
                raise ValueError(f"{arm}: {len(out)} scored announcements, {len(pub)} in the 2026-10-12 scores")
        with open(OUT / f"scores_{arm}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"{arm}: {len(out)} rows x {len(probes)} probes -> scores_{arm}.csv", flush=True)


if __name__ == "__main__":
    main()
