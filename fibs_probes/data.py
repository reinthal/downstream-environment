"""FIBSv1 rows and their stored read-token activations.

Layout (written by experiments/2026-10-13-fibs-acts/collect.py):
  rows.parquet    one line per FIBSv1 row: source (family), pair_id,
                  followup_deceptive (the label), dropped, act_index, ...
  layer_{L}.pt    (n_kept, d_model) bf16, row ``act_index`` of the kept rows

The hold-out and the class balance are copied from
experiments/2026-10-13-fibs-basharena (modal_app.py::prepare_fibs and
train.py::balance_within_family), the paper's A.6.
"""
from __future__ import annotations

import random
from pathlib import Path

import pyarrow.parquet as pq
import torch


def load_rows(acts_dir: Path) -> list[dict]:
    """The kept rows (not over the length cap) in ``act_index`` order, with
    ``label`` = 1 on falsehood completions (follow-up deceptive)."""
    tbl = pq.read_table(acts_dir / "rows.parquet", columns=["row", "source", "pair_id", "quadrant",
                                                            "followup_deceptive", "dropped", "act_index"])
    rows = [r for r in tbl.to_pylist() if not r["dropped"]]
    rows.sort(key=lambda r: r["act_index"])
    if [r["act_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("act_index of the kept rows is not 0..n-1")
    for r in rows:
        r["label"] = int(r["followup_deceptive"])
    return rows


def split_validation(rows: list[dict], n_val: int, seed: int, log=print) -> None:
    """Set ``split`` on every row: a class-balanced validation hold-out of
    ``n_val`` rows sharing no ``pair_id`` with training (paper A.6); pair
    groups are visited in seeded order and taken while they fit."""
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for j, r in enumerate(rows):
        groups.setdefault(r["pair_id"] or f"_row{j}", []).append(j)
    keys = list(groups)
    rng.shuffle(keys)
    val, counts = set(), [0, 0]
    for k in keys:
        members = groups[k]
        lab = [rows[j]["label"] for j in members]
        if sum(counts) + len(members) > n_val:
            continue
        if counts[1] + sum(lab) > n_val // 2 or counts[0] + len(lab) - sum(lab) > n_val // 2:
            continue
        val.update(members)
        counts[0] += len(lab) - sum(lab)
        counts[1] += sum(lab)
        if sum(counts) >= n_val:
            break
    for j, r in enumerate(rows):
        r["split"] = "val" if j in val else "train"
    log(f"validation {counts[0]} honest + {counts[1]} deceptive of {len(rows)} rows")


def balance_within_family(rows: list[dict], seed: int, log=print) -> list[dict]:
    """Paper A.6: the training rows are class-balanced within each family
    (``source``); the majority class of each family is subsampled (seeded).
    Validation rows are untouched."""
    rng = random.Random(seed)
    by_fam: dict[str, dict[int, list[dict]]] = {}
    for r in rows:
        if r["split"] == "train":
            by_fam.setdefault(r["source"], {}).setdefault(r["label"], []).append(r)
    kept, dropped = [], 0
    for fam, by_label in sorted(by_fam.items()):
        n = min(len(by_label.get(0, [])), len(by_label.get(1, [])))
        for lab in (0, 1):
            items = by_label.get(lab, [])
            rng.shuffle(items)
            kept.extend(items[:n])
            dropped += len(items) - n
    log(f"class balance within family: kept {len(kept)} training rows, dropped {dropped}")
    keep_ids = {r["act_index"] for r in kept}
    return [r for r in rows if r["split"] == "val" or r["act_index"] in keep_ids]


def load_layer_acts(acts_dir: Path, rows: list[dict], layers: list[int], log=print) -> torch.Tensor:
    """(len(rows), len(layers), d_model) bf16, in ``rows`` order."""
    idx = torch.tensor([r["act_index"] for r in rows])
    out = []
    for L in layers:
        out.append(torch.load(acts_dir / f"layer_{L}.pt", weights_only=True)[idx])
        log(f"  loaded layer {L}")
    return torch.stack(out, dim=1)
