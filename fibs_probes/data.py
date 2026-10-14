"""FIBSv1 rows and their stored read-token activations, one layout for every
model:

  <root>/rows.jsonl   one line per FIBSv1 row: i (position in the
                      concatenation of the quadrant parquets honest_honest,
                      honest_deceptive, deceptive_honest, deceptive_deceptive),
                      label (1 = falsehood completion), quadrant, source
                      (family), pair_id, _v32_source_content_md5, split
                      (train / val, the paper's A.6 hold-out), n_tokens,
                      window, dropped (over the length cap, not collected)
  <root>/acts_{k}.pt  {"i": list[int], "token": (n, layers, d) bf16, ...}:
                      the read token (before the answer's <|im_end|>)

Stores: the Qwen3.8-27B activations in the public-datasets bucket
(fibs/activations/Qwen3.8-27B/read-token/, written by
experiments/2026-10-13-fibs-basharena) and the Qwen3.5-27B ones in
experiments/2026-10-13-fibs-acts/acts_fibs/read-token/. Both hold layers 19,
27, 35, 43, 51, 58 in that order; ``i`` is the same row in both.

The hold-out and the class balance are copied from
experiments/2026-10-13-fibs-basharena (modal_app.py::prepare_fibs and
train.py::balance_within_family).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch


def load_rows(root: Path) -> list[dict]:
    """The collected rows (``dropped`` false) of ``root/rows.jsonl``, in ``i`` order."""
    with open(root / "rows.jsonl") as f:
        rows = [r for r in map(json.loads, f) if not r["dropped"]]
    rows.sort(key=lambda r: r["i"])
    return rows


def split_validation(rows: list[dict], n_val: int, seed: int, log=print) -> None:
    """Set ``split`` on every row: a class-balanced validation hold-out of
    ``n_val`` rows sharing no ``pair_id`` with training (paper A.6); pair
    groups are visited in seeded order and taken while they fit. Over all
    FIBSv1 rows in ``i`` order with seed 0 this reproduces the stores'
    ``split``; call it only to draw a hold-out over a subset."""
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
    keep_ids = {r["i"] for r in kept}
    return [r for r in rows if r["split"] == "val" or r["i"] in keep_ids]


def load_read_token(root: Path, rows: list[dict], log=print) -> torch.Tensor:
    """(len(rows), layers, d) bf16, in ``rows`` order. Every row must be in
    a shard (a dropped row is an error)."""
    want = {r["i"]: j for j, r in enumerate(rows)}
    out, filled = None, 0
    shards = sorted(root.glob("acts_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not shards:
        raise FileNotFoundError(f"no acts_*.pt in {root}")
    for k, path in enumerate(shards):
        blob = torch.load(path, weights_only=True)
        sel = [(n, want[i]) for n, i in enumerate(blob["i"]) if i in want]
        if not sel:
            continue
        if out is None:
            out = torch.empty((len(rows), *blob["token"].shape[1:]), dtype=blob["token"].dtype)
        src, dst = zip(*sel)
        out[list(dst)] = blob["token"][list(src)]
        filled += len(sel)
        if (k + 1) % 50 == 0:
            log(f"  read {k + 1}/{len(shards)} shards")
    if filled != len(rows):
        raise ValueError(f"{len(rows) - filled} of {len(rows)} rows are in no shard of {root}")
    return out
