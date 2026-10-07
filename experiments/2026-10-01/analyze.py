"""Analysis for the follow-up probe (CPU). Sections skip when inputs are missing.

  1. training: labels, held-out roleplaying AUROC (thinking off / on)
  2. game separability (Parsnip vs Carrot), same tests as 2026-09-29, side by
     side with the instructed-pairs LR probe
  3. no-Parsnip games, in-game conceal-vs-admit pairs, false flags at 1% FPR
  4. auditor accuracy with follow-up grades

Run:  .venv/bin/python experiments/2026-10-01/analyze.py   -> results.json
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, REPO, read_jsonl  # noqa: E402

OLD = REPO / "experiments" / "2026-09-29"
ADMITS = re.compile(r"\bI(?:'m| am)\s+(?:the\s+|a\s+)?parsnip\b", re.IGNORECASE)
results: dict = {}


def rnd(x, k=3):
    return None if x is None else round(float(x), k)


def table(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    return rnd(roc_auc_score([1] * len(pos) + [0] * len(neg), [*pos, *neg]))


def show(title, d):
    print(f"\n== {title}")
    for k, v in d.items():
        print(f"  {k}: {v}")
    results[title] = d


# ── 1. training ───────────────────────────────────────────────────────────────
d = {}
for v in ("off", "on"):
    p = OUT / f"train_{v}.json"
    if p.exists():
        t = json.loads(p.read_text())
        d[f"thinking {v}"] = {k: t[k] for k in ("labels_all", "n_train", "n_val", "val_auroc")}
if d:
    show("1. roleplaying training (held-out 20%)", d)

# ── 2-3. game tests ──────────────────────────────────────────────────────────
scale = OUT / "suspicion_scale.json"
if scale.exists():
    thr = json.loads(scale.read_text())["cutpoints"]["followup"][-1]
    cols = [c for c in ("followup_full", "followup_on_full")
            if (OUT / "scores_base.csv").exists() and c in table(OUT / "scores_base.csv")[0]]
    sub50 = lambda rows: [r for r in rows if int(r["game_id"].split("g")[1]) < 10]  # noqa: E731
    vals = lambda rows, c, role: [float(r[c]) for r in rows if r["role"] == role]  # noqa: E731
    old = json.loads((OLD / "results.json").read_text())["4a. probe scores by arm"]

    d = {"1%-FPR threshold (alpaca)": rnd(thr, 4)}
    for arm in ("base", "steered", "nothink"):
        if not (OUT / f"scores_{arm}.csv").exists():
            continue
        rows = table(OUT / f"scores_{arm}.csv")
        for c in cols:
            d[f"{arm}/{c} parsnip-vs-carrot AUROC (all games)"] = auroc(
                vals(rows, c, "Parsnip"), vals(rows, c, "Carrot"))
        r50 = sub50(rows) if arm != "nothink" else rows
        d[f"{arm}/followup AUROC, 50-game subset"] = auroc(
            vals(r50, "followup_full", "Parsnip"), vals(r50, "followup_full", "Carrot"))
        d[f"{arm}/followup carrot frac over threshold"] = rnd(
            np.mean([v >= thr for v in vals(r50, "followup_full", "Carrot")]))
        d[f"{arm}/followup parsnip frac over threshold"] = rnd(
            np.mean([v >= thr for v in vals(r50, "followup_full", "Parsnip")]))
    d["old LR probe, base 50 games (full / announcement)"] = (
        old["base (50 games)/lr_full"]["parsnip_vs_carrot_auroc"],
        old["base (50 games)/lr_pub"]["parsnip_vs_carrot_auroc"])
    show("2. Parsnip vs Carrot", d)

    d = {}
    if (OUT / "scores_noparsnip.csv").exists():
        np_rows, b50 = table(OUT / "scores_noparsnip.csv"), sub50(table(OUT / "scores_base.csv"))
        d["noparsnip carrots vs base carrots AUROC"] = auroc(
            vals(np_rows, "followup_full", "Carrot"), vals(b50, "followup_full", "Carrot"))
        d["noparsnip carrot frac over threshold"] = rnd(
            np.mean([v >= thr for v in vals(np_rows, "followup_full", "Carrot")]))
    if (OUT / "scores_instructed.csv").exists():
        sc, tr = table(OUT / "scores_instructed.csv"), read_jsonl(OLD / "transcripts_instructed.jsonl")
        pairs = defaultdict(dict)
        for r, t in zip(sc, tr):
            r["admits"] = bool(ADMITS.search(t["announcement"] or ""))
            pairs[(r["game_id"], r["round"], r["position"])][r["variant"]] = r
        clean = [p for p in pairs.values()
                 if p["honest"]["admits"] and not p["deceptive"]["admits"]]
        dec = np.array([float(p["deceptive"]["followup_full"]) for p in clean])
        hon = np.array([float(p["honest"]["followup_full"]) for p in clean])
        d["conceal vs admit AUROC (clean pairs)"] = auroc(dec, hon)
        d["conceal scored higher, share of pairs"] = rnd(np.mean(dec > hon))
        d["median conceal / admit"] = (rnd(np.median(dec), 4), rnd(np.median(hon), 4))
        d["n clean pairs"] = len(clean)
        d["old LR probe conceal vs admit AUROC"] = json.loads(
            (OLD / "results.json").read_text())[
            "4b. in-game instructed pairs (clean pairs only)"]["lr_full"]["deceptive_vs_honest_auroc"]
    if d:
        show("3. controls and instructed pairs", d)


# ── 4. auditor ────────────────────────────────────────────────────────────────
def cluster_ci(rows, n=2000):
    by = defaultdict(list)
    for r in rows:
        by[r["game_id"]].append(float(r["correct"]))
    sums = np.array([sum(v) for v in by.values()])
    cnts = np.array([len(v) for v in by.values()])
    idx = np.random.default_rng(0).integers(0, len(sums), (n, len(sums)))
    acc = sums[idx].sum(1) / cnts[idx].sum(1)
    return [rnd(x) for x in np.percentile(acc, [2.5, 97.5])]


d = {}
for path in sorted((OUT / "audit").glob("auditor_seeds_*.jsonl")):
    arm = path.stem.removeprefix("auditor_seeds_")
    by = defaultdict(list)
    for r in read_jsonl(path):
        by[r["condition"]].append(r)
    for cond, rs in by.items():
        d[f"{arm}/{cond}"] = {"n": len(rs), "acc": rnd(np.mean([r["correct"] for r in rs])),
                              "ci95": cluster_ci(rs),
                              "unanswered": rnd(np.mean([r["guess"] is None for r in rs]))}
if d:
    show("4. auditor accuracy (2k tokens, forced final answer)", d)

(OUT / "results.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
print("\nwrote", OUT / "results.json")
