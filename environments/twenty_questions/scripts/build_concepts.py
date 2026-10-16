"""Build the bounded-mode concept sets from THINGS (Hebart et al. 2019).

Downloads ``concepts-metadata_things.tsv`` from the THINGS OSF project
(https://osf.io/jum2f/, file um6a9), checks its sha256, drops every homonym
(a word listed with more than one meaning, e.g. bat = animal / club: 27
words) so a concept word names one thing, and writes seeded subsamples:

    concepts/things_n{N}_s{seed}.json      N in --sizes (and "all")

Run:  uv run --directory environments/twenty_questions python scripts/build_concepts.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import random
import httpx
from pathlib import Path

URL = "https://osf.io/download/um6a9/"
SHA256 = "3f05b6b748d1e1b685e15130d2487bff63f86e1f4da94a2a5d9e7dec073a48eb"
OUT = Path(__file__).resolve().parent.parent / "concepts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    resp = httpx.get(URL, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    data = resp.content
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"THINGS tsv sha256 {digest} != expected {SHA256}")
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t"))
    if len(rows) != 1854:
        raise SystemExit(f"expected 1854 THINGS concepts, got {len(rows)}")
    counts = collections.Counter(r["Word"] for r in rows)
    concepts = [{"word": r["Word"], "definition": r["Definition (from WordNet, Google, or Wikipedia)"].strip()}
                for r in rows if counts[r["Word"]] == 1]
    n_homonyms = sum(1 for w, n in counts.items() if n > 1)
    print(f"{len(rows)} THINGS concepts, {n_homonyms} homonym words dropped -> {len(concepts)}")

    OUT.mkdir(exist_ok=True)
    for size in [*args.sizes, len(concepts)]:
        if size > len(concepts):
            raise SystemExit(f"size {size} > {len(concepts)} concepts")
        picked = sorted(random.Random(args.seed).sample(concepts, size), key=lambda c: c["word"])
        name = f"things_n{size}_s{args.seed}.json" if size < len(concepts) else "things_all.json"
        (OUT / name).write_text(json.dumps({
            "source": f"THINGS concepts-metadata_things.tsv ({URL}), homonyms dropped",
            "sha256": SHA256, "n": size, "seed": args.seed if size < len(concepts) else None,
            "concepts": picked,
        }, indent=1) + "\n")
        print(f"wrote concepts/{name}")


if __name__ == "__main__":
    main()
