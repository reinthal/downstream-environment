"""EXPLORATORY (not preregistered): plain text written between the thinking
block and the tool call ("private text") and whether it drives the primary
result. In single_call turns the model sometimes writes visible text before
calling MakeAnnouncementTool; the other players never see it, but
collect_acts.py::step passes it as the assistant turn's content, so it is in
the context every fibs probe reads.

  private_text.json          counts, the primary AUROC with / without those rows, examples
  figures/private_text.svg   left: linear contrast by role, with / without private text;
                             right: the primary AUROC on each subset (95% game-clustered)

Run:  uv run --no-sync python experiments/2026-10-12-fibs-probes/private_text.py
"""
from __future__ import annotations

import csv
import json
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze import boot_auroc  # noqa: E402
from common import OUT, load_config, read_jsonl  # noqa: E402

MAIN = ("3p", "5p")
# the speaker stating its own role in the private text
ROLE_RE = re.compile(r"\bI(?:'m| am)(?: the| a)? (Parsnip|Carrot)\b|\bas (?:the|a) (Parsnip|Carrot)\b"
                     r"|\bmy role(?: is|:)? (?:the |a )?\**(Parsnip|Carrot)", re.I)
N_EXAMPLES = 3   # per role


def main():
    cfg = load_config()
    col = f"{cfg.primary_arch}_contrast"
    rows = []
    for arm in MAIN:
        score = {(r["game_id"], int(r["round"]), int(r["position"])): float(r[col])
                 for r in csv.DictReader(open(OUT / f"scores_{arm}.csv"))}
        for r in read_jsonl(OUT / f"transcripts_{arm}.jsonl"):
            text = r["acting_text"].strip()
            m = ROLE_RE.search(text)
            rows.append({"arm": arm, "game_id": r["game_id"], "round": r["round"], "position": r["position"],
                         "speaker": r["speaker"], "role": r["role"], "text": text,
                         "states_role": next(g for g in m.groups() if g).capitalize() if m else None,
                         "announcement": r["announcement"],
                         "score": score[(r["game_id"], r["round"], r["position"])]})
    s = np.array([r["score"] for r in rows])
    y = np.array([r["role"] == "Parsnip" for r in rows], dtype=int)
    g = np.array([f"{r['arm']}/{r['game_id']}" for r in rows])
    has = np.array([bool(r["text"]) for r in rows])

    res = {"score": col, "n": len(rows), "n_private_text": int(has.sum()),
           "by_role": {role: {"n": int((y == k).sum()), "with_private_text": int((has & (y == k)).sum())}
                       for role, k in (("Parsnip", 1), ("Carrot", 0))},
           "by_arm": {arm: {"n": sum(r["arm"] == arm for r in rows),
                            "with_private_text": sum(r["arm"] == arm and bool(r["text"]) for r in rows)} for arm in MAIN},
           "states_own_role": sum(r["states_role"] == r["role"] for r in rows if r["text"]),
           "states_other_role": sum(r["states_role"] not in (None, r["role"]) for r in rows if r["text"]),
           "no_role_statement": sum(r["states_role"] is None for r in rows if r["text"]),
           "median_chars": float(np.median([len(r["text"]) for r in rows if r["text"]])),
           "auroc": {name: boot_auroc(s[m], y[m], g[m], cfg.n_boot, cfg.boot_seed)
                     for name, m in (("all", np.ones_like(has)), ("without_private_text", ~has),
                                     ("with_private_text", has))}}
    ex = []
    for role in ("Parsnip", "Carrot"):
        ex += [r for r in rows if r["text"] and r["role"] == role and r["states_role"] == role][:N_EXAMPLES]
    res["examples"] = [{k: r[k] for k in ("arm", "game_id", "round", "position", "speaker", "role", "score")}
                       | {"private_text": r["text"], "announcement": r["announcement"]} for r in ex]
    (OUT / "private_text.json").write_text(json.dumps(res, indent=1) + "\n")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 2]})
    groups = [("Carrot\nno text", (y == 0) & ~has, "#1f77b4"), ("Carrot\nprivate text", (y == 0) & has, "#1f77b4"),
              ("Parsnip\nno text", (y == 1) & ~has, "#d62728"), ("Parsnip\nprivate text", (y == 1) & has, "#d62728")]
    parts = a0.violinplot([s[m] for _, m, _ in groups], showmedians=True, widths=0.85)
    for body, (_, m, c), hatch in zip(parts["bodies"], groups, ("", "//", "", "//")):
        body.set_facecolor(c); body.set_alpha(0.6); body.set_hatch(hatch)
    a0.set_xticks(range(1, 5)); a0.set_xticklabels([f"{lab}\nn={int(m.sum())}" for lab, m, _ in groups], fontsize=8)
    a0.set_ylabel(f"{cfg.primary_arch} contrast (Yes. − No.)")
    a0.set_title("3p + 5p announcements, by role and private text")
    names = [("all", "all"), ("without_private_text", "without\nprivate text"), ("with_private_text", "with\nprivate text")]
    for i, (k, _) in enumerate(names):
        e = res["auroc"][k]
        a1.errorbar([i], [e["auroc"]], yerr=[[e["auroc"] - e["ci95_game"][0]], [e["ci95_game"][1] - e["auroc"]]],
                    fmt="o", capsize=4, color="k")
        a1.annotate(f"{e['auroc']:.3f}\nn={e['n']}", (i, e["auroc"]), textcoords="offset points", xytext=(14, -22), fontsize=8)
    a1.axhline(cfg.auroc_threshold, ls="--", color="k", lw=0.8)
    a1.axhline(0.5, ls=":", color="grey")
    a1.set_xticks(range(3)); a1.set_xticklabels([lab for _, lab in names], fontsize=8)
    a1.set_xlim(-0.5, 2.7); a1.set_ylim(0.4, 1.01)     # figures/auroc.svg range, plus room for the point at 0.999
    a1.set_ylabel("announcement AUROC, Parsnip vs Carrot"); a1.set_title("primary AUROC by subset")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "private_text.svg"); plt.close(fig)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
