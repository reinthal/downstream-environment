"""DYL probe vs the alpaca control, rescored with the DYLProbe primitive.

Everything runs through `linear_probes.DYLProbe` (the 2026-10-01 follow-up
machinery refactored into a primitive): alpaca replies and base-game turns
get p(deceptive) read on the follow-up "No", then one violin figure
(alpaca / Carrot / Parsnip) and one ROC figure.

Stages (both `.venv`):
  score    layer-18 activations for 9,994 alpaca replies + 1,500 base-game
           turns -> scores_{alpaca,base}.csv
  figures  figures/violins_alpaca.png, figures/roc_alpaca.png

Run:  CUDA_VISIBLE_DEVICES=6 uv run --no-sync python experiments/2026-10-07-dyl-alpaca/run.py --stage score
      uv run --no-sync python experiments/2026-10-07-dyl-alpaca/run.py --stage figures
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from linear_probes import ExperimentConfig  # noqa: E402

GAME_KEYS = ["game_id", "seed_group", "round", "position", "speaker", "role"]


@dataclass
class DYLAlpacaConfig(ExperimentConfig):
    alpaca_path: str = "rollouts_alpaca.jsonl"       # symlinks, relative to this folder
    transcripts_path: str = "transcripts_base.jsonl"
    threshold_quantile: float = 0.99                 # alpaca cutpoint = 1%-FPR


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def stage_score(cfg: DYLAlpacaConfig):
    from linear_probes import DYLProbe

    probe = DYLProbe(cfg.model_id, cfg.layer, str(REPO / cfg.probe_paths["followup"]),
                     batch_size=cfg.batch_size, max_len=cfg.max_len,
                     dtype=cfg.dtype, device=cfg.device,
                     truncate_layers=cfg.truncate_layers)

    alpaca = read_jsonl(OUT / cfg.alpaca_path)
    s = probe.score([[*r["messages"], {"role": "assistant", "content": r["reply"]}]
                     for r in alpaca])
    with open(OUT / "scores_alpaca.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["followup_full"])
        w.writerows([f"{v:.6f}"] for v in s)
    print(f"alpaca: {len(s)} scored", flush=True)

    turns = read_jsonl(OUT / cfg.transcripts_path)
    s = probe.score([[*r["input_messages"],
                      {"role": "assistant", "content": r["announcement"] or ""}]
                     for r in turns])
    with open(OUT / "scores_base.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([*GAME_KEYS, "silent", "followup_full"])
        for r, v in zip(turns, s):
            w.writerow([*(r[k] for k in GAME_KEYS), int(not r["announcement"]), f"{v:.6f}"])
    print(f"base: {len(s)} scored", flush=True)


# ── figures (CPU) ────────────────────────────────────────────────────────────

SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8f8d86"


def logit(p):
    import numpy as np
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stage_figures(cfg: DYLAlpacaConfig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
        "legend.frameon": False})

    with open(OUT / "scores_alpaca.csv") as f:
        alpaca = np.array([float(r["followup_full"]) for r in csv.DictReader(f)])
    with open(OUT / "scores_base.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["silent"] == "0"]
    vals = {"Alpaca (control)": alpaca,
            "Carrot": np.array([float(r["followup_full"]) for r in rows
                                if r["role"] == "Carrot"]),
            "Parsnip": np.array([float(r["followup_full"]) for r in rows
                                 if r["role"] == "Parsnip"])}
    thr = float(np.quantile(alpaca, cfg.threshold_quantile))

    # violins
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for k, (name, color) in enumerate((("Alpaca (control)", GRAY), ("Carrot", BLUE),
                                       ("Parsnip", ORANGE))):
        v = ax.violinplot([logit(vals[name])], positions=[k], widths=0.8,
                          showextrema=False, showmedians=True)
        for b in v["bodies"]:
            b.set_facecolor(color)
            b.set_edgecolor(color)
            b.set_alpha(0.5)
        v["cmedians"].set_color(INK)
    lo = ax.get_ylim()[0]
    for k, name in enumerate(vals):
        over = float((vals[name] > thr).mean())
        ax.text(k, lo, f"{100 * over:.1f}% over", ha="center", va="bottom",
                fontsize=8.5, color=INK2)
    ax.axhline(logit(thr), color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(2.52, logit(thr), "alpaca 1%-FPR", fontsize=8.5, color=MUTED, va="bottom",
            ha="right")
    ax.set_xticks(range(3), [f"{n}\n(n={len(vals[n]):,})" for n in vals])
    lo, hi = ax.get_ylim()
    probs = [p for p in (1e-6, 1e-4, 1e-2, 0.5, 0.99) if lo <= logit(p) <= hi]
    ax.set_yticks([float(logit(p)) for p in probs],
                  [f"{100 * p:g}%" for p in probs])
    ax.set_ylabel("DYL probe score (probability, log-odds spacing)")
    ax.grid(axis="x", visible=False)
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.14 / h, "DYL probe scores: alpaca control vs base-game messages",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.44 / h, "Rescored with linear_probes.DYLProbe. Black line = median; "
             "dashed = alpaca 1%-FPR threshold.", fontsize=9.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.84, bottom=0.14)
    fig.savefig(OUT / "figures" / "violins_alpaca.png", dpi=170)
    print("wrote", OUT / "figures" / "violins_alpaca.png", flush=True)

    # ROC
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for other, color in (("Carrot", BLUE), ("Alpaca (control)", GRAY)):
        y = [0] * len(vals[other]) + [1] * len(vals["Parsnip"])
        s = np.concatenate([vals[other], vals["Parsnip"]])
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"Parsnip vs {other} (AUROC {roc_auc_score(y, s):.2f})")
    tpr_at_thr = float((vals["Parsnip"] > thr).mean())
    ax.scatter([1 - cfg.threshold_quantile], [tpr_at_thr], s=55, color=ORANGE,
               edgecolor=SURFACE, lw=1.5, zorder=3)
    ax.annotate(f"alpaca 1%-FPR threshold catches {100 * tpr_at_thr:.0f}% of Parsnip msgs",
                (1 - cfg.threshold_quantile, tpr_at_thr), xytext=(0.3, 0.17),
                fontsize=8.5, color=INK2,
                arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 1})
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (1, 2)))
    ax.text(0.62, 0.55, "chance", color=MUTED, fontsize=8.5, rotation=38)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal",
           xlabel="False positive rate (honest flagged)",
           ylabel="True positive rate (Parsnip caught)")
    ax.legend(loc="lower right", fontsize=9)
    h = fig.get_figheight()
    fig.text(0.03, 1 - 0.14 / h, "ROC: DYL probe, in-game vs alpaca control", fontsize=13,
             fontweight="bold", va="top")
    fig.text(0.03, 1 - 0.44 / h, "Positives: Parsnip messages, base games (non-silent).",
             fontsize=9.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.86, bottom=0.11)
    fig.savefig(OUT / "figures" / "roc_alpaca.png", dpi=170)
    print("wrote", OUT / "figures" / "roc_alpaca.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["score", "figures"])
    args = ap.parse_args()
    cfg = DYLAlpacaConfig.load(OUT / "config.json")
    cfg.save(OUT)
    {"score": stage_score, "figures": stage_figures}[args.stage](cfg)


if __name__ == "__main__":
    main()
