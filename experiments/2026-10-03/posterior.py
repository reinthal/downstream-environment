"""Step 2: calibrated per-player probabilities from a detector's scores (CPU).

For each detector, cross-fitted by seed group (fit on 4 groups, apply to the
5th), so no game's probabilities come from a model that saw it:
  1. remove the round / speaking-position drift from the scores (OLS on
     round and position dummies; both are balanced across roles, so this uses
     no role information)
  2. average each player's corrected scores (step 1: the best aggregation)
  3. fit a Gaussian to Parsnip-player and to Carrot-player averages; each
     player's evidence = log-likelihood ratio Parsnip vs Carrot
  4. exactly one Parsnip per game: probabilities = softmax over the 5 players
     of T * evidence, with the temperature T fitted on the training games
Evaluated on the held-out games: top-1 / top-2 accuracy, log loss, Brier,
calibration (per-player probability bins). Random directions go through the
identical pipeline as the null. Transfer: a model fitted on all base games
applied unchanged to steered and no-thinking games.

Output: step2.json, posteriors_{detector}_{arm}.json ({game: {player: p}}),
        figures/calibration.png

Run:  .venv/bin/python experiments/2026-10-03/posterior.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from data import OUT, messages, scores

N_RANDOM = 100
TEMPS = np.exp(np.linspace(np.log(0.05), np.log(5), 60))


def design(msgs):
    return np.column_stack([np.ones(len(msgs)),
                            *[[m["round"] == r for m in msgs] for r in (1, 2)],
                            *[[m["position"] == p for m in msgs] for p in (1, 2, 3, 4)]]).astype(float)


def players(msgs, s):
    """[(game, speaker, mean score, is_parsnip)]"""
    acc = defaultdict(list)
    role = {}
    for m, v in zip(msgs, s):
        acc[(m["game_id"], m["speaker"])].append(v)
        role[(m["game_id"], m["speaker"])] = m["role"] == "Parsnip"
    return [(g, p, float(np.mean(v)), role[(g, p)]) for (g, p), v in acc.items()]


def log_normal(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd)


class Model:
    def fit(self, msgs, s):
        X = design(msgs)
        self.beta, *_ = np.linalg.lstsq(X, s, rcond=None)
        pl = players(msgs, s - X @ self.beta)
        par = np.array([m for *_, m, isp in [(None, None, x[2], x[3]) for x in pl] if isp])
        car = np.array([m for *_, m, isp in [(None, None, x[2], x[3]) for x in pl] if not isp])
        self.mu_p, self.sd_p = par.mean(), par.std() + 1e-6
        self.mu_c, self.sd_c = car.mean(), car.std() + 1e-6
        games = self._evidence(pl)
        ll = [sum(np.log(self._softmax(e, t)[ip]) for e, ip in games.values()) for t in TEMPS]
        self.T = float(TEMPS[int(np.argmax(ll))])
        return self

    def _evidence(self, pl):
        """game -> (evidence per player, index of the parsnip) + player names"""
        by = defaultdict(list)
        for g, p, m, isp in pl:
            by[g].append((p, log_normal(m, self.mu_p, self.sd_p) - log_normal(m, self.mu_c, self.sd_c), isp))
        self.names = {g: [p for p, _, _ in v] for g, v in by.items()}
        return {g: (np.array([e for _, e, _ in v]), [isp for *_, isp in v].index(True))
                for g, v in by.items()}

    @staticmethod
    def _softmax(e, t):
        z = t * e - (t * e).max()
        return np.exp(z) / np.exp(z).sum()

    def predict(self, msgs, s):
        pl = players(msgs, s - design(msgs) @ self.beta)
        ev = self._evidence(pl)
        return {g: dict(zip(self.names[g], self._softmax(e, self.T))) for g, (e, _) in ev.items()}


def crossfit(msgs, s):
    out = {}
    for k in sorted({m["seed_group"] for m in msgs}):
        tr = [i for i, m in enumerate(msgs) if m["seed_group"] != k]
        te = [i for i, m in enumerate(msgs) if m["seed_group"] == k]
        model = Model().fit([msgs[i] for i in tr], s[tr])
        out.update(model.predict([msgs[i] for i in te], s[te]))
    return out


def evaluate(post, msgs):
    parsnip = {m["game_id"]: m["speaker"] for m in msgs if m["role"] == "Parsnip"}
    top1, top2, ll, brier, pairs = [], [], [], [], []
    for g, probs in post.items():
        order = sorted(probs, key=probs.get, reverse=True)
        top1.append(order[0] == parsnip[g])
        top2.append(parsnip[g] in order[:2])
        ll.append(-np.log(max(probs[parsnip[g]], 1e-9)))
        brier.append(sum((q - (p == parsnip[g])) ** 2 for p, q in probs.items()))
        pairs += [(q, p == parsnip[g]) for p, q in probs.items()]
    bins = np.linspace(0, 1, 11)
    q = np.array([a for a, _ in pairs])
    y = np.array([b for _, b in pairs], dtype=float)
    calib = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (q >= lo) & (q < hi if hi < 1 else q <= hi)
        if sel.sum():
            calib.append({"bin": [round(lo, 1), round(hi, 1)], "n": int(sel.sum()),
                          "mean_predicted": round(float(q[sel].mean()), 3),
                          "observed": round(float(y[sel].mean()), 3)})
    ece = sum(c["n"] * abs(c["mean_predicted"] - c["observed"]) for c in calib) / len(q)
    return {"top1": round(float(np.mean(top1)), 3), "top2": round(float(np.mean(top2)), 3),
            "log_loss": round(float(np.mean(ll)), 3), "brier": round(float(np.mean(brier)), 3),
            "calibration_error": round(float(ece), 3), "n_games": len(top1),
            "calibration": calib, "chance": {"top1": 0.2, "log_loss": round(float(np.log(5)), 3),
                                              "brier": 0.8}}


def main():
    summary = {}
    base = messages("base")
    for det in ("fu", "lr", "godmode"):
        s = scores(det, "base")
        post = crossfit(base, s)
        res = {"base (cross-fitted)": evaluate(post, base)}
        (OUT / f"posteriors_{det}_base.json").write_text(json.dumps(post, indent=1) + "\n")
        full = Model().fit(base, s)
        res["fitted"] = {"temperature": round(full.T, 3),
                         "parsnip_mean": round(full.mu_p, 3), "carrot_mean": round(full.mu_c, 3)}
        for arm in ("steered", "nothink"):
            s_arm = scores(det, arm)
            if s_arm is None:
                continue
            post_arm = full.predict(messages(arm), s_arm)
            res[f"{arm} (base model, no refit)"] = evaluate(post_arm, messages(arm))
            (OUT / f"posteriors_{det}_{arm}.json").write_text(json.dumps(post_arm, indent=1) + "\n")
        if det in ("fu", "lr"):
            null = [evaluate(crossfit(base, scores(f"rand_{det}", "base", seed=k)), base)
                    for k in range(N_RANDOM)]
            t1 = np.array([n["top1"] for n in null])
            lls = np.array([n["log_loss"] for n in null])
            res["random directions, same pipeline (base)"] = {
                "top1_mean": round(float(t1.mean()), 3),
                "top1_middle95": [round(float(x), 3) for x in np.percentile(t1, [2.5, 97.5])],
                "share_top1_ge_detector": round(float(np.mean(t1 >= res["base (cross-fitted)"]["top1"])), 3),
                "log_loss_mean": round(float(lls.mean()), 3),
                "share_log_loss_le_detector": round(float(np.mean(lls <= res["base (cross-fitted)"]["log_loss"])), 3)}
            # one fixed random direction (seed 0) as the auditor's null condition
            (OUT / f"posteriors_rand_{det}_base.json").write_text(
                json.dumps(crossfit(base, scores(f"rand_{det}", "base", seed=0)), indent=1) + "\n")
        summary[det] = res
    (OUT / "step2.json").write_text(json.dumps(summary, indent=1) + "\n")

    for det, res in summary.items():
        print(f"\n== {det}")
        for k, v in res.items():
            if "top1" in v and "calibration" in v:
                print(f"  {k:32s} top1 {v['top1']:.2f}  top2 {v['top2']:.2f}  log loss {v['log_loss']:.2f} "
                      f"(chance 1.61)  calibration error {v['calibration_error']:.3f}")
            else:
                print(f"  {k:32s} {v}")
    plot(summary)


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                             "#e1e0d9", "#c3c2b7")
    colors = {"fu": "#eb6834", "lr": "#2a78d6", "godmode": "#1baf7a"}
    names = {"fu": "Follow-up probe", "lr": "LR probe", "godmode": "God-mode judge"}
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (1, 2)))
    for det in ("fu", "lr", "godmode"):
        c = summary[det]["base (cross-fitted)"]
        xs = [b["mean_predicted"] for b in c["calibration"]]
        ys = [b["observed"] for b in c["calibration"]]
        ax.plot(xs, ys, color=colors[det], lw=2, marker="o", ms=5,
                label=f"{names[det]} (top-1 {c['top1']:.2f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="Predicted probability of being the Parsnip",
           ylabel="Share who actually were the Parsnip")
    ax.legend(loc="upper left", fontsize=9)
    h = fig.get_figheight()
    fig.text(0.03, 1 - 0.14 / h, "Are the per-player probabilities calibrated?", fontsize=13,
             fontweight="bold", va="top")
    fig.text(0.03, 1 - 0.44 / h, "Base games, held-out (cross-fitted by seed group). "
             "On the dotted line = calibrated.", fontsize=9.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.11)
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "calibration.png", dpi=170)
    print("wrote", OUT / "figures" / "calibration.png")


if __name__ == "__main__":
    main()
