"""EXPLORATORY (not preregistered, 2026-10-13): the Apollo-style deception
probe (probes/qwen3.5-27b_layer16_logistic_regression.npz, instructed-pairs
logistic regression after Goldowsky-Dill et al. 2025) on the announcements.

Read (researcher, 2026-10-13): no follow-up turn; the dialogue the speaker
saw plus the announcement as a plain reply (collect_acts.dialogue, no private
reasoning in context); mean over the reply's tokens except the last
``apollo_drop_last``. The reply's offset is computed with the tokenizer: the
token count of the chat template rendered up to the generation prompt. The
sequence ends at <|im_end|>, so the dropped tokens are <|im_end|> and the
last ``apollo_drop_last - 1`` tokens of the announcement.

Layers: ``apollo_read_layers`` — 19 (the researcher's choice, the lowest
layer the fibs probes use; the probe was fitted at 16, so its standardization
and weights are applied off-layer) and 16 (the layer it was fitted at).
Random control: the probe with its weight vector replaced by a seeded random
direction of the same norm, per layer.

  acts_apollo/{arm}.npz   (N, d) pooled activations per layer (untracked)
  apollo_{arm}.csv        game_id, round, position, speaker, role, apollo_l{L}, rand_apollo_l{L}_s{seed}
  results_apollo.json     AUROC per layer, arm and pooled; random-control spread

Run:
  CUDA_VISIBLE_DEVICES=6,7 uv run --no-sync python experiments/2026-10-12-fibs-probes/apollo_probe.py --stage acts --arm 3p
  uv run --no-sync python experiments/2026-10-12-fibs-probes/apollo_probe.py --stage score
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from common import OUT, REPO, load_config, model_path, read_jsonl

MAIN = ("3p", "5p")
IM_END = "<|im_end|>"
KEY = ("game_id", "round", "position", "speaker", "role")


def encode(tok, r: dict, drop_last: int) -> tuple[list[int], tuple[int, int]]:
    from collect_acts import dialogue
    conv = dialogue(r)
    prefix = tok.apply_chat_template(conv[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(conv, tokenize=False)
    if full != prefix + r["announcement"] + IM_END + "\n":
        raise ValueError(f"template does not render the announcement as a plain reply: {full[len(prefix):][:80]!r}")
    ids = tok(full[:-1], add_special_tokens=False)["input_ids"]
    offset = len(tok(prefix, add_special_tokens=False)["input_ids"])
    if tok.decode(ids[offset:]) != r["announcement"] + IM_END:
        raise ValueError("the tokens after the offset are not the announcement")
    span = (offset, len(ids) - drop_last)
    if span[1] <= span[0]:
        raise ValueError(f"reply of {len(ids) - offset} tokens leaves nothing after dropping {drop_last}")
    return ids, span


def stage_acts(cfg, arm: str):
    from linear_probes.activations import collect_activations, load_truncated_decoder
    recs = read_jsonl(OUT / f"transcripts_{arm}.jsonl")
    layers = cfg.apollo_read_layers
    model, tok = load_truncated_decoder(model_path(cfg), max(layers), dtype=cfg.dtype, device=cfg.device)
    enc = [encode(tok, r, cfg.apollo_drop_last) for r in recs]
    if max(len(i) for i, _ in enc) > cfg.apollo_max_len:
        raise ValueError("an input exceeds apollo_max_len")
    print(f"{arm}: {len(enc)} inputs, longest {max(len(i) for i, _ in enc)} tokens, "
          f"pooled span {min(b - a for _, (a, b) in enc)}-{max(b - a for _, (a, b) in enc)} tokens", flush=True)
    acts = collect_activations(model, tok, enc, layers, batch_size=cfg.apollo_batch_size)
    (OUT / "acts_apollo").mkdir(exist_ok=True)
    np.savez(OUT / "acts_apollo" / f"{arm}.npz", keys=np.array([[str(r[k]) for k in KEY] for r in recs]),
             **{f"l{L}": acts[L].astype(np.float32) for L in layers})
    print(f"wrote acts_apollo/{arm}.npz")


def stage_score(cfg):
    from analyze import auroc, boot_auroc
    p = np.load(REPO / cfg.apollo_probe, allow_pickle=True)
    if str(p["kind"]) != "logistic_regression":
        raise ValueError(f"probe kind {p['kind']}")
    mu, sd, w, b = (p[k].astype(np.float64) for k in ("mu", "sd", "w", "b"))
    cols, keys = {}, {}
    for arm in MAIN:
        d = np.load(OUT / "acts_apollo" / f"{arm}.npz")
        keys[arm] = d["keys"]
        cols[arm] = {}
        for L in cfg.apollo_read_layers:
            z = (d[f"l{L}"].astype(np.float64) - mu) / sd
            cols[arm][f"apollo_l{L}"] = z @ w + b
            for seed in cfg.random_seeds:
                v = np.random.default_rng([seed, L]).standard_normal(len(w))
                cols[arm][f"rand_apollo_l{L}_s{seed}"] = z @ (v * np.linalg.norm(w) / np.linalg.norm(v)) + b
        with open(OUT / f"apollo_{arm}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(list(KEY) + list(cols[arm]))
            for i, k in enumerate(keys[arm]):
                wr.writerow(list(k) + [f"{cols[arm][c][i]:.5f}" for c in cols[arm]])

    def arrays(col, arms):
        return (np.concatenate([cols[a][col] for a in arms]),
                np.concatenate([keys[a][:, 4] == "Parsnip" for a in arms]).astype(int),
                np.concatenate([np.char.add(a + "/", keys[a][:, 0]) for a in arms]))

    sets = (("pooled", MAIN), ("3p", ("3p",)), ("5p", ("5p",)))
    res = {"probe": cfg.apollo_probe, "fitted_layer": int(p["layer"]), "drop_last": cfg.apollo_drop_last,
           "auroc": {}, "random": {}}
    for L in cfg.apollo_read_layers:
        c = f"apollo_l{L}"
        res["auroc"][c] = {lab: boot_auroc(*arrays(c, arms), cfg.n_boot, cfg.boot_seed) for lab, arms in sets}
        res["random"][c] = {}
        for lab, arms in sets:
            a = [auroc(*arrays(f"rand_{c}_s{s}", arms)[:2]) for s in cfg.random_seeds]
            res["random"][c][lab] = {"per_seed": a, "min": min(a), "median": float(np.median(a)), "max": max(a)}
        e, r = res["auroc"][c], res["random"][c]["pooled"]
        print(f"{c}: pooled {e['pooled']['auroc']:.3f} {[round(x, 3) for x in e['pooled']['ci95_game']]}  "
              f"3p {e['3p']['auroc']:.3f}  5p {e['5p']['auroc']:.3f}   random min/med/max "
              f"{r['min']:.3f}/{r['median']:.3f}/{r['max']:.3f}")
    (OUT / "results_apollo.json").write_text(json.dumps(res, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["acts", "score"], required=True)
    ap.add_argument("--arm", choices=MAIN)
    args = ap.parse_args()
    cfg = load_config()
    stage_score(cfg) if args.stage == "score" else stage_acts(cfg, args.arm)


if __name__ == "__main__":
    main()
