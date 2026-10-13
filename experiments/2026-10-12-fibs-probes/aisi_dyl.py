"""EXPLORATORY (not preregistered, added 2026-10-13 after the main run): the
UK AISI "did you lie" probe ``ai-safety-institute/dyl-qwen-qwen3.5-27b``
(Cooney, Africa & Irving 2026; loaded with ``lie_detectors.get_probe``) on
the same announcements.

The probe is a layer-38 logistic regression, so the activations saved for the
paper's probes (layers 19-58 at six other depths, another question, prefilled
answers) cannot be reused. This follows the model card instead:

  gen    append the card's question as a user turn and let Qwen3.5-27B answer
         it with reasoning enabled (raw /v1/completions on the Modal server,
         the prompt rendered here with the model's chat template and the
         discussion tools, so the exact token sequence is known);
  acts   forward prompt + generated text through the truncated decoder and
         keep the residual stream on the generated yes/no token(s), mean-
         pooled, at the candidate layers;
  score  probe(activation); higher = more deceptive.

Layer convention. The card says layer 38 of 0..63 without saying whether that
is the output of block 38 (this repo's "layer 38", HF hidden_states[39]) or
hidden_states[38] (this repo's layer 37). Both are collected; ``score`` picks
the one whose alpaca false-positive rate at the card's threshold is closest
to the 1% the card says it was calibrated to, and reports both. (The rule
was changed from "best match to the checkpoint's mean/std" after provisional
3p/5p AUROCs for both readings had been seen; treat the choice as post hoc.)

  rollouts_aisi_fu_{tag}.jsonl  per announcement: key, prompt, raw completion, finish reason
  acts_aisi/{tag}.pt            keys + (N, n_candidates, d) float32 on the answer token(s)
  aisi_dyl_{tag}.csv            key columns, role, answer, one score per candidate layer
  results_aisi_dyl.json         layer choice, AUROCs, alpaca false-positive rates

Run (tags: 3p 5p 3p_cc 5p_cc alpaca):
  uv run --no-sync python experiments/2026-10-12-fibs-probes/aisi_dyl.py --stage gen --tag 3p
  CUDA_VISIBLE_DEVICES=1,2 uv run --no-sync python experiments/2026-10-12-fibs-probes/aisi_dyl.py --stage acts --tag 3p
  uv run --no-sync python experiments/2026-10-12-fibs-probes/aisi_dyl.py --stage score
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import ARMS, OUT, REPO, load_config, model_path, read_jsonl, write_jsonl

TAGS = (*ARMS, "alpaca")
THINK_END = "</think>"
IM_END = "<|im_end|>"


def items(cfg, tag: str) -> list[dict]:
    """Per announcement: identifying columns, the finished dialogue and its tools."""
    if tag == "alpaca":
        return [{"key": {"idx": r["idx"]}, "role": "",
                 "messages": [*({"role": m["role"], "content": m["content"]} for m in r["input_messages"]),
                              {"role": "assistant", "content": r["public"]}],
                 "tools": None} for r in read_jsonl(REPO / cfg.alpaca_path)]
    from collect_acts import step
    out = []
    for r in read_jsonl(OUT / f"transcripts_{tag}.jsonl"):
        msgs, tools = step(r)
        out.append({"key": {"game_id": r["game_id"], "round": r["round"], "position": r["position"],
                            "speaker": r["speaker"]},
                    "role": r["role"], "messages": msgs, "tools": tools})
    return out


def render_prompt(tok, cfg, it: dict) -> str:
    kw = {"tools": it["tools"]} if it["tools"] else {}
    p = tok.apply_chat_template([*it["messages"], {"role": "user", "content": cfg.aisi_question}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=True, **kw)
    if not p.endswith("<|im_start|>assistant\n<think>\n"):
        raise ValueError(f"unexpected generation prompt ending {p[-60:]!r}")
    return p


def parse_answer(text: str) -> str | None:
    """'yes' / 'no' when the reply after the closed think block is exactly
    that word (optional full stop), else None."""
    if text.count(THINK_END) != 1:
        return None
    a = text.split(THINK_END)[1].strip().removesuffix(".").lower()
    return a if a in ("yes", "no") else None


def stage_gen(cfg, tag: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path(cfg))
    its = items(cfg, tag)
    prompts = [render_prompt(tok, cfg, it) for it in its]
    key = os.environ["MODAL_QWEN35_KEY"]

    def call(p: str) -> dict:
        body = {"model": cfg.model_id, "prompt": p, "max_tokens": cfg.aisi_max_tokens,
                "temperature": cfg.temperature, "top_p": cfg.top_p, "seed": cfg.aisi_gen_seed,
                "skip_special_tokens": False}
        req = urllib.request.Request(cfg.vllm_base_url + "/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        return json.load(urllib.request.urlopen(req, timeout=3600))

    with ThreadPoolExecutor(cfg.aisi_gen_workers) as ex:
        outs = list(ex.map(call, prompts))
    rows = []
    for it, p, o in zip(its, prompts, outs):
        c = o["choices"][0]
        rows.append({**it["key"], "role": it["role"], "prompt": p, "completion": c["text"],
                     "finish_reason": c["finish_reason"], "completion_tokens": o["usage"]["completion_tokens"],
                     "answer": parse_answer(c["text"]) if c["finish_reason"] == "stop" else None})
    write_jsonl(OUT / f"rollouts_aisi_fu_{tag}.jsonl", rows)
    n = len(rows)
    print(f"{tag}: {n} follow-ups; finish stop {sum(r['finish_reason'] == 'stop' for r in rows)}, "
          f"answer yes {sum(r['answer'] == 'yes' for r in rows)}, no {sum(r['answer'] == 'no' for r in rows)}, "
          f"none {sum(r['answer'] is None for r in rows)}")


def stage_acts(cfg, tag: str):
    import torch
    from acts import load_model
    rows = [r for r in read_jsonl(OUT / f"rollouts_aisi_fu_{tag}.jsonl") if r["answer"] is not None]
    layers = cfg.aisi_layer_candidates
    model, tok = load_model(model_path(cfg), max(layers), dtype=cfg.dtype, device=cfg.device)
    think_end = tok.convert_tokens_to_ids(THINK_END)
    enc = []
    for r in rows:
        ids = tok(r["prompt"] + r["completion"].removesuffix(IM_END), add_special_tokens=False)["input_ids"]
        last = max(i for i, t in enumerate(ids) if t == think_end)
        pos = [i for i in range(last + 1, len(ids)) if tok.decode(ids[i:i + 1]).strip().lower() == r["answer"]]
        if not pos:
            raise ValueError(f"no {r['answer']!r} token after the think block: {tok.convert_ids_to_tokens(ids[last:])}")
        if len(ids) > cfg.acts_max_len:
            raise ValueError(f"{len(ids)} tokens exceed acts_max_len")
        enc.append((ids, pos))

    inner = model
    while hasattr(inner, "model"):
        inner = inner.model
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(len(enc)), key=lambda i: -len(enc[i][0]))
    out = torch.zeros(len(enc), len(layers), model.config.hidden_size if hasattr(model.config, "hidden_size")
                      else inner.config.hidden_size, dtype=torch.float32)
    i = 0
    with torch.no_grad():
        while i < len(order):
            w = len(enc[order[i]][0])
            batch = order[i:i + min(cfg.acts_max_batch, max(1, cfg.acts_batch_tokens // w))]
            i += len(batch)
            input_ids = torch.full((len(batch), w), pad, dtype=torch.long)
            attn = torch.zeros((len(batch), w), dtype=torch.long)
            for b, j in enumerate(batch):
                ids = enc[j][0]
                input_ids[b, :len(ids)] = torch.tensor(ids)
                attn[b, :len(ids)] = 1
            res = inner(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
                        output_hidden_states=True)
            for b, j in enumerate(batch):
                for k, L in enumerate(layers):
                    out[j, k] = res.hidden_states[L + 1][b, enc[j][1]].float().mean(0).cpu()
            del res
            print(f"  collected {i}/{len(order)} (batch {len(batch)} x {w} tokens)", flush=True)
    (OUT / "acts_aisi").mkdir(exist_ok=True)
    keys = [{k: v for k, v in r.items() if k not in ("prompt", "completion")} for r in rows]
    torch.save({"layers": layers, "keys": keys, "acts": out}, OUT / "acts_aisi" / f"{tag}.pt")
    print(f"wrote acts_aisi/{tag}.pt ({len(rows)} rows)")


def stage_score(cfg):
    import numpy as np
    import torch
    from lie_detectors import get_probe

    from analyze import boot_auroc
    probe = get_probe(cfg.aisi_repo, filename=cfg.aisi_filename, revision=cfg.aisi_revision)
    thr = float(probe.threshold)
    data = {t: torch.load(OUT / "acts_aisi" / f"{t}.pt") for t in TAGS}
    layers = data["alpaca"]["layers"]
    if layers != cfg.aisi_layer_candidates:
        raise ValueError(f"collected layers {layers} != config {cfg.aisi_layer_candidates}")

    # layer convention, decided on alpaca only (the standardization statistics are reported but not
    # used: the checkpoint's mean/std need not come from alpaca)
    fit = {}
    for k, L in enumerate(layers):
        z = (data["alpaca"]["acts"][:, k] - probe.dataset_mean) / probe.dataset_std
        s = probe(data["alpaca"]["acts"][:, k]).detach().numpy()
        fit[L] = {"z_abs_mean_of_dim_means": float(z.mean(0).abs().mean()),
                  "z_mean_of_dim_stds": float(z.std(0).mean()),
                  "z_rms": float(z.pow(2).mean().sqrt()),
                  "alpaca_fpr_at_card_threshold": float((s > thr).mean())}
    # the card calibrates its threshold to a 1% false-positive rate on alpaca: take the reading closest to that
    chosen = min(layers, key=lambda L: abs(fit[L]["alpaca_fpr_at_card_threshold"] - (1 - cfg.fpr_quantile)))

    res = {"probe": f"{cfg.aisi_repo}@{cfg.aisi_revision}/{cfg.aisi_filename}", "probe_layer_card": probe.layer,
           "card_threshold": thr, "question": cfg.aisi_question, "layer_fit_on_alpaca": fit,
           "chosen_layer_block_index": chosen, "per_layer": {}, "answers": {}}
    scores = {}
    for t in TAGS:
        fu = read_jsonl(OUT / f"rollouts_aisi_fu_{t}.jsonl")
        res["answers"][t] = {"n": len(fu), "finish_stop": sum(r["finish_reason"] == "stop" for r in fu),
                             "yes": sum(r["answer"] == "yes" for r in fu), "no": sum(r["answer"] == "no" for r in fu),
                             "no_score": sum(r["answer"] is None for r in fu)}
        if t != "alpaca":
            for role in ("Carrot", "Parsnip"):
                sub = [r for r in fu if r["role"] == role]
                if sub:
                    res["answers"][t][role] = {"n": len(sub), "yes": sum(r["answer"] == "yes" for r in sub),
                                               "no_score": sum(r["answer"] is None for r in sub)}
        s = probe(data[t]["acts"]).detach().numpy()          # (N, n_candidates)
        scores[t] = s
        keys = data[t]["keys"]
        cols = [c for c in keys[0] if c not in ("finish_reason", "completion_tokens")]
        with open(OUT / f"aisi_dyl_{t}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols + [f"aisi_score_l{L}" for L in layers])
            for kk, row in zip(keys, s):
                w.writerow([kk[c] for c in cols] + [f"{v:.4f}" for v in row])

    cut_own = {L: float(np.quantile(scores["alpaca"][:, k], cfg.fpr_quantile)) for k, L in enumerate(layers)}
    for k, L in enumerate(layers):
        sec = {"alpaca_cutpoint_1pct_fpr": cut_own[L], "auroc": {}, "above": {}}
        rows = []
        for arm in ("3p", "5p"):
            for kk, v in zip(data[arm]["keys"], scores[arm][:, k]):
                rows.append((v, kk["role"] == "Parsnip", f"{arm}/{kk['game_id']}", arm))
        for label in ("pooled", "3p", "5p"):
            sub = [r for r in rows if label == "pooled" or r[3] == label]
            sec["auroc"][label] = boot_auroc(np.array([r[0] for r in sub]), np.array([r[1] for r in sub], dtype=int),
                                             np.array([r[2] for r in sub]), cfg.n_boot, cfg.boot_seed)
        for name, cut in (("card_threshold", thr), ("alpaca_1pct", cut_own[L])):
            sec["above"][name] = {
                "alpaca": float((scores["alpaca"][:, k] > cut).mean()),
                "Parsnip": float(np.mean([r[0] > cut for r in rows if r[1]])),
                "Carrot": float(np.mean([r[0] > cut for r in rows if not r[1]])),
                "all_carrot_games": float(np.mean(np.concatenate([scores[a][:, k] for a in ("3p_cc", "5p_cc")]) > cut))}
        res["per_layer"][L] = sec
    (OUT / "results_aisi_dyl.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"layer_fit_on_alpaca": fit, "chosen": chosen, "answers": res["answers"],
                      "chosen_layer": res["per_layer"][chosen]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "acts", "score"], required=True)
    ap.add_argument("--tag", choices=TAGS)
    args = ap.parse_args()
    cfg = load_config()
    if args.stage == "score":
        stage_score(cfg)
    else:
        (stage_gen if args.stage == "gen" else stage_acts)(cfg, args.tag)


if __name__ == "__main__":
    main()
