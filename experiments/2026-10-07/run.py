"""Probe serving via vLLM: build, score, and parity-check the published model.

The served model mean-pools the RAW block-16 residual stream over ALL prompt
tokens (span selection is the client's job), so the reference here is
collect_activations over the WHOLE rendered conversation — deliberately NOT
LMProbe.score's final-turn span.

Stages (each stage reads config.json; earlier stages must have run):

  publish      (.venv)       build the truncated checkpoint into publish_dir
  score_hf     (.venv)       reference: bare truncated decoder + probe npz
  score_vllm   (.venv-vllm)  llm.classify() on the exact same token ids
  compare      (.venv)       Pearson r on logits + max|dp| -> parity_report.json
  push         (.venv)       upload publish_dir to cfg.hf_repo_id
  serve_smoke  (.venv-vllm)  vllm serve + POST /classify vs offline scores

  dist_score_hf / dist_score_vllm / dist_compare — same comparison at
  distribution level: n_dist conversations per dataset in cfg.dist_sources
  (carrot-parsnip transcripts, alpaca rollouts) -> overlaid histograms +
  parity scatter in figures/vllm_vs_hf.png, stats in dist_report.json.

Run:
  uv run --no-sync python experiments/2026-10-07/run.py --stage score_hf
  CUDA_VISIBLE_DEVICES=7 PATH=$PWD/.venv-vllm/bin:$PATH \
      .venv-vllm/bin/python experiments/2026-10-07/run.py --stage score_vllm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
sys.path.insert(0, str(REPO))

from serving.config import ServingConfig  # noqa: E402


def to_messages(r: dict) -> list[dict]:
    """Row -> finished conversation. The final turn's public text is
    `announcement` in transcripts_*.jsonl and `public` in rollouts_*.jsonl."""
    keys = [k for k in ("announcement", "public") if k in r]
    if len(keys) != 1:
        raise ValueError(f"row has {keys or 'no'} final-text keys; expected exactly one")
    return list(r["input_messages"]) + [{
        "role": "assistant", "content": r[keys[0]],
        "reasoning_content": r["reasoning"],
    }]


def render_prompt(tok, msgs: list[dict], cfg: ServingConfig) -> dict | None:
    """The served model's client contract, mirrored for the reference path.
    Returns {"text", "token_ids", "ref_span"} or None if over max_len.

    MEAN (span probes): the whole rendered conversation, reference span = all.
    LAST (DYL): conversation + follow-up pair, text ending exactly at the
    "No"; reference span = that final token (DYLProbe's detect mask).
    """
    if cfg.pooling == "LAST":
        from linear_probes.dyl_probe import FOLLOWUP_ANSWER, with_followup

        conv = with_followup(msgs)
        prefix = tok.apply_chat_template(conv[:-1], tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
        text = prefix + FOLLOWUP_ANSWER
        ids = tok(text, add_special_tokens=False)["input_ids"]
        n_no = len(tok(FOLLOWUP_ANSWER, add_special_tokens=False)["input_ids"])
        if n_no != 1:
            raise ValueError(f'"{FOLLOWUP_ANSWER}" is {n_no} tokens; LAST pooling '
                             "reads exactly one")
        span = (len(ids) - 1, len(ids))
    else:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        span = (0, len(ids))
    if len(ids) > cfg.max_len:
        return None
    return {"text": text, "token_ids": ids, "ref_span": span}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_parity_prompts(cfg: ServingConfig) -> list[dict]:
    """First n_parity conversations that fit max_len untruncated, rendered per
    the serving contract (render_prompt)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    rows = []
    for r in read_jsonl(REPO / cfg.parity_source):
        p = render_prompt(tok, to_messages(r), cfg)
        if p is None:
            continue
        rows.append({"game_id": r["game_id"], "round": r["round"],
                     "speaker": r["speaker"], "role": r["role"], **p})
        if len(rows) == cfg.n_parity:
            return rows
    raise ValueError(f"only {len(rows)} of {cfg.n_parity} conversations fit "
                     f"max_len={cfg.max_len} in {cfg.parity_source}")


# ── stages ───────────────────────────────────────────────────────────────────

def stage_publish(cfg: ServingConfig):
    from serving.publish import build
    build(cfg)


def stage_score_hf(cfg: ServingConfig):
    import numpy as np

    from linear_probes.activations import collect_activations, load_truncated_decoder
    from linear_probes.probes import LinearProbe

    probe = LinearProbe.load(str(REPO / cfg.probe_paths[cfg.probe_name]))
    rows = load_parity_prompts(cfg)
    with open(OUT / "parity_tokens.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    model, tok = load_truncated_decoder(cfg.model_id, cfg.layer,
                                        dtype=cfg.dtype, device=cfg.device)
    encoded = [(r["token_ids"], tuple(r["ref_span"])) for r in rows]
    X = collect_activations(model, tok, encoded, [cfg.layer], cfg.batch_size)[cfg.layer]

    logits = (X - probe.mu) / probe.sd @ probe.w + probe.b
    probs = probe.predict_proba(X)
    (OUT / "scores_hf.json").write_text(json.dumps(
        {"logits": logits.tolist(), "probs": probs.tolist()}, indent=1) + "\n")
    print(f"scored {len(rows)} conversations; "
          f"p(deceptive) mean {float(np.mean(probs)):.3f} "
          f"range [{float(np.min(probs)):.3f}, {float(np.max(probs)):.3f}]")


def stage_score_vllm(cfg: ServingConfig):
    import os

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, PoolingParams, TokensPrompt

    rows = read_jsonl(OUT / "parity_tokens.jsonl")
    llm = LLM(model=str(REPO / cfg.publish_dir),
              runner="pooling",
              tensor_parallel_size=cfg.tensor_parallel_size or 1,
              max_model_len=cfg.max_model_len,
              dtype="bfloat16",
              enforce_eager=True)
    prompts = [TokensPrompt(prompt_token_ids=r["token_ids"]) for r in rows]
    probs = [o.outputs.probs[0] for o in llm.classify(prompts)]
    logits = [o.outputs.probs[0] for o in llm.classify(
        prompts, pooling_params=PoolingParams(use_activation=False))]
    (OUT / "scores_vllm.json").write_text(json.dumps(
        {"logits": logits, "probs": probs}, indent=1) + "\n")

    # string path must agree with the token-id path (catches tokenizer drift
    # or special-token insertion on the server side)
    text_probs = [o.outputs.probs[0] for o in llm.classify([r["text"] for r in rows[:5]])]
    drift = max(abs(a - b) for a, b in zip(text_probs, probs[:5]))
    if drift > 1e-6:
        raise AssertionError(f"string-prompt scores drift from token-id scores "
                             f"by {drift:.3e}; tokenization mismatch")
    print(f"scored {len(rows)} prompts; string-path drift {drift:.1e}")


def stage_compare(cfg: ServingConfig):
    import numpy as np

    hf = json.loads((OUT / "scores_hf.json").read_text())
    vl = json.loads((OUT / "scores_vllm.json").read_text())
    ref_l, ref_p = np.array(hf["logits"]), np.array(hf["probs"])
    got_l, got_p = np.array(vl["logits"]), np.array(vl["probs"])

    r = float(np.corrcoef(ref_l, got_l)[0, 1])
    max_dp = float(np.abs(ref_p - got_p).max())
    mean_dp = float(np.abs(ref_p - got_p).mean())
    report = {"n": len(ref_l), "pearson_r_logits": r,
              "max_abs_dp": max_dp, "mean_abs_dp": mean_dp,
              "thresholds": {"min_r": cfg.parity_min_r,
                             "max_abs_dp": cfg.parity_max_abs_dp}}
    (OUT / "parity_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    ok = r > cfg.parity_min_r and max_dp < cfg.parity_max_abs_dp
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def stage_push(cfg: ServingConfig):
    from serving.publish import push
    push(cfg)


# ── distribution comparison on whole datasets (carrot-parsnip, alpaca) ──────

def stage_dist_score_hf(cfg: ServingConfig):
    from transformers import AutoTokenizer

    from linear_probes.activations import collect_activations, load_truncated_decoder
    from linear_probes.probes import LinearProbe

    probe = LinearProbe.load(str(REPO / cfg.probe_paths[cfg.probe_name]))
    tok = AutoTokenizer.from_pretrained(cfg.model_id)

    prompts: dict[str, list[dict]] = {}
    for name, src in cfg.dist_sources.items():
        rows = []
        for r in read_jsonl(REPO / src):
            p = render_prompt(tok, to_messages(r), cfg)
            if p is None:
                continue
            rows.append(p)
            if len(rows) == cfg.n_dist:
                break
        if len(rows) < cfg.n_dist:
            raise ValueError(f"{name}: only {len(rows)} of {cfg.n_dist} conversations "
                             f"fit max_len={cfg.max_len}")
        with open(OUT / f"dist_tokens_{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        prompts[name] = rows

    model, _ = load_truncated_decoder(cfg.model_id, cfg.layer,
                                      dtype=cfg.dtype, device=cfg.device)
    for name, rows in prompts.items():
        encoded = [(r["token_ids"], tuple(r["ref_span"])) for r in rows]
        X = collect_activations(model, tok, encoded, [cfg.layer], cfg.batch_size)[cfg.layer]
        logits = (X - probe.mu) / probe.sd @ probe.w + probe.b
        (OUT / f"dist_scores_hf_{name}.json").write_text(json.dumps(
            {"logits": logits.tolist(),
             "probs": probe.predict_proba(X).tolist()}, indent=1) + "\n")
        print(f"{name}: scored {len(rows)}")


def stage_dist_score_vllm(cfg: ServingConfig):
    import os

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, PoolingParams, TokensPrompt

    llm = LLM(model=str(REPO / cfg.publish_dir),
              runner="pooling",
              tensor_parallel_size=cfg.tensor_parallel_size or 1,
              max_model_len=cfg.max_model_len,
              dtype="bfloat16",
              enforce_eager=True)
    for name in cfg.dist_sources:
        rows = read_jsonl(OUT / f"dist_tokens_{name}.jsonl")
        prompts = [TokensPrompt(prompt_token_ids=r["token_ids"]) for r in rows]
        probs = [o.outputs.probs[0] for o in llm.classify(prompts)]
        logits = [o.outputs.probs[0] for o in llm.classify(
            prompts, pooling_params=PoolingParams(use_activation=False))]
        (OUT / f"dist_scores_vllm_{name}.json").write_text(json.dumps(
            {"logits": logits, "probs": probs}, indent=1) + "\n")
        print(f"{name}: scored {len(rows)}")


def stage_dist_compare(cfg: ServingConfig):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    BLUE, AQUA, INK, MUTED = "#2a78d6", "#1baf7a", "#3a3a37", "#73726c"
    names = list(cfg.dist_sources)
    scores = {}
    for name in names:
        hf = json.loads((OUT / f"dist_scores_hf_{name}.json").read_text())
        vl = json.loads((OUT / f"dist_scores_vllm_{name}.json").read_text())
        scores[name] = (np.asarray(hf["probs"]), np.asarray(vl["probs"]),
                        np.asarray(hf["logits"]), np.asarray(vl["logits"]))

    # compared panels share axes (CLAUDE.md): one p-range across all datasets
    all_p = np.concatenate([np.concatenate(s[:2]) for s in scores.values()])
    pad = 0.04 * np.ptp(all_p)
    lo, hi = all_p.min() - pad, all_p.max() + pad
    bins = np.histogram_bin_edges(all_p, bins=50)

    fig, axes = plt.subplots(len(names), 2, figsize=(9.5, 3.6 * len(names)),
                             layout="constrained")
    axes = np.atleast_2d(axes)
    report = {}
    for i, name in enumerate(names):
        ph, pv, lh, lv = scores[name]
        r = float(np.corrcoef(lh, lv)[0, 1])
        report[name] = {"n": len(ph), "pearson_r_logits": r,
                        "max_abs_dp": float(np.abs(ph - pv).max()),
                        "mean_abs_dp": float(np.abs(ph - pv).mean())}

        ax = axes[i, 0]
        ax.hist(ph, bins=bins, color=BLUE, alpha=0.3)
        ax.hist(ph, bins=bins, color=BLUE, histtype="step", lw=1.5, label="HF probe")
        ax.hist(pv, bins=bins, color=AQUA, histtype="step", lw=2, label="vLLM served")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("p(deceptive)")
        ax.set_ylabel("conversations")
        ax.set_title(f"{name} (n={len(ph)})", color=INK)
        ax.legend(frameon=False)

        ax = axes[i, 1]
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
        ax.scatter(ph, pv, s=12, color=BLUE, alpha=0.5, edgecolors="none", zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("HF probe p(deceptive)")
        ax.set_ylabel("vLLM served p(deceptive)")
        ax.set_title(f"r={r:.5f}, max|Δp|={report[name]['max_abs_dp']:.1e}", color=INK)

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e5e4e0", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)

    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "vllm_vs_hf.png", dpi=150)
    (OUT / "dist_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


def stage_serve_smoke(cfg: ServingConfig, port: int = 8321):
    """Start `vllm serve` on the published checkpoint, POST /classify, and
    check against the offline scores."""
    import subprocess
    import time
    import urllib.request

    rows = read_jsonl(OUT / "parity_tokens.jsonl")[:3]
    offline = json.loads((OUT / "scores_vllm.json").read_text())["probs"][:3]

    proc = subprocess.Popen(
        ["vllm", "serve", str(REPO / cfg.publish_dir),
         "--runner", "pooling", "--max-model-len", str(cfg.max_model_len),
         "--enforce-eager", "--port", str(port)])
    try:
        deadline = time.time() + 600
        while True:
            try:
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
                break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError("vllm serve exited during startup")
                if time.time() > deadline:
                    raise TimeoutError("vllm serve did not become healthy")
                time.sleep(5)
        body = json.dumps({"input": [r["text"] for r in rows]}).encode()
        req = urllib.request.Request(f"http://localhost:{port}/classify", data=body,
                                     headers={"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(req, timeout=120).read())
        served = [d["probs"][0] for d in data["data"]]
        # batch composition differs between server and the offline 64-prompt
        # run, so bf16 kernel noise of ~1e-4 on p is expected (the cross-stack
        # parity itself sits at ~5e-4); 1e-3 still catches any real drift
        drift = max(abs(a - b) for a, b in zip(served, offline))
        print(f"served probs {served} vs offline {offline}; max drift {drift:.3e}")
        if drift > 1e-3:
            raise AssertionError(f"server scores drift from offline classify by {drift:.3e}")
        print("PASS")
    finally:
        proc.terminate()
        proc.wait(timeout=60)


STAGES = {"publish": stage_publish, "score_hf": stage_score_hf,
          "score_vllm": stage_score_vllm, "compare": stage_compare,
          "push": stage_push, "serve_smoke": stage_serve_smoke,
          "dist_score_hf": stage_dist_score_hf,
          "dist_score_vllm": stage_dist_score_vllm,
          "dist_compare": stage_dist_compare}


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--config", default=str(OUT / "config.json"),
                    help="ServingConfig json; artifacts go to its out_dir "
                         "(default: this experiment's config.json)")
    args = ap.parse_args()
    cfg = ServingConfig.load(args.config)
    OUT = REPO / cfg.out_dir
    cfg.save(OUT)
    STAGES[args.stage](cfg)


if __name__ == "__main__":   # vLLM spawns workers that re-import this module
    main()
