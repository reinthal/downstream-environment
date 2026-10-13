"""Positive control for the activation pipeline (PREREGISTRATION.md §5):
score cfg.pc_rows fibs-v1 rows with our own activation collection and the
published linear ensemble; the AUROC must be within cfg.pc_tolerance of the
card's validation AUROC (0.972). Deviation 2026-10-12: fibs carries no
validation marker, so the rows are a seeded random balanced draw.

Also checks, on a few rows with the axial and EFC probes, that scoring the
saved window alone equals scoring the full sequence (the two must agree
because probe_inference compacts the read tokens before scoring).

  positive_control.json   AUROC per architecture, pass/fail, the row ids

Run:  CUDA_VISIBLE_DEVICES=1,2 uv run --no-sync python experiments/2026-10-12-fibs-probes/positive_control.py
"""
from __future__ import annotations

import glob
import json
import random

import numpy as np
import pyarrow.parquet as pq
import torch

from acts import collect_windows, load_model, render_followup, window_masks
from common import OUT, load_config, model_path, probe_dir

from probe_inference import load_probe  # noqa: E402
from probe_inference.score import build_read_mask  # noqa: E402

QUADRANTS = ("honest_honest", "honest_deceptive", "deceptive_honest", "deceptive_deceptive")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def sample_rows(cfg) -> list[dict]:
    from huggingface_hub import snapshot_download
    root = snapshot_download(cfg.fibs_repo, repo_type="dataset", revision=cfg.fibs_revision,
                             allow_patterns=["data/*"])
    rng = random.Random(cfg.pc_seed)
    per = cfg.pc_rows // len(QUADRANTS)
    rows = []
    for q in QUADRANTS:
        (path,) = glob.glob(f"{root}/data/{q}-*.parquet")
        tbl = pq.read_table(path, columns=["messages_json", "completion", "completion_reasoning",
                                           "tools", "quadrant", "source", "_v32_source_content_md5"])
        cand = [i for i in range(tbl.num_rows)
                if not tbl["tools"][i].as_py() and not tbl["completion_reasoning"][i].as_py()]
        for i in rng.sample(cand, per):
            r = {k: tbl[k][i].as_py() for k in ("completion", "quadrant", "source", "_v32_source_content_md5")}
            r["messages"] = json.loads(tbl["messages_json"][i].as_py())
            r["label"] = int(q.endswith("_deceptive"))
            rows.append(r)
    return rows


def main():
    cfg = load_config()
    rows = sample_rows(cfg)
    model, tok = load_model(model_path(cfg), max(cfg.probe_layers), dtype=cfg.dtype, device=cfg.device)
    rendered = []
    for r in rows:
        msgs = r["messages"]
        if msgs[-1]["role"] != "user":
            raise ValueError("fibs row does not end with the follow-up user turn")
        rendered.append(render_followup(tok, msgs[:-1], msgs[-1]["content"], r["completion"]))
    print(f"{len(rows)} rows, longest {max(x.n_tokens for x in rendered)} tokens", flush=True)
    acts = collect_windows(model, tok, rendered, cfg.probe_layers, cfg.acts_batch_tokens, cfg.acts_max_batch)

    labels = np.array([r["label"] for r in rows])
    result = {"n": len(rows), "seed": cfg.pc_seed, "target": cfg.pc_target_auroc,
              "tolerance": cfg.pc_tolerance, "auroc": {}}
    probes = {arch: load_probe(probe_dir(cfg, arch)) for arch in cfg.probe_archs}
    for arch, p in probes.items():
        scores = []
        for rd, a in zip(rendered, acts):
            pm, cm = window_masks(rd)
            mask = build_read_mask(p.read_window, pm, cm, torch.tensor(0))
            scores.append(p.score({L: a[i] for i, L in enumerate(p.layers)}, mask))
        result["auroc"][arch] = auroc(np.array(scores), labels)
        print(f"{arch}: AUROC {result['auroc'][arch]:.4f}", flush=True)
    primary = result["auroc"][cfg.primary_arch]
    result["pass"] = abs(primary - cfg.pc_target_auroc) <= cfg.pc_tolerance

    # window-only vs full-sequence scoring must agree (compaction before scoring)
    inner = model
    while hasattr(inner, "model"):
        inner = inner.model
    checks = []
    for rd, a in list(zip(rendered, acts))[:4]:
        with torch.no_grad():
            res = inner(input_ids=torch.tensor([rd.ids]).to(model.device), output_hidden_states=True)
        full = {L: res.hidden_states[L + 1][0].to(torch.bfloat16).cpu() for L in cfg.probe_layers}
        pm_full = torch.ones(rd.n_tokens, dtype=torch.bool)
        pm_full[rd.completion_start:] = False
        cm_full = ~pm_full
        pm, cm = window_masks(rd)
        for arch in ("efc", "axial", "linear"):
            p = probes[arch]
            s_full = p.score(full, build_read_mask(p.read_window, pm_full, cm_full, torch.tensor(rd.followup_start)))
            s_win = p.score({L: a[i] for i, L in enumerate(p.layers)}, build_read_mask(p.read_window, pm, cm, torch.tensor(0)))
            checks.append({"arch": arch, "full": s_full, "window": s_win})
        del res
    # bf16 batch-composition noise is ~0.3%; compaction itself is exact
    result["window_equals_full"] = all(abs(c["full"] - c["window"]) < 1e-2 * max(1.0, abs(c["full"])) for c in checks)
    result["window_checks"] = checks
    result["rows"] = [{"quadrant": r["quadrant"], "source": r["source"], "md5": r["_v32_source_content_md5"]} for r in rows]
    (OUT / "positive_control.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"primary ({cfg.primary_arch}) AUROC {primary:.4f} vs {cfg.pc_target_auroc} +- {cfg.pc_tolerance}: "
          f"{'PASS' if result['pass'] else 'FAIL'}; window==full: {result['window_equals_full']}")


if __name__ == "__main__":
    main()
