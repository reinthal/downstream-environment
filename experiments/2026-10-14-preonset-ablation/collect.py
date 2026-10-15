"""Local (A40, ``.venv``) collection + scoring for the AIS-404 pre-onset
ablation. Re-renders the stored trajectories with a modified system prompt,
recomputes the paper's SHADE read activations, and scores the fibs probes.

Reuses the main run's collection code verbatim (``acts.render_followup`` /
``collect_windows`` / ``window_masks``) and probe format
(``probe_inference.load_probe`` + ``build_read_mask``), so a condition
re-rendered with the *original* system prompt reproduces ``scores_main``
(the positive control).

Run (.venv, GPUs within allowance):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1,2,6,7 \
    uv run --no-sync python experiments/2026-10-14-preonset-ablation/collect.py \
      --conditions orig_honest,orig_preonset --pilot          # pilot + positive control
  ... --conditions swap,ctrl_neutral,ctrl_good,ctrl_evil      # the run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PILOT_UIDS = {"attack": ["1216794-1", "12352587-1"], "honest": ["1216794-1", "12352587-1"]}


def score_entry(probe, acts, pm, cm):
    import torch
    from probe_inference.score import build_read_mask
    acts = acts.to("cuda")
    mask = build_read_mask(probe.read_window, pm.to("cuda"), cm.to("cuda"),
                           torch.tensor(0, device="cuda"))
    return float(probe.score({L: acts[i] for i, L in enumerate(probe.layers)}, mask))


def _render_steps(tok, steps):
    """Render both prefills per step; flag steps over the context window."""
    from acts import render_followup
    out = []
    for s in steps:
        entry = {"condition": s["condition"], "uid": s["uid"], "step": s["step"],
                 "msg_index": s["msg_index"], "n_tokens": {}, "dropped": False, "rendered": {}}
        for prefill in C.CFG.prefills:
            rd = render_followup(tok, s["messages"], C.CFG.followup_question, prefill,
                                 tools=s["tools"], template_kwargs=C.CFG.template_kwargs)
            entry["n_tokens"][prefill] = rd.n_tokens
            entry["rendered"][prefill] = rd
            if rd.n_tokens > C.CFG.acts_max_len:
                entry["dropped"] = True
        out.append(entry)
    return out


def _row(e, scored, other, probes):
    row = {"condition": e["condition"], "uid": e["uid"], "step": e["step"],
           "msg_index": e["msg_index"], "dropped": e["dropped"], "n_tokens": e["n_tokens"]}
    if not e["dropped"]:
        for a_ in probes:
            lie, truth = scored[C.CFG.lie_prefill][a_], scored[other][a_]
            row[f"{a_}_lie"], row[f"{a_}_truth"], row[f"{a_}_score"] = lie, truth, lie - truth
    return row


def collect_condition(model, tok, probes, condition, uids=None, collector="trajectory",
                      chunk_inputs=200, save_acts=False, log=print):
    """Collect + score one condition. ``trajectory`` advances one KV cache per
    trajectory (O(steps), exact); ``windows`` re-runs each prefix (O(steps^2)).
    ``save_acts`` writes the window activations + masks per trajectory so any
    architecture (efc/axial) can be scored later without re-running the model."""
    import torch
    from acts import collect_trajectory, collect_windows, window_masks
    other = next(p for p in C.CFG.prefills if p != C.CFG.lie_prefill)
    acts_dir = C.OUT / "acts" / condition
    if save_acts:
        acts_dir.mkdir(parents=True, exist_ok=True)
    steps = C.condition_steps(condition)
    if uids is not None:
        keep = set(uids)
        steps = [s for s in steps if s["uid"] in keep]
    for s in steps:
        s["condition"] = condition
    uid_order = sorted({s["uid"] for s in steps})
    log(f"[{condition}] {len(steps)} steps, {len(uid_order)} uids, collector={collector}")

    rows, t0, done = [], time.time(), 0
    for uid in uid_order:
        rendered = _render_steps(tok, [s for s in steps if s["uid"] == uid])
        live = [e for e in rendered if not e["dropped"]]
        scored, store = {}, {}
        if collector == "trajectory":
            todo = [{"step": e["step"], "rendered": e["rendered"]} for e in live]
            acts = collect_trajectory(model, tok, todo, C.CFG.probe_layers, log=lambda s: None)
            acts.pop("timing", None)
            for e in live:
                sc = {}
                for prefill, rd in e["rendered"].items():
                    pm, cm = window_masks(rd)
                    if save_acts:
                        store.setdefault(e["step"], {})[prefill] = {
                            "acts": acts[(e["step"], prefill)], "prompt_mask": pm, "completion_mask": cm}
                    sc[prefill] = {a_: score_entry(p, acts[(e["step"], prefill)], pm, cm)
                                   for a_, p in probes.items()}
                scored[e["step"]] = sc
        else:
            flat = [(e, prefill, rd) for e in live for prefill, rd in e["rendered"].items()]
            for a in range(0, len(flat), chunk_inputs):
                part = flat[a:a + chunk_inputs]
                got = collect_windows(model, tok, [rd for _, _, rd in part], C.CFG.probe_layers,
                                      C.CFG.acts_batch_tokens, C.CFG.acts_max_batch, log=lambda s: None,
                                      pad_to=C.CFG.acts_pad_to, batch_sizes=[C.CFG.acts_fixed_batch])
                for (e, prefill, rd), a_acts in zip(part, got):
                    pm, cm = window_masks(rd)
                    scored.setdefault(e["step"], {})[prefill] = {
                        a_: score_entry(p, a_acts, pm, cm) for a_, p in probes.items()}
        index = [{k: e[k] for k in ("uid", "step", "msg_index", "n_tokens", "dropped")} for e in rendered]
        for e in rendered:
            rows.append(_row(e, scored.get(e["step"], {}), other, probes))
        if save_acts:
            torch.save({"index": index, "steps": store}, acts_dir / f"{uid}.pt")
        done += len(live) * len(C.CFG.prefills)
        log(f"[{condition}] {uid}: {len(live)}/{len(rendered)} steps kept; "
            f"{done} inputs, {time.time() - t0:.0f}s ({(time.time() - t0) / max(done, 1):.2f}s/input)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", required=True, help="comma list from CONDITIONS + PC_CONDITIONS")
    ap.add_argument("--archs", default=",".join(C.ARCHS_READY))
    ap.add_argument("--probe-tag", default="main", help="probes_main/<tag>")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--pilot", action="store_true", help="restrict to the pilot uids")
    ap.add_argument("--uids", default="", help="explicit uid subset (comma)")
    ap.add_argument("--collector", default="trajectory", choices=["trajectory", "windows"])
    ap.add_argument("--chunk-inputs", type=int, default=200)
    ap.add_argument("--save-acts", action="store_true", help="store window acts+masks per uid for later archs")
    args = ap.parse_args()

    from acts import load_model
    from probe_inference import load_probe
    archs = args.archs.split(",")
    probe_root = C.MAIN / "probes_main" / args.probe_tag
    probes = {a: load_probe(probe_root / a, device="cuda") for a in archs}
    print(f"loaded probes {list(probes)} from {probe_root}", flush=True)
    print(f"loading {C.MODEL_ID} truncated at layer {C.TOP_LAYER} ...", flush=True)
    t0 = time.time()
    model, tok = load_model(C.MODEL_ID, C.TOP_LAYER, dtype="bfloat16", device="auto")
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    for condition in args.conditions.split(","):
        uids = None
        if args.uids:
            uids = args.uids.split(",")
        elif args.pilot:
            uids = PILOT_UIDS[C.source_mode(condition)]
        rows = collect_condition(model, tok, probes, condition, uids=uids,
                                 collector=args.collector, chunk_inputs=args.chunk_inputs,
                                 save_acts=args.save_acts)
        out = C.OUT / f"scores_{args.tag}_{condition}.jsonl"
        C.write_jsonl(out, rows)
        n_drop = sum(r["dropped"] for r in rows)
        print(f"[{condition}] wrote {out.name}: {len(rows)} steps, {n_drop} dropped", flush=True)


if __name__ == "__main__":
    main()
