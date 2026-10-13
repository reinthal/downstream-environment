"""Modal app for the GPU stages: fibs rendering (CPU), window-activation
collection for fibs rows and BashArena steps on one H200 (bf16, decoder
truncated at the top probe layer), probe training (``train.py``), step
scoring, and the published-probe collector check.

Layout on the ``fibs-basharena`` Volume (``/data``):
  hf/                      fibs-v1 parquet snapshot, published probe weights
  fibs/{tag}/rows.jsonl    one line per row: i, label, quadrant, source (family), pair_id, md5, split, n_tokens, window, dropped
  fibs/{tag}/jobs_{k}.pt   rendered rows of shard k: [{i, ids, followup_start, completion_start, n_tokens}]
  fibs/{tag}/acts_{k}.pt   {"i": [...], "acts": [(6, W, d) bf16], "prompt_mask": [...], "completion_mask": [...]}
  steps/{tag}/transcripts_{mode}.jsonl   copied from the experiment folder (modal volume put)
  steps/{tag}/{mode}/{sample_id}.pt      {"index": [...], "steps": {step: {prefill: {acts, prompt_mask, completion_mask}}}}
  probes/{tag}/{arch}/     trained probe directories (probe_inference format) + {arch}_train.json

Drivers (this host has no GPU and no Python outside uvx; run from the repo root):
  uvx modal run experiments/2026-10-13-fibs-basharena/modal_app.py::prepare --tag pilot --n-rows 2000
  uvx modal run experiments/2026-10-13-fibs-basharena/modal_app.py::collect_fibs --tag pilot
  uvx modal run experiments/2026-10-13-fibs-basharena/modal_app.py::collect_steps --tag pilot --mode attack --uids 1216794-1,12352587-1
  uvx modal run experiments/2026-10-13-fibs-basharena/modal_app.py::train --tag pilot --archs linear,mlp,efc,axial
  uvx modal run experiments/2026-10-13-fibs-basharena/modal_app.py::score --tag pilot --probe-tag pilot --mode attack --uids 1216794-1,12352587-1
  uvx modal run experiments/2026-10-13-fibs-basharena/modal_app.py::published_check --n-rows 500
"""
# no `from __future__ import annotations`: modal.parameter needs a real type annotation
import json
import sys
from pathlib import Path

import modal

LOCAL = modal.is_local()                 # inside the container the code lives under /root/src (see image)
HERE = Path(__file__).resolve().parent if LOCAL else Path("/root/src/exp")
REPO = HERE.parents[1] if LOCAL else Path("/root/src")
sys.path.insert(0, str(HERE))
from common import FibsBashArenaConfig, OUT  # noqa: E402

CFG = FibsBashArenaConfig.load(OUT / "config.json")
REVISIONS = {CFG.model_id: CFG.model_revision, CFG.pc_model_id: CFG.pc_model_revision}
MINUTES = 60
DATA = "/data"
HF_CACHE = "/root/.cache/huggingface"      # the weights Volume (serving/modal_qwen_27b.py::download)

image = (
    # CUDA 13 devel base (torch 2.14's PyPI wheel is a CUDA 13.0 build; Modal's hosts run it): causal-conv1d,
    # the GDN layers' conv kernel, builds from source against it (the reference fallback is ~5x slower)
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build")
    .uv_pip_install(
        "torch==2.14.0", "transformers==5.17.0", "flash-linear-attention==0.5.2", "accelerate==1.15.0",
        "huggingface_hub", "pyarrow", "numpy", "scikit-learn", "packaging", "wheel", "setuptools",
    )
    .env({"TORCH_CUDA_ARCH_LIST": "9.0", "MAX_JOBS": "16", "CC": "gcc", "CXX": "g++"})   # sm90 (H100/H200); torch's
    # extension builder misreads the default compiler as clang++ 0.0.0 without CC/CXX
    .pip_install("causal-conv1d==1.7.0", extra_options="--no-build-isolation")
    .env({"PYTHONPATH": "/root/src:/root/src/exp", "TOKENIZERS_PARALLELISM": "false"})
)
if LOCAL:
    image = (
        image.add_local_dir(REPO / "linear_probes", "/root/src/linear_probes")
        .add_local_dir(REPO / "vendor" / "caught-in-the-act-probes" / "src" / "probe_inference", "/root/src/probe_inference")
        .add_local_dir(HERE, "/root/src/exp", ignore=["*.jsonl", "*.pt", "*.npz", "*.csv", "__pycache__", "figures"])
    )
weights = modal.Volume.from_name(CFG.modal_weights_volume)
data = modal.Volume.from_name(CFG.modal_data_volume, create_if_missing=True)
triton_cache = modal.Volume.from_name(CFG.modal_triton_cache_volume, create_if_missing=True)
VOLUMES = {HF_CACHE: weights, DATA: data, "/root/.triton": triton_cache}   # TRITON_CACHE_DIR defaults to ~/.triton/cache
hf_secret = modal.Secret.from_name(CFG.modal_hf_secret)

app = modal.App("fibs-basharena")

QUADRANTS = ("honest_honest", "honest_deceptive", "deceptive_honest", "deceptive_deceptive")
ROW_COLUMNS = ["messages_json", "completion", "completion_reasoning", "tools", "quadrant", "source",
               "pair_id", "_v32_source_content_md5"]


def tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, revision=REVISIONS[model_id], local_files_only=True)


def fibs_root() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(CFG.fibs_repo, repo_type="dataset", revision=CFG.fibs_revision,
                             allow_patterns=["data/*"], cache_dir=f"{DATA}/hf")


# ───────────────────────────── fibs rows ─────────────────────────────

@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], cpu=8, memory=32768, timeout=4 * 60 * MINUTES)
def prepare_fibs(tag: str, n_rows: int | None, seed: int, model_id: str) -> dict:
    """Read the four quadrant parquets, hold out the validation rows, render
    every row with the follow-up contract for ``model_id``'s tokenizer and
    write the job shards. ``n_rows`` (pilot) draws a seeded balanced subset
    per quadrant before the split."""
    import random
    from collections import Counter
    import pyarrow.parquet as pq
    import torch
    from acts import render_completion

    root = fibs_root()
    rng = random.Random(seed)
    rows = []
    for q in QUADRANTS:
        (path,) = Path(root).glob(f"data/{q}-*.parquet")
        tbl = pq.read_table(path, columns=ROW_COLUMNS)
        idx = list(range(tbl.num_rows))
        if n_rows is not None:
            idx = rng.sample(idx, n_rows // len(QUADRANTS))
        for i in idx:
            r = {k: tbl[k][i].as_py() for k in ROW_COLUMNS if k != "messages_json"}
            r["messages"] = json.loads(tbl["messages_json"][i].as_py())
            r["label"] = int(q.endswith("_deceptive"))     # y = 1 on falsehood completions
            rows.append(r)
    print(f"{len(rows)} rows read", flush=True)

    # validation hold-out: class-balanced, no pair_id shared with training (paper A.6)
    groups: dict[str, list[int]] = {}
    for j, r in enumerate(rows):
        groups.setdefault(r["pair_id"] or f"_row{j}", []).append(j)
    keys = list(groups)
    rng.shuffle(keys)
    n_val = CFG.n_val if n_rows is None else max(2, n_rows // 10)
    val, counts = set(), [0, 0]
    for k in keys:
        members = groups[k]
        lab = [rows[j]["label"] for j in members]
        if sum(counts) + len(members) > n_val:
            continue
        if counts[1] + sum(lab) > n_val // 2 or counts[0] + len(lab) - sum(lab) > n_val // 2:
            continue
        val.update(members)
        counts[0] += len(lab) - sum(lab)
        counts[1] += sum(lab)
        if sum(counts) >= n_val:
            break
    for j, r in enumerate(rows):
        r["split"] = "val" if j in val else "train"
    print(f"validation {counts[0]} honest + {counts[1]} deceptive of {len(rows)}", flush=True)

    tok = tokenizer(model_id)
    out = Path(DATA) / "fibs" / tag
    out.mkdir(parents=True, exist_ok=True)
    jobs, n_drop, longest, shard, meta = [], 0, 0, 0, []
    last_roles = Counter()
    for j, r in enumerate(rows):
        msgs = r["messages"]
        last_roles[msgs[-1]["role"]] += 1      # user (the follow-up) or tool (tool follow-up rows)
        for m in msgs:          # fibs stores tool-call arguments as JSON strings; the template wants mappings
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", tc)
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"])
        tools = json.loads(r["tools"]) if r["tools"] else None
        rd = render_completion(tok, msgs, r["completion"], tools=tools, template_kwargs=CFG.template_kwargs)
        longest = max(longest, rd.n_tokens)
        entry = {k: r[k] for k in ("label", "quadrant", "source", "pair_id", "_v32_source_content_md5", "split")}
        entry.update(i=j, n_tokens=rd.n_tokens, window=rd.n_tokens - rd.followup_start)
        if rd.n_tokens > CFG.acts_max_len:
            entry["dropped"] = True
            n_drop += 1
        else:
            entry["dropped"] = False
            jobs.append({"i": j, "ids": rd.ids, "followup_start": rd.followup_start,
                         "completion_start": rd.completion_start, "n_tokens": rd.n_tokens})
            if len(jobs) == CFG.fibs_shard_rows:
                torch.save(jobs, out / f"jobs_{shard}.pt")
                shard += 1
                jobs = []
        meta.append(entry)
        if j % 5000 == 0:
            print(f"  rendered {j}/{len(rows)}", flush=True)
    if jobs:
        torch.save(jobs, out / f"jobs_{shard}.pt")
        shard += 1
    with open(out / "rows.jsonl", "w") as f:
        for e in meta:
            f.write(json.dumps(e) + "\n")
    data.commit()
    windows = sorted(e["window"] for e in meta)
    summary = {"tag": tag, "model_id": model_id, "rows": len(rows), "dropped": n_drop, "shards": shard,
               "longest": longest, "val": counts, "window_tokens_p50": windows[len(windows) // 2],
               "window_tokens_p90": windows[int(len(windows) * 0.9)], "window_tokens_max": windows[-1],
               "total_tokens": sum(e["n_tokens"] for e in meta), "last_roles": dict(last_roles)}
    print(json.dumps(summary), flush=True)
    return summary


# ───────────────────────────── activation collection ─────────────────────────────

@app.cls(image=image, gpu=CFG.modal_gpu, volumes=VOLUMES, timeout=6 * 60 * MINUTES, memory=131072,
         scaledown_window=2 * MINUTES, max_containers=4)
class Collector:
    model_id: str = modal.parameter(default=CFG.model_id)

    @modal.enter()
    def load(self):
        from huggingface_hub import snapshot_download
        from acts import load_model
        path = snapshot_download(self.model_id, revision=REVISIONS[self.model_id], local_files_only=True)
        self.model, self.tok = load_model(path, max(CFG.probe_layers), dtype=CFG.dtype, device="cuda")
        print(f"{self.model_id} loaded", flush=True)

    @modal.method()
    def bench(self, n_tokens: int = 16384, batch: int = 1) -> dict:
        """Forward throughput of the loaded decoder and which kernels are in
        use (flash-linear-attention chunk kernels, conv kernel, attention)."""
        import importlib
        import time
        import torch
        from acts import _decoder
        inner = _decoder(self.model)
        info = {"attn_implementation": getattr(inner.config, "_attn_implementation", None),
                "fla": importlib.util.find_spec("fla") is not None,
                "causal_conv1d": importlib.util.find_spec("causal_conv1d") is not None,
                "dtype": str(next(inner.parameters()).dtype), "n_blocks": len(inner.layers)}
        try:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as m
            info["modeling_flags"] = {k: bool(v) for k, v in vars(m).items()
                                      if k.startswith("is_") and isinstance(v, bool)}
            info["chunk_fn"] = getattr(getattr(m, "chunk_gated_delta_rule", None), "__module__", None)
        except Exception as e:                      # noqa: BLE001 (diagnostic only)
            info["modeling_error"] = repr(e)
        # which gated-delta-rule / conv functions the GDN layers call
        import inspect
        import re
        src = inspect.getsource(m) if "m" in dir() else ""
        info["fla_lines"] = [l.strip() for l in src.splitlines()
                             if re.search(r"fla\b|gated_delta|causal_conv1d|is_fast_path|flash_linear", l)][:25]
        try:
            import fla
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _c
            info["fla_version"] = getattr(fla, "__version__", "?")
            info["fla_chunk_import"] = "ok"
        except Exception as e:                      # noqa: BLE001
            info["fla_chunk_import"] = repr(e)
        gdn = next(l for l in inner.layers if hasattr(l, "linear_attn")).linear_attn
        info["gdn_attrs"] = {k: (getattr(v, "__module__", None) or type(v).__name__) for k, v in vars(gdn).items()
                             if callable(v) or "fn" in k or "rule" in k or "conv" in k}
        ids = torch.randint(1000, 20000, (batch, n_tokens), device=self.model.device)
        with torch.no_grad():
            inner(input_ids=ids[:, :512], use_cache=False)      # warm-up
            torch.cuda.synchronize()
            reps = []
            for _ in range(3):                                   # same shape thrice: autotune / launch overhead?
                t = time.time()
                inner(input_ids=ids, use_cache=False)
                torch.cuda.synchronize()
                reps.append(round(time.time() - t, 2))
            dt = reps[-1]
            info["repeat_seconds"] = reps
            shapes = {}
            for n in (n_tokens - 1, n_tokens - 7, n_tokens + 3):   # fresh but similar shapes
                t = time.time()
                inner(input_ids=ids[:, :n] if n <= n_tokens else torch.cat([ids, ids[:, :n - n_tokens]], 1),
                      use_cache=False)
                torch.cuda.synchronize()
                shapes[n] = round(time.time() - t, 2)
            info["fresh_shape_seconds"] = shapes
            # profile a 4k forward: top CUDA ops
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                inner(input_ids=ids[:, :4096], use_cache=False)
                torch.cuda.synchronize()
            info["profile"] = prof.key_averages().table(sort_by="cuda_time_total", row_limit=14)
        info.update(n_tokens=n_tokens, batch=batch, seconds=round(dt, 2), tokens_per_s=round(batch * n_tokens / dt))
        print(json.dumps({k: v for k, v in info.items() if k != "profile"}), flush=True)
        print(info["profile"], flush=True)
        return info

    @modal.method()
    def fla_bench(self, T: int = 16384) -> dict:
        """fla's chunk_gated_delta_rule alone at the model's GDN shapes, plus
        the model's first GDN layer alone, with kernel-launch counts."""
        import os
        import time
        import torch
        from torch.profiler import ProfilerActivity, profile
        from acts import _decoder
        inner = _decoder(self.model)
        gdn_layer = next(l for l in inner.layers if hasattr(l, "linear_attn"))
        gdn = gdn_layer.linear_attn
        cfg = inner.config
        H, HV = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        K, V = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        info = {"T": T, "H": H, "HV": HV, "K": K, "V": V, "env": {k: os.environ.get(k) for k in
                ("FLA_CACHE_RESULTS", "FLA_DISABLE_TENSOR_CACHE", "TRITON_INTERPRET", "FLA_USE_TMA", "TRITON_CACHE_DIR")}}
        try:
            import triton
            info["triton_target"] = str(triton.runtime.driver.active.get_current_target())
        except Exception as e:                      # noqa: BLE001
            info["triton_target"] = repr(e)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        dev = self.model.device
        q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, T, HV, V, device=dev, dtype=torch.bfloat16)
        g = torch.rand(1, T, HV, device=dev, dtype=torch.float32).log()
        beta = torch.rand(1, T, HV, device=dev, dtype=torch.bfloat16)

        def run(tag, fn, *a, **kw):
            fn(*a, **kw)
            torch.cuda.synchronize()
            t = time.time()
            fn(*a, **kw)
            torch.cuda.synchronize()
            info[tag] = round(time.time() - t, 3)

        with torch.no_grad():
            run("fla_op_s", chunk_gated_delta_rule, q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
            q2 = q.clone()                          # different tensor identity: defeats a result cache
            t = time.time()
            chunk_gated_delta_rule(q2, k.clone(), v.clone(), g.clone(), beta.clone(), use_qk_l2norm_in_kernel=True)
            torch.cuda.synchronize()
            info["fla_op_fresh_tensors_s"] = round(time.time() - t, 3)
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                chunk_gated_delta_rule(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
                                       use_qk_l2norm_in_kernel=True)
                torch.cuda.synchronize()
            info["fla_op_profile"] = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            # the model's GDN layer alone on hidden states
            x = torch.randn(1, T, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
            pos = torch.arange(T, device=dev)[None]
            try:
                run("gdn_layer_s", gdn, x, cache_params=None, cache_position=pos, attention_mask=None)
            except TypeError as e:
                info["gdn_layer_s"] = repr(e)
        print(json.dumps({k: v for k, v in info.items() if k != "fla_op_profile"}), flush=True)
        print(info["fla_op_profile"], flush=True)
        return info

    @modal.method()
    def fibs_shard(self, tag: str, k: int) -> dict:
        import time
        import torch
        from acts import Rendered, collect_windows, window_masks
        t0 = time.time()
        out = Path(DATA) / "fibs" / tag
        jobs = torch.load(out / f"jobs_{k}.pt", weights_only=True)
        rendered = [Rendered(j["ids"], j["followup_start"], j["completion_start"], j["n_tokens"]) for j in jobs]
        from probe_inference.score import LAST_ANSWER_TOKEN, build_read_mask
        acts = collect_windows(self.model, self.tok, rendered, CFG.probe_layers,
                               CFG.acts_batch_tokens, CFG.acts_max_batch, log=lambda s: None, pad_to=CFG.acts_pad_to,
                               batch_sizes=CFG.acts_fibs_batches)
        tokens, windows, pms, cms, n_capped = [], [], [], [], 0
        for r, a in zip(rendered, acts):
            pm, cm = window_masks(r)
            (pos,) = build_read_mask(LAST_ANSWER_TOKEN, pm, cm, torch.tensor(0)).nonzero().flatten().tolist()
            tokens.append(a[:, pos, :].clone())                 # (L, d): the linear/mlp read token
            if CFG.store_windows:
                if a.shape[1] > CFG.window_cap:                # keep the END of the window for efc/axial
                    n_capped += 1
                    a, pm, cm = a[:, -CFG.window_cap:], pm[-CFG.window_cap:], cm[-CFG.window_cap:]
                windows.append(a.clone())
                pms.append(pm)
                cms.append(cm)
        del acts
        torch.save({"i": [j["i"] for j in jobs], "token": torch.stack(tokens), "acts": windows,
                    "prompt_mask": pms, "completion_mask": cms,
                    "window_cap": CFG.window_cap if CFG.store_windows else 0}, out / f"acts_{k}.pt")
        data.commit()
        n_tok = sum(r.n_tokens for r in rendered)
        summary = {"shard": k, "rows": len(jobs), "tokens": n_tok, "seconds": round(time.time() - t0, 1),
                   "capped": n_capped, "bytes": sum(a.numel() * 2 for a in windows)}
        print(json.dumps(summary), flush=True)
        return summary

    @modal.method()
    def steps(self, tag: str, mode: str, uid: str) -> dict:
        """Every model step of one trajectory, both prefills, as the paper
        reads it: the messages up to and including the step's assistant
        message (tool calls; reasoning dropped by the template), then the
        follow-up and the prefilled answer. The step's own tool results are
        NOT in the input (paper H.2)."""
        import time
        import torch
        from acts import collect_trajectory, collect_windows, render_followup, window_masks
        root = Path(DATA) / "steps" / tag
        traj = None
        with open(root / f"transcripts_{mode}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["uid"] == uid:
                    traj = r
                    break
        if traj is None:
            raise ValueError(f"{mode} trajectory {uid} not in transcripts_{mode}.jsonl")
        if traj["error"] is not None:
            raise ValueError(f"{mode} trajectory {uid} errored: {traj['error']}")
        t0 = time.time()
        msgs = traj["messages"]
        index, todo = [], []
        for s in traj["steps"]:
            if s["forced"]:
                continue
            prefix = [to_template(m) for m in msgs[: s["msg_index"] + 1]]
            entry = {"step": s["step"], "msg_index": s["msg_index"], "n_tokens": {}, "dropped": False}
            rds = {}
            for prefill in CFG.prefills:
                rd = render_followup(self.tok, prefix, CFG.followup_question, prefill, tools=traj["tools"],
                                     template_kwargs=CFG.template_kwargs)
                entry["n_tokens"][prefill] = rd.n_tokens
                rds[prefill] = rd
                if rd.n_tokens > CFG.acts_max_len:
                    entry["dropped"] = True
            if not entry["dropped"]:
                todo.append({"step": s["step"], "rendered": rds})
            index.append(entry)
        if CFG.step_collector == "trajectory":
            acts = collect_trajectory(self.model, self.tok, todo, CFG.probe_layers, log=lambda s: None)
            timing = acts.pop("timing")
        elif CFG.step_collector == "windows":
            flat = [(s["step"], prefill, rd) for s in todo for prefill, rd in s["rendered"].items()]
            got = collect_windows(self.model, self.tok, [rd for _, _, rd in flat], CFG.probe_layers,
                                  CFG.acts_batch_tokens, CFG.acts_max_batch, log=lambda s: None, pad_to=CFG.acts_pad_to,
                                  batch_sizes=[CFG.acts_fixed_batch])
            acts = {(step, prefill): a for (step, prefill, _), a in zip(flat, got)}
            timing = {"collector": "windows", "forwards": collect_windows.last_timings}
        else:
            raise ValueError(f"unknown step_collector {CFG.step_collector!r}")
        per_step: dict = {}
        for s in todo:
            for prefill, rd in s["rendered"].items():
                pm, cm = window_masks(rd)
                per_step.setdefault(s["step"], {})[prefill] = {"acts": acts[(s["step"], prefill)],
                                                               "prompt_mask": pm, "completion_mask": cm}
        out = root / mode
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"index": index, "steps": per_step}, out / f"{uid}.pt")
        data.commit()
        summary = {"uid": uid, "mode": mode, "steps": len(index),
                   "dropped": sum(e["dropped"] for e in index),
                   "max_tokens": max(max(e["n_tokens"].values()) for e in index),
                   "total_tokens": sum(sum(e["n_tokens"].values()) for e in index),
                   "seconds": round(time.time() - t0, 1), "timing": timing}
        print(json.dumps(summary), flush=True)
        return summary


def to_template(m: dict) -> dict:
    """An extracted message in the chat template's expected shape."""
    if m["role"] == "assistant":
        d = {"role": "assistant", "content": m["content"], "reasoning_content": m["reasoning"]}
        if m["tool_calls"]:
            d["tool_calls"] = [{"type": "function", "function": {"name": tc["function"]["name"],
                                                                   "arguments": tc["function"]["arguments"]}}
                               for tc in m["tool_calls"]]
        return d
    if m["role"] == "tool":
        return {"role": "tool", "content": m["content"]}
    return {"role": m["role"], "content": m["content"]}


# ───────────────────────────── training and scoring ─────────────────────────────

@app.function(image=image, gpu=CFG.modal_train_gpu, volumes=VOLUMES, cpu=8, memory=131072, timeout=12 * 60 * MINUTES)
def train_probe(tag: str, arch: str) -> dict:
    from dataclasses import asdict
    from train import train_arch
    from probe_inference import load_probe
    fibs_dir = Path(DATA) / "fibs" / tag
    out = Path(DATA) / "probes" / tag
    result = train_arch(arch, fibs_dir, out, asdict(CFG))
    probe = load_probe(out / arch)            # the vendored loader is the format check
    result["loaded"] = {"arch": probe.arch, "layers": list(probe.layers), "used_layers": list(probe.used_layers),
                        "read_window": probe.read_window}
    data.commit()
    result.pop("train_log")
    return result


def _score_entry(probe, entry: dict) -> float:
    import torch
    from probe_inference.score import build_read_mask
    acts = entry["acts"]
    mask = build_read_mask(probe.read_window, entry["prompt_mask"], entry["completion_mask"],
                           torch.tensor(0, device=acts.device))
    return probe.score({L: acts[i] for i, L in enumerate(probe.layers)}, mask)


@app.function(image=image, gpu="L4", volumes=VOLUMES, timeout=2 * 60 * MINUTES)
def score_steps(tag: str, probe_tag: str, mode: str, uid: str, archs: list[str]) -> list[dict]:
    import torch
    from probe_inference import load_probe
    probes = {a: load_probe(Path(DATA) / "probes" / probe_tag / a, device="cuda") for a in archs}
    blob = torch.load(Path(DATA) / "steps" / tag / mode / f"{uid}.pt", weights_only=True)
    (other,) = [p for p in CFG.prefills if p != CFG.lie_prefill]
    rows = []
    for e in blob["index"]:
        row = {"uid": uid, "mode": mode, "step": e["step"], "msg_index": e["msg_index"],
               "dropped": e["dropped"], "n_tokens": e["n_tokens"]}
        if not e["dropped"]:
            entries = blob["steps"][e["step"]]
            for a, p in probes.items():
                s = {prefill: _score_entry(p, {k: v.to("cuda") if hasattr(v, "to") else v for k, v in entries[prefill].items()})
                     for prefill in CFG.prefills}
                row[f"{a}_lie"], row[f"{a}_truth"] = s[CFG.lie_prefill], s[other]
                row[f"{a}_contrast"] = s[CFG.lie_prefill] - s[other]
        rows.append(row)
    return rows


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], cpu=8, memory=32768, timeout=60 * MINUTES)
def score_published(tag: str) -> dict:
    """The published Qwen3.5-27B linear ensemble on fibs rows collected with
    the new collection path (rendered and read with Qwen3.5-27B): the
    collector check of PREREGISTRATION §5."""
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from probe_inference import load_probe
    from train import auroc, load_rows
    root = snapshot_download(CFG.probe_repo, revision=CFG.probe_revision, cache_dir=f"{DATA}/hf",
                             allow_patterns=[f"{CFG.probe_model_dir}/linear/*"])
    probe = load_probe(Path(root) / CFG.probe_model_dir / "linear")
    fibs_dir = Path(DATA) / "fibs" / tag
    by_i = {r["i"]: r for r in load_rows(fibs_dir)}
    scores, labels = [], []
    for path in sorted(fibs_dir.glob("acts_*.pt")):
        blob = torch.load(path, weights_only=True)
        for i, a, pm, cm in zip(blob["i"], blob["acts"], blob["prompt_mask"], blob["completion_mask"]):
            # the linear probe reads the token before end-of-turn, inside the stored (end-kept) window
            scores.append(_score_entry(probe, {"acts": a, "prompt_mask": pm, "completion_mask": cm}))
            labels.append(by_i[i]["label"])
    value = auroc(np.array(scores), np.array(labels))
    result = {"tag": tag, "n": len(scores), "auroc": value, "target": CFG.pc_target_auroc,
              "tolerance": CFG.pc_tolerance, "pass": abs(value - CFG.pc_target_auroc) <= CFG.pc_tolerance}
    print(json.dumps(result), flush=True)
    return result


# ───────────────────────────── drivers ─────────────────────────────

def _vol_names(prefix: str) -> list[str]:
    vol = modal.Volume.from_name(CFG.modal_data_volume)
    return [e.path for e in vol.listdir(prefix)]


@app.local_entrypoint()
def prepare(tag: str = "pilot", n_rows: int = 0, seed: int = 0, model_id: str = CFG.model_id):
    print(prepare_fibs.remote(tag, n_rows or None, seed, model_id))


@app.local_entrypoint()
def collect_fibs(tag: str = "pilot", shards: str = "", model_id: str = CFG.model_id):
    import re
    names = _vol_names(f"fibs/{tag}")
    ks = sorted(int(m.group(1)) for n in names if (m := re.search(r"jobs_(\d+)\.pt$", n)))
    if shards:
        ks = [int(x) for x in shards.split(",")]
    done = {int(m.group(1)) for n in names if (m := re.search(r"acts_(\d+)\.pt$", n))}
    ks = [k for k in ks if k not in done]
    print(f"{len(ks)} shards to collect")
    for r in Collector(model_id=model_id).fibs_shard.map([tag] * len(ks), ks):
        print(r)


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], cpu=4, timeout=2 * 60 * MINUTES)
def download_weights() -> dict:
    """Both Qwen checkpoints into the weights Volume's HF cache (a new
    workspace starts empty; serving/modal_qwen_27b.py::download does the
    same with the serving image)."""
    from huggingface_hub import snapshot_download
    out = {m: snapshot_download(m, revision=r) for m, r in REVISIONS.items()}
    weights.commit()
    print(json.dumps(out), flush=True)
    return out


@app.local_entrypoint()
def fetch_weights():
    print(download_weights.remote())


@app.function(image=image, cpu=2, timeout=10 * MINUTES)
def gdn_source() -> str:
    """The loops and kernel calls in transformers' Qwen3.5 GDN layer forward."""
    import inspect
    import re
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    cls = next(v for k, v in vars(m).items() if k.endswith("GatedDeltaNet"))
    src = inspect.getsource(cls.forward)
    lines = src.splitlines()
    keep = [f"{i:4d} {l}" for i, l in enumerate(lines)
            if re.search(r"\bfor\b|\bwhile\b|chunk|recurrent|causal_conv1d|cu_seqlens|attention_mask|is_fast_path|"
                         r"use_cache|cache_params|split|range\(", l)]
    wrap = getattr(m, "torch_chunk_gated_delta_rule", None)
    extra = {k: str(getattr(wrap, k, None))[:200] for k in ("__wrapped__", "kernel", "_kernel", "func", "__module__")}
    hub = inspect.getsource(m.use_kernel_func_from_hub_with_fallback) if hasattr(m, "use_kernel_func_from_hub_with_fallback") else "n/a"
    return "\n".join(keep) + "\n---WRAP---\n" + json.dumps(extra) + "\n---HUB---\n" + hub[:3000]


@app.local_entrypoint()
def gdn_src():
    print(gdn_source.remote())


@app.function(image=image, cpu=2, timeout=10 * MINUTES)
def fla_diag() -> str:
    """Autotune keys and env switches of the installed flash-linear-attention."""
    import subprocess
    import fla
    root = Path(fla.__file__).parent
    cmd = (f"grep -rn -A8 'triton.autotune' {root}/ops/gated_delta_rule/*.py {root}/ops/common/chunk_h.py "
           f"{root}/ops/common/chunk_o.py {root}/ops/utils/solve_tril.py {root}/ops/utils/cumsum.py 2>/dev/null "
           f"| grep 'key=' | head -40; echo ---ENV---; grep -rn 'environ\\|getenv' {root}/ops/utils/cache.py "
           f"{root}/utils/*.py 2>/dev/null | head -40; echo ---TRITON---; python -c 'import triton,os; "
           f"print(triton.__version__); print(os.environ.get(\"TRITON_CACHE_DIR\"))'")
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


@app.local_entrypoint()
def diag():
    print(fla_diag.remote())


@app.local_entrypoint()
def bench(n_tokens: int = 16384, batch: int = 1):
    print(Collector().bench.remote(n_tokens, batch))


@app.local_entrypoint()
def fla_bench(t: int = 16384):
    Collector().fla_bench.remote(t)


@app.local_entrypoint()
def collect_steps(tag: str = "pilot", mode: str = "attack", uids: str = ""):
    ids = [s for s in uids.split(",") if s]
    try:
        done = {Path(n).stem for n in _vol_names(f"steps/{tag}/{mode}")}
    except Exception:                               # noqa: BLE001  (folder does not exist yet)
        done = set()
    skipped = [u for u in ids if u in done]
    ids = [u for u in ids if u not in done]
    print(f"{len(ids)} trajectories to collect ({len(skipped)} already on the volume)")
    for r in Collector().steps.map([tag] * len(ids), [mode] * len(ids), ids):
        print(r)


@app.local_entrypoint()
def train(tag: str = "pilot", archs: str = "linear,mlp,efc,axial"):
    names = [a for a in archs.split(",") if a]
    for r in train_probe.map([tag] * len(names), names):
        print(json.dumps(r))


@app.local_entrypoint()
def score(tag: str = "pilot", probe_tag: str = "pilot", mode: str = "attack", uids: str = "",
          archs: str = "linear,mlp,efc,axial", out: str = ""):
    ids = [s for s in uids.split(",") if s]
    names = [a for a in archs.split(",") if a]
    rows = []
    for r in score_steps.map([tag] * len(ids), [probe_tag] * len(ids), [mode] * len(ids), ids, [names] * len(ids)):
        rows.extend(r)
    path = Path(out) if out else HERE / f"scores_{tag}_{mode}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} step rows -> {path}")


@app.local_entrypoint()
def published_check(n_rows: int = 500, seed: int = 0):
    """Collector check: 500 fibs rows rendered and read with Qwen3.5-27B,
    scored by the published linear ensemble; must reproduce the card's 0.972
    within 0.02."""
    tag = "pc"
    print(prepare_fibs.remote(tag, n_rows, seed, CFG.pc_model_id))
    collect_fibs(tag=tag, model_id=CFG.pc_model_id)
    print(score_published.remote(tag))
