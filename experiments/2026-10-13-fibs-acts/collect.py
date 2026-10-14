"""Residual-stream activations of Qwen3.5-27B on every FIBSv1 row, at the one
token the paper's linear and MLP probes read, for training new probes.

A row is a finished transcript (``messages_json``, ending with the follow-up
user turn or, in b1-bare, a tool result) plus the assistant ``completion``.
It is rendered with the model's chat template and an empty think block
(vendor/caught-in-the-act-probes README, "Activation contract"); the stored
token is the one before the final ``<|im_end|>``, at ``cfg.layers``
(layer k = output of decoder block k), bf16.

Stages:
  render   CPU. rows.parquet: one line per fibs row (row = position in the
           concatenation of the four quadrant files in common.QUADRANTS
           order), its token count and whether it is dropped (over
           cfg.acts_max_len).
  collect  GPU, one process per shard. Kept rows are ranked by length and
           dealt out to the shards; chunks land in
           acts_fibs/shard{i}/chunk_{k}.pt. Re-running continues after the
           chunks already written.
  merge    acts_fibs/layer_{L}.pt, (n_kept, d_model) bf16, row order =
           rows.parquet kept rows (column ``act_index``).
  check    The published linear ensemble on the merged activations:
           AUROC of follow-up deceptive vs honest, written to check.json.

Run:
  uv run --no-sync python experiments/2026-10-13-fibs-acts/collect.py --stage render
  CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync python experiments/2026-10-13-fibs-acts/collect.py --stage collect --shard 0
  uv run --no-sync python experiments/2026-10-13-fibs-acts/collect.py --stage merge
  uv run --no-sync python experiments/2026-10-13-fibs-acts/collect.py --stage check
Add --pilot to any stage: cfg.pilot_rows random rows plus the longest kept
row, one shard, files under acts_fibs_pilot/.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import OUT, QUADRANTS, acts_dir, fibs_root, load_config, model_path

IM_END = "<|im_end|>"
META = ["source", "deception_type", "follow_up_category", "language", "pair_id", "pair_type",
        "_v32_source_content_md5"]
RENDER_CHUNK = 2000


def render(tok, messages_json: str, completion: str, tools_json: str | None) -> list[int]:
    """Token ids of the row, ending at the end-of-turn token. Fails loudly if
    the template does not render the answer after an empty think block."""
    conv = [*json.loads(messages_json), {"role": "assistant", "content": completion}]
    for m in conv:   # fibs stores tool-call arguments as the OpenAI JSON string; the template wants the mapping
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    kw = {"tools": json.loads(tools_json)} if tools_json else {}
    full = tok.apply_chat_template(conv, tokenize=False, **kw)
    gen = tok.apply_chat_template(conv[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False, **kw)
    if not (full.endswith(IM_END + "\n") and full.startswith(gen) and gen.endswith("<think>\n\n</think>\n\n")):
        raise ValueError("chat template does not render the row as the probes expect")
    ids = tok(full[:-1], add_special_tokens=False)["input_ids"]
    n_answer = len(ids) - len(tok(gen, add_special_tokens=False)["input_ids"])
    if n_answer < 2 or tok.decode(ids[-1:]) != IM_END:
        raise ValueError(f"the answer has {n_answer - 1} tokens before {IM_END}")
    return ids


def _count(job: tuple[str, str, int, int]) -> list[int]:
    path, model, lo, hi = job
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    t = pq.read_table(path, columns=["messages_json", "completion", "tools"]).slice(lo, hi - lo).to_pylist()
    return [len(render(tok, r["messages_json"], r["completion"], r["tools"])) for r in t]


def stage_render(cfg, pilot: bool):
    root, model = fibs_root(cfg), model_path(cfg)
    tables, jobs = [], []
    for q in QUADRANTS:
        (path,) = root.glob(f"{q}-*.parquet")
        t = pq.read_table(path, columns=META)
        tables.append(t.append_column("quadrant", pa.array([q] * t.num_rows))
                       .append_column("quadrant_index", pa.array(range(t.num_rows))))
        jobs += [(str(path), model, i, min(t.num_rows, i + RENDER_CHUNK)) for i in range(0, t.num_rows, RENDER_CHUNK)]
    with Pool(cfg.render_workers) as pool:
        n_tokens = np.array([n for chunk in pool.map(_count, jobs) for n in chunk])
    rows = pa.concat_tables(tables)
    dropped = n_tokens > cfg.acts_max_len
    if pilot:
        kept = np.flatnonzero(~dropped)
        chosen = set(random.Random(cfg.pilot_seed).sample(kept.tolist(), cfg.pilot_rows))
        chosen.add(int(kept[np.argmax(n_tokens[kept])]))
        dropped = np.array([i not in chosen for i in range(len(n_tokens))])
    act_index = np.where(dropped, -1, np.cumsum(~dropped) - 1)
    rows = (rows.append_column("row", pa.array(range(rows.num_rows)))
                .append_column("base_deceptive", pa.array([q.startswith("deceptive_") for q in rows["quadrant"].to_pylist()]))
                .append_column("followup_deceptive", pa.array([q.endswith("_deceptive") for q in rows["quadrant"].to_pylist()]))
                .append_column("n_tokens", pa.array(n_tokens))
                .append_column("dropped", pa.array(dropped))
                .append_column("act_index", pa.array(act_index)))
    acts_dir(pilot).mkdir(exist_ok=True)
    pq.write_table(rows, acts_dir(pilot) / "rows.parquet")
    kept = n_tokens[~dropped]
    print(f"{rows.num_rows} rows, {int((~dropped).sum())} kept ({int(kept.sum()):,} tokens, longest {int(kept.max())}), "
          f"{int((n_tokens > cfg.acts_max_len).sum())} over {cfg.acts_max_len} tokens")


def shard_rows(rows: pa.Table, shard: int, n_shards: int) -> list[int]:
    """Kept rows of this shard, longest first: rank by length, deal out."""
    kept = [(n, r) for n, r, d in zip(rows["n_tokens"].to_pylist(), rows["row"].to_pylist(),
                                      rows["dropped"].to_pylist()) if not d]
    return [r for _, r in sorted(kept, reverse=True)][shard::n_shards]


class ReadToken:
    """Forward hooks on the decoder blocks in ``layers`` that keep, for each
    row of the batch, the block's output at ``self.pos``."""

    def __init__(self, blocks, layers: list[int]):
        self.pos = None
        self.out: dict[int, "torch.Tensor"] = {}
        for L in layers:
            blocks[L].register_forward_hook(self._hook(L))

    def _hook(self, L: int):
        import torch

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self.out[L] = h[torch.arange(h.shape[0], device=h.device), self.pos.to(h.device)].to(torch.bfloat16).cpu()
        return hook


def stage_collect(cfg, pilot: bool, shard: int):
    import torch
    from linear_probes.activations import load_truncated_decoder

    n_shards = 1 if pilot else cfg.n_shards
    out_dir = acts_dir(pilot) / f"shard{shard}"
    out_dir.mkdir(exist_ok=True)
    rows = pq.read_table(acts_dir(pilot) / "rows.parquet", columns=["row", "quadrant", "quadrant_index", "n_tokens", "dropped"])
    where = dict(zip(rows["row"].to_pylist(), zip(rows["quadrant"].to_pylist(), rows["quadrant_index"].to_pylist(),
                                                  rows["n_tokens"].to_pylist())))
    done = {int(r) for p in out_dir.glob("chunk_*.pt") for r in torch.load(p)["row"]}
    todo = [r for r in shard_rows(rows, shard, n_shards) if r not in done]
    n_chunk = len(list(out_dir.glob("chunk_*.pt")))
    print(f"shard {shard}/{n_shards}: {len(todo)} rows to do, {len(done)} done", flush=True)

    root = fibs_root(cfg)
    source = {}
    for q in QUADRANTS:
        (path,) = root.glob(f"{q}-*.parquet")
        source[q] = pq.read_table(path, columns=["messages_json", "completion", "tools"])
    model, tok = load_truncated_decoder(model_path(cfg), max(cfg.layers), dtype=cfg.dtype, device=cfg.device)
    inner = model
    while hasattr(inner, "model"):
        inner = inner.model
    reader = ReadToken(inner.layers, cfg.layers)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def ids_of(r: int) -> list[int]:
        q, i, n = where[r]
        ids = render(tok, source[q]["messages_json"][i].as_py(), source[q]["completion"][i].as_py(), source[q]["tools"][i].as_py())
        if len(ids) != n:
            raise ValueError(f"row {r}: {len(ids)} tokens now, {n} at the render stage")
        return ids

    def forward(batch: list[int], hidden_states: bool = False):
        seqs = [ids_of(r) for r in batch]
        w = len(seqs[0])
        input_ids = torch.full((len(batch), w), pad, dtype=torch.long)
        attn = torch.zeros((len(batch), w), dtype=torch.long)
        for k, ids in enumerate(seqs):
            input_ids[k, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[k, :len(ids)] = 1
        reader.pos = torch.tensor([len(ids) - 2 for ids in seqs])   # the token before <|im_end|>
        with torch.no_grad():
            res = inner(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
                        output_hidden_states=hidden_states)
        return torch.stack([reader.out[L] for L in cfg.layers], dim=1), res   # (batch, layers, d)

    if pilot:   # the hook reads the same tensor as hidden_states[L + 1] (the activation contract's layer k)
        check = todo[-cfg.pilot_check_rows:]
        got, res = forward(check, hidden_states=True)
        for j, L in enumerate(cfg.layers):
            ref = res.hidden_states[L + 1][torch.arange(len(check)), reader.pos].to(torch.bfloat16).cpu()
            if not torch.equal(got[:, j], ref):
                raise ValueError(f"layer {L}: hook output differs from hidden_states[{L + 1}]")
        del res
        print(f"hook == hidden_states on {len(check)} rows, layers {cfg.layers}", flush=True)

    buf_rows, buf_acts, buf_tok, t0, n_tok, i = [], [], 0, time.time(), 0, 0
    while i < len(todo):
        w = where[todo[i]][2]
        batch = todo[i:i + min(cfg.acts_max_batch, max(1, cfg.acts_batch_tokens // w))]
        i += len(batch)
        acts, _ = forward(batch)
        buf_rows += batch
        buf_acts.append(acts)
        buf_tok += sum(where[r][2] for r in batch)
        if len(buf_rows) >= cfg.save_rows or buf_tok >= cfg.save_tokens or i == len(todo):
            torch.save({"row": torch.tensor(buf_rows), "acts": torch.cat(buf_acts)}, out_dir / f"chunk_{n_chunk:04d}.pt")
            n_chunk += 1
            n_tok, buf_rows, buf_acts, buf_tok = n_tok + buf_tok, [], [], 0
            dt = time.time() - t0
            print(f"shard {shard}: {i}/{len(todo)} rows, {n_tok:,} tokens, {n_tok / dt:,.0f} tok/s, "
                  f"{dt / 60:.1f} min (last batch {len(batch)} x {w})", flush=True)
    print(f"shard {shard}: finished", flush=True)


def stage_merge(cfg, pilot: bool):
    import torch
    d = acts_dir(pilot)
    rows = pq.read_table(d / "rows.parquet", columns=["row", "act_index", "dropped"])
    index = {r: a for r, a, dr in zip(rows["row"].to_pylist(), rows["act_index"].to_pylist(), rows["dropped"].to_pylist()) if not dr}
    chunks = [torch.load(p) for p in sorted(d.glob("shard*/chunk_*.pt"))]
    got = torch.cat([c["row"] for c in chunks]).tolist()
    if sorted(got) != sorted(index):
        raise ValueError(f"chunks hold {len(got)} rows ({len(set(got))} distinct), rows.parquet keeps {len(index)}")
    order = torch.argsort(torch.tensor([index[r] for r in got]))
    acts = torch.cat([c["acts"] for c in chunks])[order]             # (n_kept, layers, d)
    if not bool(torch.isfinite(acts.float()).all()):
        raise ValueError("non-finite activations")
    for j, L in enumerate(cfg.layers):
        torch.save(acts[:, j].clone(), d / f"layer_{L}.pt")
    print(f"wrote {len(cfg.layers)} files of shape {tuple(acts[:, 0].shape)} to {d}")


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    rank, n1 = rankdata(s), y.sum()
    return float((rank[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def stage_check(cfg, pilot: bool):
    import torch
    from probe_inference import load_probe_from_hub
    d = acts_dir(pilot)
    rows = pq.read_table(d / "rows.parquet")
    rows = rows.filter(pc.invert(rows["dropped"]))
    probe = load_probe_from_hub(cfg.probe_path, repo_id=cfg.probe_repo, revision=cfg.probe_revision)
    acts = {L: torch.load(d / f"layer_{L}.pt")[:, None] for L in probe.used_layers}       # (n, 1, d)
    n = rows.num_rows
    scores = torch.cat([probe.score_batch({L: a[i:i + 4096] for L, a in acts.items()},
                                          torch.ones(min(4096, n - i), 1, dtype=torch.bool))
                        for i in range(0, n, 4096)]).numpy()
    y = np.array(rows["followup_deceptive"].to_pylist()).astype(int)
    src = np.array(rows["source"].to_pylist())
    res = {"n": n, "probe": cfg.probe_path, "auroc": auroc(scores, y),
           "auroc_by_source": {s: auroc(scores[src == s], y[src == s]) for s in sorted(set(src))
                               if 0 < y[src == s].sum() < (src == s).sum()}}
    (d / "check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"published linear ensemble on {n} stored rows: AUROC {res['auroc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["render", "collect", "merge", "check"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.stage == "collect":
        stage_collect(cfg, args.pilot, args.shard)
    else:
        {"render": stage_render, "merge": stage_merge, "check": stage_check}[args.stage](cfg, args.pilot)


if __name__ == "__main__":
    main()
