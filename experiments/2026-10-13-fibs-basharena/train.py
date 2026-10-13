"""Train the paper's four probe architectures on stored fibs window
activations (Appendix A of arXiv:2610.12445, Table 2 hyperparameters) and
save them in the probe_inference directory format, so the vendored loader and
scorer are the format check and the scoring path.

Recipe (common.py ``train_hparams``, Table 2 / A.6 / card):
  * linear, mlp: one probe per layer on the token before the answer's
    end-of-turn token; the six share one gradient clip (1.0) and one
    checkpoint chosen on their mean validation loss; input scale = RMS norm
    of the layer's activations over 512 training rows ("unit_norm");
    AdamW 5e-3, constant (linear) / cosine (mlp), batch 32, 20 warm-up steps,
    up to 6 epochs, early stop after one epoch without a validation
    improvement. Used layers: mean train loss over the last 5 logged steps
    below 0.6 (A.2).
  * efc, axial: one module over the whole read window; Muon 5e-3 cosine for
    matrix parameters (AdamW for the rest), no clipping, batch 32, up to 4
    epochs, same early stop. axial: centred unit norm per layer (mean and
    RMS over 512 training rows), logit at the last window token; efc: no
    input normalisation, pooled logit.
  * loss: BCE on the single logit per row, y = 1 on falsehood completions;
    seed 0 for data order and initialisation; probe forward under bf16
    autocast, parameters in float32.

Runs inside the Modal app (``modal_app.py::Trainer``); nothing here reads
the subject model.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from probe_inference.archs.axial import AxialProbe
from probe_inference.archs.efc import EFCProbe
from probe_inference.archs.linear import LinearProbe
from probe_inference.archs.mlp import MLPProbe
from probe_inference.score import LAST_ANSWER_TOKEN, LAST_USER_AND_ANSWER, aggregate, build_read_mask

PER_LAYER = ("linear", "mlp")
WINDOW = ("efc", "axial")


# ───────────────────────────── data ─────────────────────────────

def load_rows(fibs_dir: Path) -> list[dict]:
    with open(fibs_dir / "rows.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_acts(fibs_dir: Path, rows: list[dict], arch: str, layers: list[int], log=print):
    """Per split: for linear/mlp an (N, L, d) tensor of the read token; for
    efc/axial a list of (L, W, d) window tensors. Labels alongside."""
    by_i = {r["i"]: r for r in rows}
    data = {"train": [], "val": []}
    labels = {"train": [], "val": []}
    shards = sorted(fibs_dir.glob("acts_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not shards:
        raise FileNotFoundError(f"no activation shards in {fibs_dir}")
    for k, path in enumerate(shards):
        blob = torch.load(path, weights_only=True)
        for n, i in enumerate(blob["i"]):
            r = by_i[i]
            if arch in PER_LAYER:
                x = blob["token"][n]                  # (L, d): the token before the answer's end-of-turn token
            else:
                x = blob["acts"][n]                   # (L, W, d), W <= window_cap; every window token is read
            data[r["split"]].append(x)
            labels[r["split"]].append(r["label"])
        log(f"  loaded shard {k + 1}/{len(shards)}")
    if arch in PER_LAYER:
        data = {s: torch.stack(v) for s, v in data.items()}
    labels = {s: torch.tensor(v, dtype=torch.float32) for s, v in labels.items()}
    return data, labels


def batches(n: int, batch_size: int, rng: random.Random | None):
    order = list(range(n))
    if rng is not None:
        rng.shuffle(order)
    for s in range(0, n, batch_size):
        yield order[s:s + batch_size]


def pad_windows(items: list[torch.Tensor], device) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, L, T, d) zero-padded stack and its (B, T) valid mask."""
    T = max(x.shape[1] for x in items)
    L, d = items[0].shape[0], items[0].shape[2]
    out = torch.zeros((len(items), L, T, d), dtype=items[0].dtype)
    mask = torch.zeros((len(items), T), dtype=torch.bool)
    for b, x in enumerate(items):
        out[b, :, :x.shape[1]] = x
        mask[b, :x.shape[1]] = True
    return out.to(device), mask.to(device)


# ───────────────────────────── models ─────────────────────────────

class LayerEnsemble(torch.nn.Module):
    """Six per-layer probes trained jointly (shared clip and checkpoint)."""

    def __init__(self, arch: str, layers: list[int], d_model: int):
        super().__init__()
        self.layers = layers
        if arch == "linear":
            self.probes = torch.nn.ModuleList([LinearProbe(d_model, 1, "unit_norm") for _ in layers])
        else:
            self.probes = torch.nn.ModuleList([MLPProbe(d_model, 256, 1, 0.0, "unit_norm", "relu", None) for _ in layers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # x (B, L, d) -> logits (B, L)
        return torch.stack([p(x[:, j].unsqueeze(1))[:, 0, 0] for j, p in enumerate(self.probes)], dim=1)


def init_args(arch: str, layers: list[int], d_model: int) -> dict:
    keys = [str(L) for L in layers]
    if arch == "efc":
        return dict(num_layers=len(layers), d_model=d_model, feature_mode="combined", center=True, d_hidden=64,
                    d_probe=256, normalization="rms", normalization_eps=1e-6, shrinkage="fixed", shrinkage_alpha=0.1,
                    spectral_transform="eigh", newton_schulz_iterations=10, jitter=1e-5, feature_chunk_size=4,
                    normalize_input="none", layer_keys=keys)
    if arch == "axial":
        return dict(d_model=d_model, num_layers=len(layers), d_proj=256, n_attn_heads=4, n_blocks=4, d_ff=1024,
                    dropout=0.0, use_layer_embed=True, pool_mode="cls", use_checkpoint=True,
                    normalize_input="centered_unit_norm", proj_adapter_rank=0, input_adapter_rank=0,
                    sliding_window=None, layer_keys=keys)
    raise ValueError(arch)


def build_window_probe(arch: str, layers: list[int], d_model: int):
    args = init_args(arch, layers, d_model)
    return (EFCProbe if arch == "efc" else AxialProbe)(**args), args


# ───────────────────────────── optimisers ─────────────────────────────

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalise a matrix (Jordan et al. 2024, the quintic iteration)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Momentum + Newton-Schulz orthogonalised updates for 2-D weights."""

    def __init__(self, params, lr: float, momentum: float = 0.95, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(group["momentum"]).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=group["momentum"])
                else:
                    g = buf
                g = zeropower_via_newtonschulz5(g.reshape(g.shape[0], -1)).reshape(p.shape).to(p.dtype)
                g = g * max(1.0, p.size(0) / p.size(1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(g, alpha=-group["lr"])


def make_optimizers(model: torch.nn.Module, hp: dict, total_steps: int, warmup: int):
    params = list(model.named_parameters())
    if hp["optimizer"] == "adamw":
        decay = [p for n, p in params if p.ndim >= 2]
        no_decay = [p for n, p in params if p.ndim < 2]
        opts = [torch.optim.AdamW([{"params": decay, "weight_decay": hp["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=hp["lr"])]
    elif hp["optimizer"] == "muon":
        # matrices go to Muon; biases, norm gains, embeddings, the cls vector and the single-row heads to AdamW
        muon, adam = [], []
        for n, p in params:
            if p.ndim == 2 and p.shape[0] > 1 and p.shape[1] > 1 and not any(k in n for k in ("embed", "cls")):
                muon.append(p)
            else:
                adam.append(p)
        opts = [Muon(muon, lr=hp["lr"], weight_decay=hp["weight_decay"]),
                torch.optim.AdamW(adam, lr=hp["lr"], weight_decay=0.0)]
    else:
        raise ValueError(hp["optimizer"])

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if hp["schedule"] == "constant":
            return 1.0
        if hp["schedule"] == "cosine":
            t = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
        raise ValueError(hp["schedule"])

    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_at) for o in opts]
    return opts, scheds


# ───────────────────────────── training ─────────────────────────────

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def set_scales(arch: str, model, train_data, layers: list[int], n_rows: int, device):
    """Input normalisation buffers from the first ``n_rows`` training rows."""
    if arch in PER_LAYER:
        x = train_data[:n_rows].float()                              # (n, L, d)
        for j, p in enumerate(model.probes):
            p.set_input_scale(float(x[:, j].norm(dim=-1).pow(2).mean().sqrt()))   # RMS norm over rows
    elif arch == "axial":
        toks = torch.cat([x.float().transpose(0, 1) for x in train_data[:n_rows]], dim=0)   # (T_total, L, d)
        mean = toks.mean(dim=0)                                      # (L, d)
        centred = toks - mean
        scale = centred.norm(dim=-1).pow(2).mean(dim=0).sqrt()       # (L,)
        model.set_input_mean(mean.reshape(1, len(layers), 1, -1).to(device))
        model.set_input_scale(scale.reshape(1, len(layers), 1, 1).to(device))
    # efc: normalize_input "none"


def forward_logits(arch: str, model, items, device) -> torch.Tensor:
    if arch in PER_LAYER:
        return model(items.to(device).float())                       # (B, L)
    x, mask = pad_windows(items, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        values = model(x.float(), padding_mask=mask)                 # (B, T, 1)
    return aggregate(values.float(), mask, "last" if arch == "axial" else "mean")   # (B,)


def loss_of(arch: str, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if arch in PER_LAYER:
        return F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits), reduction="none").mean(dim=0)  # (L,)
    return F.binary_cross_entropy_with_logits(logits, y, reduction="mean")


@torch.no_grad()
def evaluate(arch: str, model, data, labels, batch_size: int, device) -> dict:
    model.eval()
    losses, scores = [], []
    for idx in batches(len(labels), batch_size, None):
        items = data[idx] if arch in PER_LAYER else [data[i] for i in idx]
        y = labels[idx].to(device)
        logits = forward_logits(arch, model, items, device)
        losses.append(loss_of(arch, logits, y).detach().cpu() * len(idx))
        scores.append((torch.sigmoid(logits).mean(dim=1) if arch in PER_LAYER else logits).detach().cpu())
    model.train()
    loss = torch.stack(losses).sum(dim=0) / len(labels)
    s = torch.cat(scores).numpy()
    out = {"loss": float(loss.mean()), "auroc": auroc(s, labels.numpy())}
    if arch in PER_LAYER:
        out["layer_loss"] = loss.tolist()
    return out


def train_arch(arch: str, fibs_dir: Path, out_dir: Path, cfg: dict, log=print) -> dict:
    """``cfg``: the experiment config as a dict (train_hparams, layers, seeds, ...)."""
    torch.manual_seed(cfg["train_seed"])
    rng = random.Random(cfg["train_seed"])
    device = torch.device("cuda")
    layers = cfg["probe_layers"]
    hp = cfg["train_hparams"][arch]
    rows = load_rows(fibs_dir)
    rows = [r for r in rows if not r["dropped"]]
    data, labels = load_acts(fibs_dir, rows, arch, layers, log)
    n_train, n_val = len(labels["train"]), len(labels["val"])
    d_model = (data["train"].shape[-1] if arch in PER_LAYER else data["train"][0].shape[-1])
    log(f"{arch}: {n_train} train / {n_val} val rows, d_model {d_model}, "
        f"positives {int(labels['train'].sum())} / {int(labels['val'].sum())}")

    if arch in PER_LAYER:
        model = LayerEnsemble(arch, layers, d_model).to(device)
        args = None
    else:
        model, args = build_window_probe(arch, layers, d_model)
        model = model.to(device)
    set_scales(arch, model, data["train"], layers, cfg["scale_rows"], device)

    bs = cfg["train_batch_size"]
    steps_per_epoch = math.ceil(n_train / bs)
    total = steps_per_epoch * hp["max_epochs"]
    opts, scheds = make_optimizers(model, hp, total, cfg["warmup_steps"])
    val_every = max(1, steps_per_epoch // 4)

    best = {"loss": float("inf"), "state": None, "step": 0, "epoch": 0, "val": None}
    train_log, recent = [], []
    step, stop = 0, False
    for epoch in range(hp["max_epochs"]):
        improved = False
        for idx in batches(n_train, bs, rng):
            items = data["train"][idx] if arch in PER_LAYER else [data["train"][i] for i in idx]
            y = labels["train"][idx].to(device)
            logits = forward_logits(arch, model, items, device)
            loss = loss_of(arch, logits, y)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.mean().backward()
            if hp["optimizer"] == "adamw":
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_adamw"])
            for o, s in zip(opts, scheds):
                o.step()
                s.step()
            step += 1
            recent.append(loss.detach().cpu())
            if step % 50 == 0 or step == 1:
                train_log.append({"step": step, "loss": (loss.detach().cpu().tolist() if arch in PER_LAYER
                                                          else float(loss))})
            if step % val_every == 0 or step == total:
                v = evaluate(arch, model, data["val"], labels["val"], bs, device)
                log(f"  {arch} epoch {epoch + 1} step {step}/{total}: train {float(loss.mean()):.4f} "
                    f"val loss {v['loss']:.4f} auroc {v['auroc']:.4f}")
                if v["loss"] < best["loss"]:
                    best = {"loss": v["loss"], "state": {k: t.detach().cpu().clone() for k, t in model.state_dict().items()},
                            "step": step, "epoch": epoch + 1, "val": v}
                    improved = True
        if not improved:
            log(f"  {arch}: no validation improvement in epoch {epoch + 1}; stopping")
            stop = True
        if stop:
            break
    if best["state"] is None:
        raise RuntimeError(f"{arch}: no checkpoint was selected")
    model.load_state_dict(best["state"])

    # layer rule (A.2): mean train loss over the last 5 logged training batches at the selected step
    used = layers
    if arch in PER_LAYER:
        tail = torch.stack(recent[-5:]).mean(dim=0).tolist()
        used = [L for L, l in zip(layers, tail) if l < cfg["layer_loss_cut"]]
        if not used:
            raise RuntimeError(f"{arch}: no layer below the {cfg['layer_loss_cut']} training-loss cut ({tail})")

    save_probe(arch, model, args, layers, used, out_dir / arch, cfg)
    final = evaluate(arch, model, data["val"], labels["val"], bs, device)
    result = {"arch": arch, "n_train": n_train, "n_val": n_val, "best_step": best["step"],
              "best_epoch": best["epoch"], "val": final, "used_layers": used, "train_log": train_log,
              "hparams": hp}
    (out_dir / f"{arch}_train.json").write_text(json.dumps(result, indent=1) + "\n")
    log(f"{arch}: best step {best['step']} (epoch {best['epoch']}), val loss {final['loss']:.4f}, "
        f"val AUROC {final['auroc']:.4f}, used layers {used}")
    return result


def save_probe(arch: str, model, args: dict | None, layers: list[int], used: list[int], out: Path, cfg: dict):
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": cfg["model_id"], "model_revision": cfg["model_revision"], "architecture": arch,
            "layers": layers, "eval_sequence_aggregator": "last" if arch == "axial" else "mean",
            "obfuscate_over": LAST_ANSWER_TOKEN if arch in PER_LAYER else LAST_USER_AND_ANSWER,
            "trained_by": "experiments/2026-10-13-fibs-basharena/train.py",
            "fibs": {"repo": cfg["fibs_repo"], "revision": cfg["fibs_revision"]}}
    if arch in PER_LAYER:
        meta["layer_rule"] = {"used_layers": used}
        for j, L in enumerate(layers):
            d = out / f"layer_{L}"
            d.mkdir(exist_ok=True)
            p = model.probes[j]
            if arch == "linear":
                ia = {"d_model": p.d_model, "nhead": 1, "normalize_input": "unit_norm"}
            else:
                ia = {"d_model": p.d_model, "d_mlp": 256, "nhead": 1, "normalize_input": "unit_norm",
                      "activation": "relu", "d_mlp2": None}
            (d / "config.json").write_text(json.dumps({"class_name": type(p).__name__, "init_args": ia}, indent=2))
            torch.save({k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
                        for k, v in p.state_dict().items()}, d / "model.pt")
    else:
        (out / "config.json").write_text(json.dumps({"class_name": type(model).__name__, "init_args": args}, indent=2))
        torch.save({k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
                    for k, v in model.state_dict().items()}, out / "model.pt")
    (out / "probe_metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
