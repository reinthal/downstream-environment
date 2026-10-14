"""The paper's per-layer probe ensembles (Appendix A of arXiv:2610.12445,
Table 2 hyperparameters), copied from
experiments/2026-10-13-fibs-basharena/train.py. Only linear and MLP: they read
one token per row, which is what the stored activations hold; EFC and axial
read the whole follow-up window and stay in that experiment.

Recipe:
  * one probe per layer on the token before the answer's end-of-turn token;
    the layers share one gradient clip and one checkpoint chosen on their mean
    validation loss; input scale = RMS norm of the layer's activations over
    ``scale_rows`` training rows ("unit_norm");
  * AdamW, constant (linear) / cosine (mlp) schedule after linear warm-up,
    up to ``max_epochs``, validation four times per epoch, early stop after
    an epoch without a validation improvement;
  * used layers (A.2): mean train loss over the last 5 steps below
    ``layer_loss_cut``;
  * loss: BCE on the logit per row and layer, y = 1 on falsehood completions;
    the ensemble score is the mean sigmoid over layers.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from probe_inference.archs.linear import LinearProbe
from probe_inference.archs.mlp import MLPProbe
from probe_inference.score import LAST_ANSWER_TOKEN

PER_LAYER = ("linear", "mlp")


def auroc(s: np.ndarray, y: np.ndarray) -> float:
    rank, n1 = rankdata(s), y.sum()
    return float((rank[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


class LayerEnsemble(torch.nn.Module):
    """One probe per layer, trained jointly (shared clip and checkpoint)."""

    def __init__(self, arch: str, n_layers: int, d_model: int):
        super().__init__()
        if arch == "linear":
            self.probes = torch.nn.ModuleList([LinearProbe(d_model, 1, "unit_norm") for _ in range(n_layers)])
        elif arch == "mlp":
            self.probes = torch.nn.ModuleList([MLPProbe(d_model, 256, 1, 0.0, "unit_norm", "relu", None)
                                               for _ in range(n_layers)])
        else:
            raise ValueError(arch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # x (B, L, d) -> logits (B, L)
        return torch.stack([p(x[:, j].unsqueeze(1))[:, 0, 0] for j, p in enumerate(self.probes)], dim=1)


def make_optimizer(model: torch.nn.Module, hp: dict, total_steps: int, warmup: int):
    if hp["optimizer"] != "adamw":
        raise ValueError(f"per-layer probes train with adamw, not {hp['optimizer']}")
    params = list(model.parameters())
    opt = torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": hp["weight_decay"]},
                             {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}], lr=hp["lr"])

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if hp["schedule"] == "constant":
            return 1.0
        if hp["schedule"] == "cosine":
            t = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
        raise ValueError(hp["schedule"])

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


def loss_of(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(L,) mean BCE per layer."""
    return F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits), reduction="none").mean(dim=0)


@torch.no_grad()
def evaluate(model, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict:
    model.eval()
    losses, scores = [], []
    for s in range(0, len(y), batch_size):
        logits = model(x[s:s + batch_size].float())
        losses.append(loss_of(logits, y[s:s + batch_size]) * len(logits))
        scores.append(torch.sigmoid(logits).mean(dim=1))
    model.train()
    loss = torch.stack(losses).sum(dim=0) / len(y)
    return {"loss": float(loss.mean()), "layer_loss": loss.tolist(),
            "auroc": auroc(torch.cat(scores).cpu().numpy(), y.cpu().numpy())}


def train_layer_ensemble(arch: str, layers: list[int], train_x: torch.Tensor, train_y: torch.Tensor,
                         val_x: torch.Tensor, val_y: torch.Tensor, hp: dict, *, batch_size: int,
                         warmup_steps: int, grad_clip: float, scale_rows: int, layer_loss_cut: float,
                         seed: int, log=print) -> tuple[LayerEnsemble, list[int], dict]:
    """Train on (N, L, d) activations already on the training device.
    Returns the model at the selected checkpoint, the used layers and a
    training record."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = train_x.device
    model = LayerEnsemble(arch, len(layers), train_x.shape[-1]).to(device)
    x0 = train_x[:scale_rows].float()
    for j, p in enumerate(model.probes):
        p.set_input_scale(float(x0[:, j].norm(dim=-1).pow(2).mean().sqrt()))   # RMS norm over rows

    n = len(train_y)
    steps_per_epoch = math.ceil(n / batch_size)
    total = steps_per_epoch * hp["max_epochs"]
    opt, sched = make_optimizer(model, hp, total, warmup_steps)
    val_every = max(1, steps_per_epoch // 4)
    log(f"{arch}: {n} train / {len(val_y)} val rows, {steps_per_epoch} steps per epoch, up to {total}")

    best = {"loss": float("inf"), "state": None}
    train_log, recent, step = [], [], 0
    for epoch in range(hp["max_epochs"]):
        improved = False
        order = list(range(n))
        rng.shuffle(order)
        for s in range(0, n, batch_size):
            idx = torch.tensor(order[s:s + batch_size], device=device)
            loss = loss_of(model(train_x[idx].float()), train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            step += 1
            recent = (recent + [loss.detach()])[-5:]
            if step % 50 == 0 or step == 1:
                train_log.append({"step": step, "loss": loss.detach().cpu().tolist()})
            if step % val_every == 0 or step == total:
                v = evaluate(model, val_x, val_y, 4096)
                log(f"  {arch} epoch {epoch + 1} step {step}/{total}: train {float(loss.detach().mean()):.4f} "
                    f"val loss {v['loss']:.4f} auroc {v['auroc']:.4f}")
                if v["loss"] < best["loss"]:
                    best = {"loss": v["loss"], "state": {k: t.detach().clone() for k, t in model.state_dict().items()},
                            "step": step, "epoch": epoch + 1, "tail": torch.stack(recent).mean(dim=0).tolist()}
                    improved = True
        if not improved:
            log(f"  {arch}: no validation improvement in epoch {epoch + 1}; stopping")
            break
    if best["state"] is None:
        raise RuntimeError(f"{arch}: no checkpoint was selected")
    model.load_state_dict(best["state"])

    # layer rule (A.2): mean train loss over the last 5 training batches at the selected step
    used = [L for L, l in zip(layers, best["tail"]) if l < layer_loss_cut]
    if not used:
        raise RuntimeError(f"{arch}: no layer below the {layer_loss_cut} training-loss cut ({best['tail']})")
    final = evaluate(model, val_x, val_y, 4096)
    log(f"{arch}: best step {best['step']} (epoch {best['epoch']}), val loss {final['loss']:.4f}, "
        f"val AUROC {final['auroc']:.4f}, used layers {used}")
    record = {"arch": arch, "n_train": n, "n_val": len(val_y), "best_step": best["step"], "best_epoch": best["epoch"],
              "tail_train_loss": best["tail"], "used_layers": used, "val": final, "train_log": train_log, "hparams": hp}
    return model, used, record


def save_probe(arch: str, model: LayerEnsemble, layers: list[int], used: list[int], out: Path, metadata: dict):
    """Write the ``probe_inference`` directory: probe_metadata.json plus
    layer_<L>/{config.json, model.pt}. ``metadata`` adds provenance (model,
    revision, trained_by, data) to the format's own keys."""
    out.mkdir(parents=True, exist_ok=True)
    meta = {**metadata, "architecture": arch, "layers": layers, "eval_sequence_aggregator": "mean",
            "obfuscate_over": LAST_ANSWER_TOKEN, "layer_rule": {"used_layers": used}}
    for j, L in enumerate(layers):
        d = out / f"layer_{L}"
        d.mkdir(exist_ok=True)
        p = model.probes[j]
        ia = {"d_model": p.d_model, "nhead": 1, "normalize_input": "unit_norm"}
        if arch == "mlp":
            ia.update(d_mlp=256, activation="relu", d_mlp2=None)
        (d / "config.json").write_text(json.dumps({"class_name": type(p).__name__, "init_args": ia}, indent=2))
        torch.save({k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
                    for k, v in p.state_dict().items()}, d / "model.pt")
    (out / "probe_metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
