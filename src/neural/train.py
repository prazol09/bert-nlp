"""N3: train one model on clean text and keep the epoch with the lowest clean-validation loss.

Assignment 3 uses the defaults (cpu, one hidden layer, no dropout, no weight decay, no early stop). The Phase 3
extensions pass `n_layers`, `dropout`, `weight_decay`, a longer `max_epochs`, an optional `patience` and
`device: mps` through the same function; nothing else about training changes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .evaluate import evaluate
from .model import NextTokenNet


def pick_best_epoch(val_losses: list[float]) -> int:
    """1-based epoch with the lowest loss; an exact tie keeps the EARLIER epoch."""
    best, best_i = float("inf"), 0
    for i, v in enumerate(val_losses, start=1):
        if v < best:                       # strict: equal loss never replaces the earlier epoch
            best, best_i = v, i
    return best_i


def resolve_device(name: str | None) -> torch.device:
    """'cpu' (Assignment 3), 'mps' (Apple GPU) or 'cuda'; falls back to cpu when the GPU is unavailable."""
    name = (name or "cpu").lower()
    if name == "mps" and not torch.backends.mps.is_available():
        name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def train_model(mode: str, E: torch.Tensor, n_outputs: int, train, val, cfg: dict, log=None) -> dict:
    """`train`, `val` are (ctx, tgt) tensor pairs. Returns best weights, history and the selected epoch."""
    device = resolve_device(cfg.get("device"))
    torch.manual_seed(cfg["seed"])                         # identical initialisation recipe for both models
    model = NextTokenNet(E, mode, n_outputs, cfg["hidden_size"], int(cfg.get("n_layers", 1)),
                         float(cfg.get("dropout", 0.0))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(cfg["seed"])       # same shuffles for both models (drawn on cpu)
    ctx, tgt = (t.to(device) for t in train)
    N, bs = len(tgt), cfg["batch_size"]
    untrained = evaluate(model, *val, chunk=cfg["eval_chunk"])
    history, best_state, best_loss = [], None, float("inf")
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(N, generator=gen).to(device)
        total = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(ctx[idx]), tgt[idx])      # raw logits in, no softmax first
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        v = evaluate(model, *val, chunk=cfg["eval_chunk"])          # no parameter updates
        history.append({"model": mode, "epoch": epoch, "train_loss": total / N, "val_loss": v["H"],
                        "val_accuracy": v["accuracy"]})
        if v["H"] < best_loss:              # same rule as pick_best_epoch: strict, so a tie keeps the earlier epoch
            best_loss = v["H"]
            best_state = {k: p.detach().to("cpu", copy=True) for k, p in model.state_dict().items()}
        if log:
            log(f"  {mode:6} layers={model.n_layers} epoch {epoch:2d}  train {total / N:.4f}  val {v['H']:.4f}  "
                f"acc {v['accuracy']:.4f}")
        patience = cfg.get("patience")       # extension: stop after this many epochs without a new best (None = never)
        if patience and epoch - pick_best_epoch([h["val_loss"] for h in history]) >= int(patience):
            break
    best = pick_best_epoch([h["val_loss"] for h in history])
    for h in history:
        h["selected"] = h["epoch"] == best
    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "history": history, "best_epoch": best, "device": str(device),
            "untrained_val_loss": untrained["H"], "untrained_val_accuracy": untrained["accuracy"]}
