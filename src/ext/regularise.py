"""R1: dropout x weight-decay grid on the 500-record training slice (extension beyond the Assignment 3 spec).

Question: can regularisation stop the FFNNs memorising the 500 training records, without new data?

    grid    dropout {0, 0.2, 0.5} x weight decay {0, 1e-5, 1e-4, 1e-3}, mean and concat, 1 hidden layer, 30 epochs
    pick    per model, the setting with the lowest clean validation loss (ties keep the setting listed first)
    then    the winner at 2 and 3 hidden layers; winners scored on all 73 noise conditions with the true loss

Everything else (data, optimizer, learning rate, batch, seed, selection rule) is as Assignment 3. Selecting on
the validation records and then reporting validation results is optimistic; the test split stays sealed.
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..geometry.embeddings import DERIVED_P3_DIR, RESULTS_P3_DIR, load_embedding_table, load_phase3_config, table_sha256
from ..neural.data import encode_texts
from ..neural.evaluate import to_tensors
from ..neural.model import NextTokenNet
from ..neural.run import build_output_space, set_determinism
from ..neural.train import resolve_device, train_model
from ..phase2_config import DERIVED_P2_DIR, load_phase2_config, sha256_file
from .conditions import load_all
from .scoring import add_true_loss, encode_conditions, score_neural, summarise

OUT = RESULTS_P3_DIR / "ext" / "reg"
CKPT = DERIVED_P3_DIR / "ext" / "reg"


def tag(model: str, n_layers: int, dropout: float, wd: float) -> str:
    return f"{model}_L{n_layers}_do{dropout:g}_wd{wd:g}"


def pick_best(grid: pd.DataFrame) -> pd.DataFrame:
    """Per model: lowest best-epoch validation loss; the grid is in listed order, so a tie keeps the simpler setting."""
    return grid.loc[grid.groupby("model", sort=False).best_val_loss.idxmin()].reset_index(drop=True)


def run(verbose: bool = True) -> dict:
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    p2, p3 = load_phase2_config(), load_phase3_config()
    cfgN, R = p3["neural"], p3["extensions"]["regularisation"]
    set_determinism(cfgN)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    device = resolve_device(R["device"])

    pipe, vocab, _, hf_ids = build_output_space(p2)
    K = vocab.size
    train_text = [json.loads(l)["text"] for l in open(DERIVED_P2_DIR / f"{R['train_set']}.jsonl", encoding="utf-8")]
    train_seq = encode_texts(train_text, pipe, vocab, hf_ids)
    assert train_seq.T == 35642, "training targets differ from Assignment 2/3"
    conds = load_all(p2, p3)
    seqs = encode_conditions(conds, pipe, vocab, hf_ids)
    E_np = load_embedding_table(p3); E_hash = table_sha256(E_np)
    E_t = torch.from_numpy(np.array(E_np))
    tr_t, va_t = to_tensors(train_seq), to_tensors(seqs["clean"])

    def fit(model, n, do, wd):
        cfg = {**cfgN, "n_layers": n, "dropout": do, "weight_decay": wd, "hidden_size": R["hidden_size"],
               "max_epochs": R["max_epochs"], "device": R["device"]}
        ts = time.time()
        out = train_model(model, E_t, K, tr_t, va_t, cfg)
        out["seconds"] = time.time() - ts
        return out

    # ---- the grid ---------------------------------------------------------------------------------------
    hist, grid, models = [], [], {}
    for model in ("mean", "concat"):
        for do, wd in product(R["dropout"], R["weight_decay"]):
            n = int(R["n_layers"])
            tr = fit(model, n, float(do), float(wd))
            t = tag(model, n, do, wd)
            h = pd.DataFrame(tr["history"]).assign(n_layers=n, dropout=do, weight_decay=wd, run=t)
            hist.append(h)
            b = h[h.selected].iloc[0]; last = h.iloc[-1]
            grid.append({"run": t, "model": model, "n_layers": n, "dropout": do, "weight_decay": wd,
                         "best_epoch": int(b.epoch), "best_val_loss": b.val_loss, "val_acc_at_best": b.val_accuracy,
                         "train_loss_at_best": b.train_loss, "gap_at_best": b.val_loss - b.train_loss,
                         "val_loss_at_30": last.val_loss, "train_loss_at_30": last.train_loss,
                         "val_rise_after_best": last.val_loss - b.val_loss, "seconds": round(tr["seconds"], 1)})
            models[t] = {k: v.cpu() for k, v in tr["model"].state_dict().items()}
            log(f"R1  {t:28} best epoch {int(b.epoch):2d}  val {b.val_loss:.4f}  rise to ep30 {last.val_loss - b.val_loss:+.3f}  ({tr['seconds']:.0f}s)")
    grid = pd.DataFrame(grid)
    best = pick_best(grid)

    # ---- the winners at more depth ----------------------------------------------------------------------
    for _, w in best.iterrows():
        for n in R["depth_check"]:
            tr = fit(w.model, int(n), float(w.dropout), float(w.weight_decay))
            t = tag(w.model, int(n), w.dropout, w.weight_decay)
            h = pd.DataFrame(tr["history"]).assign(n_layers=int(n), dropout=w.dropout, weight_decay=w.weight_decay, run=t)
            hist.append(h); models[t] = {k: v.cpu() for k, v in tr["model"].state_dict().items()}
            b = h[h.selected].iloc[0]; last = h.iloc[-1]
            grid = pd.concat([grid, pd.DataFrame([{"run": t, "model": w.model, "n_layers": int(n), "dropout": w.dropout,
                    "weight_decay": w.weight_decay, "best_epoch": int(b.epoch), "best_val_loss": b.val_loss,
                    "val_acc_at_best": b.val_accuracy, "train_loss_at_best": b.train_loss,
                    "gap_at_best": b.val_loss - b.train_loss, "val_loss_at_30": last.val_loss,
                    "train_loss_at_30": last.train_loss, "val_rise_after_best": last.val_loss - b.val_loss,
                    "seconds": round(tr["seconds"], 1)}])], ignore_index=True)
            log(f"R1  {t:28} best epoch {int(b.epoch):2d}  val {b.val_loss:.4f}  (depth check)")
    history = pd.concat(hist, ignore_index=True)

    # ---- score: unregularised baseline and winners (1/2/3 layers) on all 73 conditions -----------------------
    score_runs = [tag(m, 1, 0.0, 0.0) for m in ("mean", "concat")]
    score_runs += [tag(w.model, n, w.dropout, w.weight_decay) for _, w in best.iterrows() for n in [1] + list(R["depth_check"])]
    rows, ckpts = [], {}
    for t in dict.fromkeys(score_runs):
        g = grid[grid.run == t].iloc[0]
        log(f"R1  scoring {t} on {len(seqs)} conditions")
        net = NextTokenNet(E_t, g.model, K, R["hidden_size"], int(g.n_layers), float(g.dropout)).to(device)
        net.load_state_dict(models[t]); net.eval()
        rows.append(score_neural(net, seqs, cfgN["eval_chunk"], run=t, model=g.model, n_layers=int(g.n_layers),
                                 dropout=g.dropout, weight_decay=g.weight_decay,
                                 variant="no regularisation" if (g.dropout == 0 and g.weight_decay == 0) else "regularised"))
        ck = CKPT / f"{t}.pt"
        torch.save(models[t], ck)
        ckpts[t] = sha256_file(ck)
    scored = add_true_loss(pd.concat(rows, ignore_index=True), ["run"])
    summary = summarise(scored, ["run", "model", "n_layers", "dropout", "weight_decay", "operation", "p"],
                        ["H", "H_true", "dH", "dH_true", "accuracy"])
    if table_sha256(E_np) != E_hash or table_sha256(E_t.numpy()) != E_hash:
        raise RuntimeError("the frozen embedding table changed")

    grid.to_csv(OUT / "reg_grid.csv", index=False)
    history.to_csv(OUT / "reg_history.csv", index=False)
    best.to_csv(OUT / "reg_best.csv", index=False)
    scored.to_csv(OUT / "reg_condition_rows.csv", index=False)
    summary.to_csv(OUT / "reg_noise_summary.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    from . import figures
    figures.plot_reg_grid(grid, OUT / "figures" / "reg_grid.png")
    figures.plot_reg_curves(history, best, OUT / "figures" / "reg_curves.png")
    figures.plot_reg_noise(summary, best, OUT / "figures" / "reg_noise.png")
    info = {"label": p3["extensions"]["label"], "settings": R, "device": str(device), "runs": len(grid),
            "best": best[["model", "dropout", "weight_decay", "best_epoch", "best_val_loss"]].to_dict("records"),
            "checkpoints_sha256": ckpts, "E_sha256": E_hash, "torch": torch.__version__,
            "seconds": round(time.time() - t0, 1),
            "note": "selection on clean validation records and reporting on validation records is optimistic; test split sealed"}
    (OUT / "reg_run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log(f"done in {info['seconds']:.0f}s -> {OUT}")
    return {"grid": grid, "best": best, "history": history, "scored": scored, "summary": summary, "info": info}


def run_long(verbose: bool = True) -> dict:
    """R1b: the grid winners hit the 30-epoch limit, so re-train them (1 layer) for up to `long_check.max_epochs`
    epochs with early stopping, and score the result on all 73 conditions. Reads reg_best.csv from `run`."""
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    p2, p3 = load_phase2_config(), load_phase3_config()
    cfgN, R = p3["neural"], p3["extensions"]["regularisation"]
    L = R["long_check"]
    set_determinism(cfgN)
    best = pd.read_csv(OUT / "reg_best.csv")
    pipe, vocab, _, hf_ids = build_output_space(p2)
    K = vocab.size
    train_text = [json.loads(l)["text"] for l in open(DERIVED_P2_DIR / f"{R['train_set']}.jsonl", encoding="utf-8")]
    train_seq = encode_texts(train_text, pipe, vocab, hf_ids)
    seqs = encode_conditions(load_all(p2, p3), pipe, vocab, hf_ids)
    E_np = load_embedding_table(p3); E_hash = table_sha256(E_np); E_t = torch.from_numpy(np.array(E_np))
    hist, rows, summ_rows, ck = [], [], [], {}
    for _, w in best.iterrows():
        cfg = {**cfgN, "n_layers": 1, "dropout": float(w.dropout), "weight_decay": float(w.weight_decay),
               "hidden_size": R["hidden_size"], "max_epochs": int(L["max_epochs"]), "patience": int(L["patience"]),
               "device": R["device"]}
        ts = time.time()
        tr = train_model(w.model, E_t, K, to_tensors(train_seq), to_tensors(seqs["clean"]), cfg)
        t = tag(w.model, 1, w.dropout, w.weight_decay) + "_long"
        h = pd.DataFrame(tr["history"]).assign(run=t, dropout=w.dropout, weight_decay=w.weight_decay, n_layers=1)
        hist.append(h)
        b = h[h.selected].iloc[0]
        summ_rows.append({"run": t, "model": w.model, "dropout": w.dropout, "weight_decay": w.weight_decay,
                          "epochs_run": len(h), "best_epoch": int(b.epoch), "best_val_loss": b.val_loss,
                          "val_acc_at_best": b.val_accuracy, "train_loss_at_best": b.train_loss,
                          "best_val_loss_30_epoch_grid": w.best_val_loss, "seconds": round(time.time() - ts, 1)})
        log(f"R1b {t:34} ran {len(h)} epochs, best epoch {int(b.epoch)}  val {b.val_loss:.4f} (30-epoch grid: {w.best_val_loss:.4f})")
        rows.append(score_neural(tr["model"], seqs, cfgN["eval_chunk"], run=t, model=w.model, n_layers=1,
                                 dropout=w.dropout, weight_decay=w.weight_decay, variant="regularised, trained to convergence"))
        p = CKPT / f"{t}.pt"; torch.save({k: v.cpu() for k, v in tr["model"].state_dict().items()}, p); ck[t] = sha256_file(p)
    if table_sha256(E_np) != E_hash:
        raise RuntimeError("the frozen embedding table changed")
    scored = add_true_loss(pd.concat(rows, ignore_index=True), ["run"])
    summary = summarise(scored, ["run", "model", "n_layers", "dropout", "weight_decay", "operation", "p"],
                        ["H", "H_true", "dH", "dH_true", "accuracy"])
    long = pd.DataFrame(summ_rows)
    long.to_csv(OUT / "reg_long.csv", index=False)
    pd.concat(hist, ignore_index=True).to_csv(OUT / "reg_long_history.csv", index=False)
    scored.to_csv(OUT / "reg_long_condition_rows.csv", index=False)
    summary.to_csv(OUT / "reg_long_noise_summary.csv", index=False)
    info = json.loads((OUT / "reg_run_info.json").read_text())
    info["long_check"] = {"settings": L, "results": long.to_dict("records"), "checkpoints_sha256": ck,
                          "seconds": round(time.time() - t0, 1)}
    (OUT / "reg_run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log(f"done in {time.time() - t0:.0f}s")
    return {"long": long, "scored": scored, "summary": summary}
