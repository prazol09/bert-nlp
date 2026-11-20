"""R2: more training data, a 10K-record chunk of the training split (extension beyond the Assignment 3 spec).

Question: does more data fix what regularisation alone could not, and does the n-gram keep its lead under heavy noise
when it gets the same data?

    data      the first 10,000 train_full records in file order, extended to the end of the article in progress
              (it contains all 500 train_a1_slice records); validation records and noise files unchanged
    networks  per model: no regularisation, the R1 winner, and the R1 winner's dropout with weaker weight decay;
              1 hidden layer of 64, up to 30 epochs, stop 3 epochs after the last improvement
    control   the Assignment 2 WordPiece n-gram rebuilt on the same chunk, interpolation weights re-selected on clean
              validation with the Assignment 2 procedure (never written into configs/phase2.yaml)
    scored    every network run and the n-gram on all 73 conditions, with the true loss
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from ..geometry.embeddings import DERIVED_P3_DIR, RESULTS_P3_DIR, load_embedding_table, load_phase3_config, table_sha256
from ..lm.interp import select_lambdas
from ..lm.score import build_engine, q_for_records
from ..neural.data import encode_texts
from ..neural.evaluate import to_tensors
from ..neural.model import NextTokenNet
from ..neural.run import build_output_space, set_determinism
from ..neural.train import resolve_device, train_model
from ..phase2_config import DERIVED_P2_DIR, load_phase2_config, sha256_file
from .conditions import load_all
from .scoring import add_true_loss, encode_conditions, score_neural, score_ngram, summarise

OUT = RESULTS_P3_DIR / "ext" / "scale"
CKPT = DERIVED_P3_DIR / "ext" / "scale"


def chunk_records(source: str, n: int) -> list[dict]:
    """First n records in file order, extended so the last article is complete."""
    recs = [json.loads(l) for l in open(DERIVED_P2_DIR / f"{source}.jsonl", encoding="utf-8")]
    k = n
    while k < len(recs) and recs[k]["article_id"] == recs[n - 1]["article_id"]:
        k += 1
    return recs[:k]


def summarise_rows(allrows: pd.DataFrame) -> pd.DataFrame:
    """Seed summary per run (networks and the n-gram) with the true loss and, for networks, accuracy."""
    summary = summarise(allrows, ["run", "model", "operation", "p"], ["H", "H_true", "dH", "dH_true"])
    settings = allrows.groupby("run", sort=False)[["dropout", "weight_decay"]].first().reset_index()
    neural = allrows[allrows.model != "ngram"]
    acc = summarise(neural, ["run", "operation", "p"], ["accuracy"])[["run", "operation", "p", "accuracy_mean", "accuracy_min", "accuracy_max"]]
    summary = summary.merge(settings, on="run", how="left").merge(acc, on=["run", "operation", "p"], how="left")
    assert set(summary.run) == set(allrows.run), "a run is missing from the summary"
    return summary


def rebuild(verbose: bool = True) -> pd.DataFrame:
    """Recompute the summary and figures from the saved per-condition rows and histories (no training)."""
    rows = pd.read_csv(OUT / "scale_condition_rows.csv")
    summary = summarise_rows(rows)
    summary.to_csv(OUT / "scale_noise_summary.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    from . import figures
    figures.plot_scale(pd.read_csv(OUT / "scale_history.csv"), summary, pd.read_csv(OUT / "scale_best.csv"), OUT / "figures")
    if verbose:
        print(f"rebuilt {OUT / 'scale_noise_summary.csv'} ({summary.run.nunique()} runs) and figures")
    return summary


def run(verbose: bool = True) -> dict:
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    p2, p3 = load_phase2_config(), load_phase3_config()
    cfgN, D = p3["neural"], p3["extensions"]["data_scale"]
    set_determinism(cfgN)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
    device = resolve_device(D["device"])

    # ---- data ------------------------------------------------------------------------------------------
    chunk = chunk_records(D["source"], int(D["n_records"]))
    ids = [r["record_id"] for r in chunk]
    slice_ids = {json.loads(l)["record_id"] for l in open(DERIVED_P2_DIR / "train_a1_slice.jsonl", encoding="utf-8")}
    manifest = {"source": D["source"], "requested": int(D["n_records"]), "records": len(chunk),
                "articles": len({r["article_id"] for r in chunk}), "first": ids[0], "last": ids[-1],
                "contains_slice_records": len(slice_ids & set(ids)),
                "record_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    (OUT / "chunk_manifest.json").write_text(json.dumps({**manifest, "record_ids": ids}, indent=1), encoding="utf-8")
    pipe, vocab, _, hf_ids = build_output_space(p2)
    K = vocab.size
    train_seq = encode_texts([r["text"] for r in chunk], pipe, vocab, hf_ids)
    conds = load_all(p2, p3)
    seqs = encode_conditions(conds, pipe, vocab, hf_ids)
    manifest["train_targets"] = train_seq.T
    manifest["distinct_outputs_seen"] = int(len(np.unique(train_seq.tgt)))
    log(f"R2  chunk: {len(chunk):,} records, {manifest['articles']} articles, {train_seq.T:,} targets "
        f"(contains {manifest['contains_slice_records']} of the 500 slice records)")

    # ---- n-gram control on the same chunk ---------------------------------------------------------------
    eng = build_engine(f"train_chunk_{D['n_records']}", chunk, pipe, p2)
    clean_tokens = pipe.tokenize_batch([r["noisy_text"] for r in conds["clean"]])
    lam, lam_table = select_lambdas(np.vstack(q_for_records(eng, clean_tokens)), float(p2.lm["lambda_step"]),
                                    float(p2.lm["lambda_tie_tolerance"]))
    pd.DataFrame(lam_table).to_csv(OUT / "ngram_lambda_selection.csv", index=False)
    log(f"R2  n-gram on the chunk: interpolation weights re-selected on clean validation = {lam}")
    ngram = add_true_loss(score_ngram(eng, lam, conds).assign(run="ngram"), ["model"])

    # ---- networks ---------------------------------------------------------------------------------------
    E_np = load_embedding_table(p3); E_hash = table_sha256(E_np); E_t = torch.from_numpy(np.array(E_np))
    tr_t, va_t = to_tensors(train_seq), to_tensors(seqs["clean"])
    hist, runs, rows, ck = [], [], [], {}
    for model, settings in D["settings"].items():
        for st in settings:
            do, wd = float(st["dropout"]), float(st["weight_decay"])
            run_id = f"{model}_L1_do{do:g}_wd{wd:g}"
            cfg = {**cfgN, "n_layers": int(D["n_layers"]), "dropout": do, "weight_decay": wd,
                   "hidden_size": int(D["hidden_size"]), "max_epochs": int(D["max_epochs"]),
                   "patience": int(D["patience"]), "device": D["device"]}
            ts = time.time()
            tr = train_model(model, E_t, K, tr_t, va_t, cfg)
            secs = time.time() - ts
            h = pd.DataFrame(tr["history"]).assign(run=run_id, dropout=do, weight_decay=wd)
            hist.append(h)
            b = h[h.selected].iloc[0]
            runs.append({"run": run_id, "model": model, "dropout": do, "weight_decay": wd, "epochs_run": len(h),
                         "best_epoch": int(b.epoch), "best_val_loss": b.val_loss, "val_acc_at_best": b.val_accuracy,
                         "train_loss_at_best": b.train_loss, "seconds": round(secs, 1)})
            log(f"R2  {run_id:26} ran {len(h):2d} epochs, best {int(b.epoch):2d}  val {b.val_loss:.4f}  acc {b.val_accuracy:.4f}  ({secs:.0f}s)")
            rows.append(score_neural(tr["model"], seqs, cfgN["eval_chunk"], run=run_id, model=model, n_layers=1,
                                     dropout=do, weight_decay=wd))
            p = CKPT / f"{run_id}.pt"; torch.save({k: v.cpu() for k, v in tr["model"].state_dict().items()}, p)
            ck[run_id] = sha256_file(p)
            del tr
    if table_sha256(E_np) != E_hash or table_sha256(E_t.numpy()) != E_hash:
        raise RuntimeError("the frozen embedding table changed")

    runs = pd.DataFrame(runs)
    best = runs.loc[runs.groupby("model", sort=False).best_val_loss.idxmin()].reset_index(drop=True)
    neural = add_true_loss(pd.concat(rows, ignore_index=True), ["run"])
    allrows = pd.concat([neural, ngram], ignore_index=True)
    summary = summarise_rows(allrows)

    runs.to_csv(OUT / "scale_runs.csv", index=False)
    best.to_csv(OUT / "scale_best.csv", index=False)
    pd.concat(hist, ignore_index=True).to_csv(OUT / "scale_history.csv", index=False)
    allrows.to_csv(OUT / "scale_condition_rows.csv", index=False)
    summary.to_csv(OUT / "scale_noise_summary.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    from . import figures
    figures.plot_scale(pd.concat(hist, ignore_index=True), summary, best, OUT / "figures")
    info = {"label": p3["extensions"]["label"], "settings": D, "device": str(device), "chunk": manifest,
            "ngram_lambdas": list(lam), "best": best.to_dict("records"), "checkpoints_sha256": ck,
            "E_sha256": E_hash, "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
            "note": "settings chosen on clean validation and reported on validation (optimistic); test split sealed"}
    (OUT / "scale_run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log(f"done in {info['seconds']:.0f}s -> {OUT}")
    return {"runs": runs, "best": best, "summary": summary, "rows": allrows, "info": info}


def run_depth(verbose: bool = True) -> dict:
    """R2b: each model's best 10K-chunk setting (from scale_best.csv) re-trained at 2 and 3 hidden layers, scored on all
    73 conditions. The 1-layer result is the R2 run itself (same settings and seed), so it is reused, not re-trained."""
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    p2, p3 = load_phase2_config(), load_phase3_config()
    cfgN, D = p3["neural"], p3["extensions"]["data_scale"]
    set_determinism(cfgN)
    best = pd.read_csv(OUT / "scale_best.csv")
    chunk = chunk_records(D["source"], int(D["n_records"]))
    pipe, vocab, _, hf_ids = build_output_space(p2)
    K = vocab.size
    train_seq = encode_texts([r["text"] for r in chunk], pipe, vocab, hf_ids)
    seqs = encode_conditions(load_all(p2, p3), pipe, vocab, hf_ids)
    E_np = load_embedding_table(p3); E_hash = table_sha256(E_np); E_t = torch.from_numpy(np.array(E_np))
    tr_t, va_t = to_tensors(train_seq), to_tensors(seqs["clean"])
    hist, runs, rows, ck = [], [], [], {}
    for _, w in best.iterrows():
        for n in D["depth_check"]:
            do, wd = float(w.dropout), float(w.weight_decay)
            run_id = f"{w.model}_L{n}_do{do:g}_wd{wd:g}"
            cfg = {**cfgN, "n_layers": int(n), "dropout": do, "weight_decay": wd, "hidden_size": int(D["hidden_size"]),
                   "max_epochs": int(D["max_epochs"]), "patience": int(D["patience"]), "device": D["device"]}
            ts = time.time()
            tr = train_model(w.model, E_t, K, tr_t, va_t, cfg)
            secs = time.time() - ts
            h = pd.DataFrame(tr["history"]).assign(run=run_id, n_layers=int(n), dropout=do, weight_decay=wd)
            hist.append(h)
            b = h[h.selected].iloc[0]
            runs.append({"run": run_id, "model": w.model, "n_layers": int(n), "dropout": do, "weight_decay": wd,
                         "epochs_run": len(h), "best_epoch": int(b.epoch), "best_val_loss": b.val_loss,
                         "val_acc_at_best": b.val_accuracy, "train_loss_at_best": b.train_loss, "seconds": round(secs, 1),
                         **tr["model"].counts()})
            log(f"R2b {run_id:26} ran {len(h):2d} epochs, best {int(b.epoch):2d}  val {b.val_loss:.4f}  acc {b.val_accuracy:.4f}  ({secs:.0f}s)")
            rows.append(score_neural(tr["model"], seqs, cfgN["eval_chunk"], run=run_id, model=w.model, n_layers=int(n),
                                     dropout=do, weight_decay=wd))
            pth = CKPT / f"{run_id}.pt"; torch.save({k: v.cpu() for k, v in tr["model"].state_dict().items()}, pth)
            ck[run_id] = sha256_file(pth)
            del tr
    if table_sha256(E_np) != E_hash or table_sha256(E_t.numpy()) != E_hash:
        raise RuntimeError("the frozen embedding table changed")
    # add the reused 1-layer runs so the outputs hold all three depths
    r1 = pd.read_csv(OUT / "scale_runs.csv").merge(best[["run"]], on="run").assign(n_layers=1)
    r1["trainable"] = [NextTokenNet(E_t, m, K, int(D["hidden_size"])).counts()["trainable"] for m in r1.model]
    r1["frozen"] = int(E_t.numel())
    runs = pd.concat([r1, pd.DataFrame(runs)], ignore_index=True).sort_values(["model", "n_layers"]).reset_index(drop=True)
    old_rows = pd.read_csv(OUT / "scale_condition_rows.csv")
    neural = add_true_loss(pd.concat(rows, ignore_index=True), ["run"])
    l1 = old_rows[old_rows.run.isin(best.run)].assign(n_layers=1)
    allrows = pd.concat([l1, neural], ignore_index=True)
    summary = summarise(allrows, ["run", "model", "n_layers", "operation", "p"], ["H", "H_true", "dH", "dH_true", "accuracy"])
    h1 = pd.read_csv(OUT / "scale_history.csv").merge(best[["run"]], on="run").assign(n_layers=1)
    history = pd.concat([h1, *hist], ignore_index=True)
    runs.to_csv(OUT / "scale_depth_runs.csv", index=False)
    history.to_csv(OUT / "scale_depth_history.csv", index=False)
    allrows.to_csv(OUT / "scale_depth_condition_rows.csv", index=False)
    summary.to_csv(OUT / "scale_depth_noise_summary.csv", index=False)
    info = json.loads((OUT / "scale_run_info.json").read_text())
    info["depth_check"] = {"depths": D["depth_check"], "runs": runs.to_dict("records"), "checkpoints_sha256": ck,
                           "seconds": round(time.time() - t0, 1),
                           "note": "1-layer rows are the R2 runs with the same settings (reused, not re-trained)"}
    (OUT / "scale_run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log(f"done in {time.time() - t0:.0f}s")
    return {"runs": runs, "summary": summary, "history": history}
