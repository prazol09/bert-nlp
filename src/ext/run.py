"""E1-E3 as one reproducible pipeline; `phase3_extensions.py` (repo root) is the command-line wrapper.

    E1  generate the new noise rates (0.30-0.90), verify the generator against a saved A2 file
    E2  geometry drift, n-gram control and the two REQUIRED Assignment 3 checkpoints on all 73 conditions,
        with the true loss H_true = S / T_clean
    E3  1 / 2 / 3 hidden layers x mean / concat, 30 epochs, width 64; each best epoch scored on all 73 conditions

Assignment 2 / Assignment 3 artifacts are only read. Outputs: results/phase3/ext/, derived/phase3/ext/.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..geometry.drift import audit_table, build_pairs, seed_summary
from ..geometry.embeddings import DERIVED_P3_DIR, RESULTS_P3_DIR, load_embedding_table, load_phase3_config, table_sha256
from ..geometry.pooling import load_tokenizer
from ..geometry.spans import select_spans
from ..neural.model import NextTokenNet
from ..neural.run import build_output_space, set_determinism
from ..neural.train import resolve_device, train_model
from ..neural.data import encode_texts
from ..neural.evaluate import to_tensors
from ..phase2_config import DERIVED_P2_DIR, load_phase2_config, sha256_file
from . import figures
from .conditions import (condition_path, ext_cfg, ext_condition_ids, generate_new, load_all, noise_stats,
                         validation_records, verify_generator)
from .scoring import (add_true_loss, encode_conditions, ngram_engine, operation_ranking, score_neural, score_ngram,
                      summarise)

RESULTS_EXT = RESULTS_P3_DIR / "ext"
DERIVED_EXT = DERIVED_P3_DIR / "ext"
METRICS = ["H", "H_true", "PP", "accuracy", "dH", "dH_true"]


def _load_net(E_t, mode, K, hidden, path, device):
    net = NextTokenNet(E_t, mode, K, hidden).to(device)
    net.load_state_dict(torch.load(path, map_location=device)); net.eval()
    return net


def run(out_dir: Path | str = RESULTS_EXT, verbose: bool = True, sweep: bool = True) -> dict:
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    p2, p3 = load_phase2_config(), load_phase3_config()
    cfgN, X = p3["neural"], ext_cfg(p3)
    set_determinism(cfgN)
    out_dir = Path(out_dir); (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    DERIVED_EXT.mkdir(parents=True, exist_ok=True)

    # ---- E1: noise -----------------------------------------------------------------------------------
    records = validation_records()
    log("E1  verifying the noise generator against a saved Assignment 2 file, then writing new rates")
    gen_check = verify_generator(p2, records)
    index = generate_new(p2, p3, records, log=log)
    cids = ext_condition_ids(p2, p3)
    conds = load_all(p2, p3)
    nstats = noise_stats(conds)
    nstats.to_csv(out_dir / "noise_realized.csv", index=False)
    log(f"    {len(cids)} conditions ({len(index)} new files)")

    # ---- E2a: geometry -------------------------------------------------------------------------------
    log("E2  geometry drift on all conditions")
    E_np = load_embedding_table(p3); E_hash = table_sha256(E_np)
    tok = load_tokenizer(p3["embedding"]["checkpoint"], p3["embedding"]["revision"])
    spans = select_spans(records, p3["geometry"]["span_cap"], p2.protected_spans)
    pairs = build_pairs(spans, records, p2, tok, E_np, conditions=cids, path_for=lambda c: condition_path(c, p2, p3))
    g_audit = audit_table(pairs); g_sum = seed_summary(g_audit)
    g_audit.to_csv(out_dir / "geometry_audit.csv", index=False)
    g_sum.to_csv(out_dir / "geometry_summary.csv", index=False)
    req_audit = pd.read_csv(RESULTS_P3_DIR / "static_span_audit.csv")       # parity with the required 37 rows
    m = req_audit.merge(g_audit, on="condition_id", suffixes=("_req", "_ext"))
    geo_parity = bool(len(m) == 37 and np.allclose(m.mean_cosine_req, m.mean_cosine_ext, rtol=0, atol=1e-12, equal_nan=True))

    # ---- E2b: neural (required checkpoints) and n-gram control -------------------------------------------
    pipe, vocab, _, hf_ids = build_output_space(p2)
    K = vocab.size
    seqs = encode_conditions(conds, pipe, vocab, hf_ids)
    device = resolve_device(X["sweep"]["device"])
    E_t = torch.from_numpy(np.array(E_np))
    log(f"E2  scoring the two required checkpoints on {device} and the n-gram control on cpu")
    req = []
    for mode in ("mean", "concat"):
        net = _load_net(E_t, mode, K, cfgN["hidden_size"], DERIVED_P3_DIR / "neural" / f"{mode}_best.pt", device)
        req.append(score_neural(net, seqs, cfgN["eval_chunk"], model=mode, n_layers=1, variant="A3 required"))
    eng, lam = ngram_engine(p2, pipe)
    ngram = add_true_loss(score_ngram(eng, lam, conds), ["model"])
    neural = add_true_loss(pd.concat(req, ignore_index=True), ["model"])
    rows = pd.concat([neural, ngram], ignore_index=True)
    rows.to_csv(out_dir / "condition_rows.csv", index=False)
    summary = summarise(rows, ["model", "operation", "p"], ["H", "H_true", "PP", "dH", "dH_true", "accuracy"])
    summary.to_csv(out_dir / "noise_summary.csv", index=False)

    saved = pd.read_csv(RESULTS_P3_DIR / "neural_condition_rows.csv")    # parity with the required 74 rows
    pm = saved.merge(neural, on=["model", "condition_id"], suffixes=("_req", "_ext"))
    tol = 1e-9 if device.type == "cpu" else 1e-4
    neural_parity = {"rows": len(pm), "max_abs_dH": float((pm.H_req - pm.H_ext).abs().max()),
                     "T_identical": bool((pm.T_req == pm.T_ext).all()), "tolerance": tol}
    neural_parity["ok"] = bool(len(pm) == 74 and neural_parity["T_identical"] and neural_parity["max_abs_dH"] < tol)
    ctrl = pd.read_csv(RESULTS_P3_DIR / "ngram_control.csv").merge(ngram, on="condition_id", suffixes=("_req", "_ext"))
    ngram_parity = bool(len(ctrl) == 37 and (ctrl.T_req == ctrl.T_ext).all() and np.allclose(ctrl.H_req, ctrl.H_ext, rtol=0, atol=1e-9))
    if not (neural_parity["ok"] and ngram_parity and geo_parity and gen_check["identical"]):
        raise RuntimeError(f"parity with the required results failed: neural={neural_parity} ngram={ngram_parity} geometry={geo_parity}")

    # which operation hurts most: per model and p, on every metric
    nz = summary[summary.operation != "clean"]
    gs = g_sum.rename(columns={"rate": "p"}).assign(model="geometry")
    ranking = pd.concat([
        operation_ranking(nz, "H_true_mean", "high", ["model", "p"]),
        operation_ranking(nz, "H_mean", "high", ["model", "p"]),
        operation_ranking(nz[nz.model != "ngram"], "accuracy_mean", "low", ["model", "p"]),
        operation_ranking(gs, "mean_cosine_mean", "low", ["model", "p"]),
        operation_ranking(gs, "mean_distance_mean", "high", ["model", "p"]),
    ], ignore_index=True)
    ranking.to_csv(out_dir / "operation_ranking.csv", index=False)

    # ---- E3: hidden layers x epochs ---------------------------------------------------------------------
    sweep_rows, history = pd.DataFrame(), pd.DataFrame()
    sweep_info = {}
    if sweep:
        S = X["sweep"]
        train_text = [json.loads(l)["text"] for l in open(DERIVED_P2_DIR / "train_a1_slice.jsonl", encoding="utf-8")]
        train_seq = encode_texts(train_text, pipe, vocab, hf_ids)
        assert train_seq.T == 35642, "training targets differ from Assignment 2/3"
        hist, srows = [], []
        for n in S["n_layers"]:
            for mode in ("mean", "concat"):
                cfg = {**cfgN, "n_layers": int(n), "hidden_size": int(S["hidden_size"]),
                       "max_epochs": int(S["max_epochs"]), "device": S["device"]}
                ts = time.time()
                log(f"E3  training {mode}, {n} hidden layer(s), {cfg['max_epochs']} epochs on {device}")
                tr = train_model(mode, E_t, K, to_tensors(train_seq), to_tensors(seqs["clean"]), cfg)
                secs = time.time() - ts
                for h in tr["history"]:
                    hist.append({**h, "n_layers": n})
                ck = DERIVED_EXT / f"{mode}_L{n}_best.pt"
                torch.save({k: v.cpu() for k, v in tr["model"].state_dict().items()}, ck)
                sweep_info[f"{mode}_L{n}"] = {"best_epoch": tr["best_epoch"], "train_seconds": round(secs, 1),
                                              "checkpoint": str(ck.relative_to(DERIVED_P3_DIR.parent.parent)),
                                              "checkpoint_sha256": sha256_file(ck),
                                              "untrained_val_loss": tr["untrained_val_loss"], **tr["model"].counts()}
                log(f"    best epoch {tr['best_epoch']}  val {min(h['val_loss'] for h in tr['history']):.4f}  ({secs:.0f}s)")
                srows.append(score_neural(tr["model"], seqs, cfgN["eval_chunk"], model=mode, n_layers=n,
                                          variant=f"{n} layer(s), best of {cfg['max_epochs']} epochs"))
        history = pd.DataFrame(hist)
        sweep_rows = add_true_loss(pd.concat(srows, ignore_index=True), ["model", "n_layers"])
        history.to_csv(out_dir / "sweep_history.csv", index=False)
        sweep_rows.to_csv(out_dir / "sweep_condition_rows.csv", index=False)
        sweep_summary = summarise(sweep_rows, ["model", "n_layers", "operation", "p"], ["H", "H_true", "dH", "dH_true", "accuracy"])
        sweep_summary.to_csv(out_dir / "sweep_noise_summary.csv", index=False)
        if table_sha256(E_t.numpy()) != E_hash or table_sha256(E_np) != E_hash:
            raise RuntimeError("the frozen embedding table changed during the sweep")

    # ---- figures and run info ---------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    figures.plot_noise_ext(summary, out_dir / "figures" / "ext_noise_loss.png")
    figures.plot_operation_vs_med(summary, g_sum, nstats, out_dir / "figures" / "ext_operation_vs_med.png")
    figures.plot_geometry_ext(g_sum, out_dir / "figures" / "ext_geometry_drift.png")
    if sweep:
        figures.plot_sweep_training(history, out_dir / "figures" / "ext_sweep_training.png")
        figures.plot_sweep_noise(pd.read_csv(out_dir / "sweep_noise_summary.csv"), out_dir / "figures" / "ext_sweep_noise.png")
    info = {"label": X["label"], "conditions": len(cids), "new_condition_files": len(index),
            "generator_check": gen_check, "parity": {"neural_required_74": neural_parity, "ngram_control_37": ngram_parity,
                                                     "geometry_audit_37": geo_parity},
            "device": str(device), "true_loss": X["true_loss"]["definition"], "T_clean": int(neural.T_clean.iloc[0]),
            "E_sha256": E_hash, "sweep": sweep_info, "sweep_settings": X["sweep"],
            "python": platform.python_version(), "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
            "note": "GPU (mps): scoring matches cpu within 1e-7 nats; training repeats exactly run to run on this machine but follows a slightly different numerical path from cpu (1-layer validation loss within 0.008 nats of the cpu run over 5 epochs, same best epoch)"}
    (out_dir / "ext_run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log(f"done in {info['seconds']:.0f}s -> {out_dir}")
    return {"rows": rows, "summary": summary, "ranking": ranking, "geometry_audit": g_audit, "geometry_summary": g_sum,
            "noise": nstats, "history": history, "sweep_rows": sweep_rows, "info": info}

