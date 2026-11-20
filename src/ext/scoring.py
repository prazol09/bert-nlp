"""E2: score every extended condition with a neural model or the n-gram control, and add the TRUE loss.

Two losses per condition, both from the same summed loss S (nats over every target, incl. one EOS per record):

    H      = S / T          loss per NOISY target (the Assignment 3 number; T grows with noise)
    H_true = S / T_clean    loss per CLEAN target: the same denominator (35,971) for every condition

Noise splits words into more pieces, so T grows (36k clean -> 70k at 20% insertion). H divides the extra cost
by the extra pieces and understates the damage; H_true charges it to the same clean text every time, so
conditions are compared like-for-like. For the clean condition H_true == H. Accuracy stays per noisy target.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from ..geometry.drift import OP_ORDER, parse_condition
from ..lm.score import build_engine, q_for_records, record_losses
from ..neural.data import encode_texts
from ..neural.evaluate import evaluate, to_tensors
from ..phase2_config import DERIVED_P2_DIR


def encode_conditions(conds: dict[str, list[dict]], pipe, vocab, hf_ids) -> dict:
    return {c: encode_texts([r["noisy_text"] for r in rows], pipe, vocab, hf_ids) for c, rows in conds.items()}


def add_true_loss(rows: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Add T_clean, H_true, and the change of H / H_true from each model's own clean row."""
    clean = rows[rows.condition_id == "clean"][group + ["T", "H"]].rename(columns={"T": "T_clean", "H": "H_clean"})
    out = rows.merge(clean, on=group, how="left", validate="many_to_one")
    out["H_true"] = out.S / out.T_clean
    out["dH"] = out.H - out.H_clean
    out["dH_true"] = out.H_true - out.H_clean
    return out.drop(columns="H_clean")


def score_neural(net, seqs: dict, chunk: int, **labels) -> pd.DataFrame:
    rows = []
    for cid, seq in seqs.items():
        op, p, seed = parse_condition(cid)
        r = evaluate(net, *to_tensors(seq), chunk=chunk)
        rows.append({**labels, "condition_id": cid, "operation": op, "p": p, "seed": -1 if seed is None else seed,
                     "T": r["T"], "S": r["S"], "H": r["H"], "PP": r["PP"], "accuracy": r["accuracy"],
                     **{k: v for k, v in seq.diagnostics().items() if k != "T"}})
    return pd.DataFrame(rows)


# ---- the n-gram control on any condition -----------------------------------------------------------
def ngram_engine(p2, pipe):
    """The Assignment 2 WordPiece engine on train_a1_slice, with its frozen interpolation weights."""
    train = [json.loads(l) for l in open(DERIVED_P2_DIR / "train_a1_slice.jsonl", encoding="utf-8")]
    eng = build_engine("train_a1_slice", train, pipe, p2)
    lam = tuple(p2.frozen_lambdas["train_a1_slice"]["wordpiece"])
    return eng, lam


def score_ngram(eng, lam, conds: dict[str, list[dict]]) -> pd.DataFrame:
    before = eng.counts.snapshot()
    rows = []
    for cid, recs in conds.items():
        op, p, seed = parse_condition(cid)
        toks = eng.pipeline.tokenize_batch([r["noisy_text"] for r in recs])
        S = T = 0
        for q in q_for_records(eng, toks):
            _, nats, t = record_losses(q, "p_mix", lam)
            S += nats; T += t
        rows.append({"model": "ngram", "n_layers": 0, "condition_id": cid, "operation": op, "p": p,
                     "seed": -1 if seed is None else seed, "T": T, "S": S, "H": S / T, "PP": math.exp(S / T)})
    assert eng.counts.snapshot() == before, "scoring changed the n-gram counts"
    return pd.DataFrame(rows)


# ---- summaries -------------------------------------------------------------------------------------
def summarise(rows: pd.DataFrame, group: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean / min / max over the 3 corruption seeds per group; the clean row is kept as its own group (p = 0)."""
    g = rows.groupby(group, sort=False, dropna=False)       # keep groups whose key is empty (e.g. the n-gram has no dropout)
    out = g.size().rename("n_seeds").reset_index()
    for m in metrics:
        agg = g[m].agg(["mean", "min", "max"]).reset_index()
        out[f"{m}_mean"], out[f"{m}_min"], out[f"{m}_max"] = agg["mean"], agg["min"], agg["max"]
    out["operation"] = pd.Categorical(out.operation, ["clean"] + OP_ORDER, ordered=True)
    return out.sort_values(group).reset_index(drop=True).astype({"operation": str})


def operation_ranking(summary: pd.DataFrame, metric: str, worse: str, by: list[str]) -> pd.DataFrame:
    """For each group (e.g. model x p) order the three operations from most to least harmful.

    worse = "high" when a larger value means more damage (loss, distance), "low" when smaller does (cosine, accuracy).
    """
    noisy = summary[summary.operation != "clean"]
    rows = []
    for k, g in noisy.groupby(by, sort=False):
        g = g.sort_values(metric, ascending=(worse == "low"))
        k = k if isinstance(k, tuple) else (k,)
        vals = dict(zip(g.operation, g[metric]))
        rows.append({**dict(zip(by, k)), "metric": metric, "most harmful": g.operation.iloc[0],
                     "order (most > least)": " > ".join(g.operation.str[:3]),
                     **{op: vals.get(op, np.nan) for op in OP_ORDER}})
    return pd.DataFrame(rows)
