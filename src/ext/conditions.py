"""E1: the extended noise grid, 1 clean + 3 operations x 8 rates x 3 seeds = 73 conditions.

The four Assignment 2 rates (0.01-0.20) are READ from the saved files in derived/phase2/noisy/ and never
rewritten. The new rates (0.30-0.90) are generated with exactly the same function (`corrupt_record`), the
same a2_default rule, the same seeds and the same per-record RNG key, and saved to derived/phase3/noisy_ext/.
`verify_generator` proves that the function still reproduces a saved Assignment 2 file before anything new
is written.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import ROOT
from ..geometry.drift import OP_ORDER, OP_SHORT, parse_condition
from ..noise.grid import condition_id, corrupt_record, load_condition
from ..phase2_config import DERIVED_P2_DIR, NOISY_DIR, sha256_file


def ext_cfg(p3: dict) -> dict:
    return p3["extensions"]


def ext_rates(p3: dict) -> list[float]:
    return [float(p) for p in ext_cfg(p3)["noise"]["rates"]]


def new_rates(p2, p3) -> list[float]:
    return [p for p in ext_rates(p3) if p not in p2.rates]


def new_dir(p3: dict) -> Path:
    return ROOT / ext_cfg(p3)["noise"]["new_dir"]


def ext_condition_ids(p2, p3) -> list[str]:
    """Clean first, then operation -> rate -> seed (same order as Assignment 2)."""
    return ["clean"] + [f"{OP_SHORT[op]}-p{p:.2f}-s{s}" for op in p2.operations for p in ext_rates(p3) for s in p2.seeds]


def condition_path(cid: str, p2, p3) -> Path:
    _, p, _ = parse_condition(cid)
    return NOISY_DIR / f"{cid}.jsonl" if (cid == "clean" or p in p2.rates) else new_dir(p3) / f"{cid}.jsonl"


def load_ext_condition(cid: str, p2, p3) -> list[dict]:
    path = condition_path(cid, p2, p3)
    return load_condition(cid, path.parent)


def load_all(p2, p3) -> dict[str, list[dict]]:
    return {c: load_ext_condition(c, p2, p3) for c in ext_condition_ids(p2, p3)}


def validation_records() -> list[dict]:
    return [json.loads(l) for l in open(DERIVED_P2_DIR / "val_sample.jsonl", encoding="utf-8")]


def verify_generator(p2, records: list[dict], cid: str = "sub-p0.20-s47") -> dict:
    """Regenerate one saved Assignment 2 condition in memory and compare it row by row with the file."""
    op, p, seed = parse_condition(cid)
    saved = load_condition(cid)
    fresh = [corrupt_record(r, op, p, seed, p2) for r in records]
    same = all(a == b for a, b in zip(saved, fresh)) and len(saved) == len(fresh)
    return {"condition": cid, "records": len(saved), "identical": same}


def generate_new(p2, p3, records: list[dict] | None = None, overwrite: bool = False, log=None) -> pd.DataFrame:
    """Write the condition files for the new rates (skips files that already exist unless overwrite)."""
    records = records or validation_records()
    check = verify_generator(p2, records)
    if not check["identical"]:
        raise RuntimeError(f"the noise generator no longer reproduces {check['condition']}; refusing to extend")
    out = new_dir(p3); out.mkdir(parents=True, exist_ok=True)
    index = []
    for op in p2.operations:
        for p in new_rates(p2, p3):
            for seed in p2.seeds:
                cid = condition_id(op, p, seed)
                path = out / f"{cid}.jsonl"
                if overwrite or not path.exists():
                    rows = [corrupt_record(r, op, p, seed, p2) for r in records]
                    with open(path, "w", encoding="utf-8") as fh:
                        for row in rows:
                            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if log:
                        log(f"  wrote {path.name}")
                index.append({"condition_id": cid, "operation": op, "p": p, "seed": seed,
                              "file": str(path.relative_to(ROOT)), "sha256": sha256_file(path)})
    idx = pd.DataFrame(index)
    idx.to_csv(out / "index.csv", index=False)
    return idx


def noise_stats(conds: dict[str, list[dict]]) -> pd.DataFrame:
    """Realized corruption per condition. Corpus rates pool counts first (sum / sum)."""
    rows = []
    for cid, recs in conds.items():
        op, p, seed = parse_condition(cid)
        med = sum(r["med"] for r in recs); chars = sum(r["clean_char_count"] for r in recs)
        rows.append({"condition_id": cid, "operation": op, "p": p, "seed": -1 if seed is None else seed,
                     "records": len(recs), "eligible_letters": sum(r["eligible_count"] for r in recs),
                     "realized_edits": sum(r["realized_edits"] for r in recs), "med": med, "clean_chars": chars,
                     "normalized_med": med / chars, "empty_records": sum(r["empty_after_noise"] for r in recs)})
    df = pd.DataFrame(rows)
    df["realized_rate"] = df.realized_edits / df.eligible_letters
    return df


def order_ops(df: pd.DataFrame, col: str = "operation") -> pd.DataFrame:
    order = ["clean"] + OP_ORDER
    return df.assign(**{col: pd.Categorical(df[col], order, ordered=True)})
