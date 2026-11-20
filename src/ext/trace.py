"""E4: follow ONE sentence of your choice through the whole pipeline.

    sentence --noise--> noisy text --MED--> edit distance --WordPiece--> pieces
             --frozen E, mean-pool per word--> cosine / distance per word
             --mean / concat FFNN and n-gram--> per-token loss, top-3 guesses, H, true loss, accuracy

Three kinds of noisy variant, all optional:
    controlled  one edit of each operation on the middle letter of `focus_word` (e.g. cow -> coow / cw / cpw),
                so the three operations are compared on exactly the same position
    random      the real a2_default noise function at each rate, per-sentence RNG key, fixed seed
    custom      any noisy strings you type yourself, e.g. {"my typo": "A cwo jumps ovr the fence"}

This is an illustration, not evidence: the models were trained on 500 WikiText records, so a sentence like
"A cow jumps over the fence" is far from what they saw, and their guesses are mostly frequent tokens.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ..geometry.embeddings import DERIVED_P3_DIR, load_embedding_table, load_phase3_config
from ..geometry.metrics import pair_metrics
from ..geometry.pooling import pool_span
from ..geometry.spans import noisy_span, whitespace_spans
from ..med.fast import med_fast
from ..neural.data import BOS, encode_texts
from ..neural.model import NextTokenNet
from ..neural.run import build_output_space
from ..noise.rng import example_rng
from ..noise.rule_v2 import apply_v2, eligible_positions, render_v2
from ..phase2_config import load_phase2_config
from .scoring import ngram_engine

OPS = ("insertion", "deletion", "substitution")


@dataclass
class TraceContext:
    """Everything loaded once (tokenizer, frozen E, the two required checkpoints, the n-gram engine)."""
    p2: object
    p3: dict
    pipe: object
    vocab: object
    hf_ids: dict
    E: np.ndarray
    nets: dict
    eng: object
    lam: tuple
    inv_bert: dict = field(default_factory=dict)
    _ngram_cache: dict = field(default_factory=dict)


def load_context(checkpoints: dict | None = None) -> TraceContext:
    """checkpoints: {"mean": path, "concat": path}; default = the two required Assignment 3 models."""
    p2, p3 = load_phase2_config(), load_phase3_config()
    pipe, vocab, _, hf_ids = build_output_space(p2)
    E = load_embedding_table(p3)
    E_t = torch.from_numpy(np.array(E))
    checkpoints = checkpoints or {m: DERIVED_P3_DIR / "neural" / f"{m}_best.pt" for m in ("mean", "concat")}
    nets = {}
    for m, path in checkpoints.items():
        sd = torch.load(path, map_location="cpu")
        n_layers = 1 + sum(1 for k in sd if k.startswith("extra.") and k.endswith(".weight"))
        net = NextTokenNet(E_t, m.split("_")[0], vocab.size, p3["neural"]["hidden_size"], n_layers)
        net.load_state_dict(sd); net.eval(); nets[m] = net
    eng, lam = ngram_engine(p2, pipe)
    return TraceContext(p2, p3, pipe, vocab, hf_ids, E, nets, eng, lam, {v: k for k, v in hf_ids.items()})


# ---- variants --------------------------------------------------------------------------------------
def _controlled(sentence: str, focus_word: str, sub_char: str | None, protected) -> list[dict]:
    """One edit per operation on the middle eligible letter of the first occurrence of focus_word."""
    elig = set(eligible_positions(sentence, protected))
    for s, e in whitespace_spans(sentence):
        if sentence[s:e].lower().strip(".,;:!?\"'") == focus_word.lower():
            letters = [i for i in range(s, e) if i in elig]
            break
    else:
        raise ValueError(f"focus word {focus_word!r} not found in the sentence")
    i = letters[len(letters) // 2]
    ch = sentence[i]
    sub = sub_char or chr((ord(ch.lower()) - ord("a") + 1) % 26 + ord("a"))     # next letter: o -> p
    logs = {"insertion": [{"op": "insertion", "index": i, "source_char": None, "replacement_char": ch.lower()}],
            "deletion": [{"op": "deletion", "index": i, "source_char": ch, "replacement_char": None}],
            "substitution": [{"op": "substitution", "index": i, "source_char": ch, "replacement_char": sub}]}
    return [{"variant": f"controlled {op} on '{focus_word}'", "kind": "controlled", "operation": op, "p": None,
             "text": render_v2(sentence, log), "log": log} for op, log in logs.items()]


def make_variants(sentence: str, rates=(0.1, 0.3, 0.5, 0.9), seed: int = 11, focus_word: str | None = None,
                  custom: dict | None = None, ops=OPS, sub_char: str | None = None, p2=None) -> list[dict]:
    p2 = p2 or load_phase2_config()
    out = [{"variant": "clean", "kind": "clean", "operation": None, "p": 0.0, "text": sentence, "log": []}]
    if focus_word:
        out += _controlled(sentence, focus_word, sub_char, p2.protected_spans)
    for op in ops:
        for p in rates:
            rng = example_rng(seed, f"trace|{sentence}|{op}|{float(p)}")
            text, log, _ = apply_v2(sentence, op, float(p), p2.alphabet, rng, p2.protected_spans)
            out.append({"variant": f"{op} p={p:g}", "kind": "random", "operation": op, "p": float(p),
                        "text": text, "log": log})
    for name, text in (custom or {}).items():
        out.append({"variant": name, "kind": "custom", "operation": None, "p": None, "text": text, "log": None})
    return out


# ---- per-variant measurements ---------------------------------------------------------------------
def _ngram_topk(ctx: TraceContext, a: int, b: int, k: int) -> list[tuple[str, float]]:
    """Top-k of the n-gram mixture for history (a, b) (output-vocabulary ids; start symbol allowed)."""
    key = (a, b)
    if key not in ctx._ngram_cache:
        c, V, B = ctx.eng.counts, ctx.vocab.size, ctx.eng.counts.B
        l1, l2, l3 = ctx.lam
        p = np.zeros(V)
        if l1:
            q1 = np.ones(V)
            for w, n in c.joint[1].items():
                q1[w] += n
            p += l1 * q1 / (c.hist[1].get(0, 0) + V)
        if l2:
            q2 = np.ones(V)
            for jk, n in c.joint[2].items():
                if jk // B == b:
                    q2[jk % B] += n
            p += l2 * q2 / (c.hist[2].get(b, 0) + V)
        if l3:
            q3 = np.ones(V)
            ab = a * B + b
            for jk, n in c.joint[3].items():
                if jk // B == ab:
                    q3[jk % B] += n
            p += l3 * q3 / (c.hist[3].get(ab, 0) + V)
        top = np.argsort(-p)[:k]
        ctx._ngram_cache[key] = [(ctx.vocab.tokens[i], float(p[i])) for i in top]
    return ctx._ngram_cache[key]


def _tokens_table(ctx: TraceContext, text: str, k: int = 3) -> tuple[pd.DataFrame, dict]:
    seq = encode_texts([text], ctx.pipe, ctx.vocab, ctx.hf_ids)
    pieces = ctx.pipe.tokenize_batch([text])[0]
    name = lambda i: "BOS" if i == BOS else ctx.inv_bert[int(i)]
    rows = [{"context": f"{name(a)} {name(b)}", "target": ctx.vocab.tokens[t]} for (a, b), t in zip(seq.ctx, seq.tgt)]
    totals = {"pieces": len(pieces), "T": seq.T}
    with torch.no_grad():
        for m, net in ctx.nets.items():
            logp = torch.log_softmax(net(torch.from_numpy(seq.ctx)), dim=1)
            lt = -logp[torch.arange(seq.T), torch.from_numpy(seq.tgt)]
            top = torch.topk(logp, k, dim=1)
            for r, l, ti, tv in zip(rows, lt.tolist(), top.indices.tolist(), top.values.exp().tolist()):
                r[f"{m} loss"] = l
                r[f"{m} top-{k}"] = ", ".join(f"{ctx.vocab.tokens[i]} {v:.3f}" for i, v in zip(ti, tv))
            totals[f"{m} S"] = float(lt.double().sum())
            totals[f"{m} acc"] = float((logp.argmax(1) == torch.from_numpy(seq.tgt)).double().mean())
    ids = ctx.vocab.map_tokens(pieces)
    q = ctx.eng.lm.q_arrays(ids)
    pn = ctx.lam[0] * q[:, 0] + ctx.lam[1] * q[:, 1] + ctx.lam[2] * q[:, 2]
    f = ctx.eng.counts.frame(ids)
    for t, (r, pr) in enumerate(zip(rows, pn)):
        r["ngram loss"] = float(-np.log(pr))
        r[f"ngram top-{k}"] = ", ".join(f"{w} {v:.3f}" for w, v in _ngram_topk(ctx, f[t], f[t + 1], k))
    totals["ngram S"] = float(-np.log(pn).sum())
    return pd.DataFrame(rows), totals


def _word_drift(ctx: TraceContext, clean: str, v: dict) -> pd.DataFrame:
    tok = ctx.pipe.hf
    spans = whitespace_spans(clean)
    if v["log"] is not None:
        noisy_words = [noisy_span(clean, v["log"], s, e) for s, e in spans]
    else:                                                       # custom text: align by whitespace if possible
        nw = v["text"].split()
        noisy_words = nw if len(nw) == len(spans) else [None] * len(spans)
    rows = []
    for (s, e), nw in zip(spans, noisy_words):
        cw = clean[s:e]
        a = pool_span(cw, tok, ctx.E)
        b = pool_span(nw, tok, ctx.E) if nw is not None else None
        cos, dist, ok = pair_metrics(a.vector, b.vector if b is not None else None)
        rows.append({"variant": v["variant"], "clean word": cw, "noisy word": "∅" if nw == "" else nw,
                     "clean pieces": " ".join(a.pieces), "noisy pieces": " ".join(b.pieces) if b else "",
                     "cosine": cos if ok else np.nan, "distance": dist if ok else np.nan,
                     "note": "" if ok else ("fully deleted" if nw == "" else "not aligned")})
    return pd.DataFrame(rows)


def trace(sentence: str, ctx: TraceContext | None = None, rates=(0.1, 0.3, 0.5, 0.9), seed: int = 11,
          focus_word: str | None = None, custom: dict | None = None, top_k: int = 3, sub_char: str | None = None) -> dict:
    """Returns {"summary", "words", "tokens"} DataFrames; tokens is one table per variant."""
    ctx = ctx or load_context()
    variants = make_variants(sentence, rates, seed, focus_word, custom, sub_char=sub_char, p2=ctx.p2)
    tokens, summ, words = {}, [], []
    T_clean = None
    for v in variants:
        tt, tot = _tokens_table(ctx, v["text"], top_k)
        tokens[v["variant"]] = tt
        if v["kind"] == "clean":
            T_clean = tot["T"]
        d = med_fast(sentence, v["text"])
        row = {"variant": v["variant"], "text": v["text"], "edits": len(v["log"]) if v["log"] is not None else None,
               "MED": d, "MED per char": d / max(1, len(sentence)), "pieces": tot["pieces"], "T": tot["T"]}
        for m in list(ctx.nets) + ["ngram"]:
            S = tot[f"{m} S"]
            row[f"{m} H"] = S / tot["T"]
            row[f"{m} true loss"] = S / T_clean
            if f"{m} acc" in tot:
                row[f"{m} acc"] = tot[f"{m} acc"]
        summ.append(row)
        words.append(_word_drift(ctx, sentence, v))
    w = pd.concat(words, ignore_index=True)
    s = pd.DataFrame(summ)
    mean_cos = w.groupby("variant", sort=False).cosine.mean()
    s.insert(6, "mean word cosine", s.variant.map(mean_cos))
    return {"summary": s, "words": w, "tokens": tokens, "variants": variants}


# ---- notebook display ------------------------------------------------------------------------------
def show(result: dict, detail: list[str] | None = None, max_colwidth: int = 60) -> None:
    from IPython.display import Markdown, display
    with pd.option_context("display.max_colwidth", max_colwidth, "display.width", 250):
        display(Markdown("**Summary per variant** (H = loss per noisy target; true loss = summed loss / clean target count)"))
        display(result["summary"].round(3))
        display(Markdown("**Per-word static-vector drift** (frozen BERT table, mean of piece vectors)"))
        changed = result["words"][result["words"]["clean word"] != result["words"]["noisy word"]]
        display(changed.round(3) if len(changed) else result["words"].round(3))
        for name in (detail or ["clean"]):
            display(Markdown(f"**Per-token predictions: {name}** (loss in nats; top guesses with probability)"))
            display(result["tokens"][name].round(2))


def widget(ctx: TraceContext | None = None, sentence: str = "A cow jumps over the fence"):
    """Interactive version: type a sentence / focus word / rates / custom typos and press Run."""
    import ipywidgets as W
    from IPython.display import display
    ctx = ctx or load_context()
    t_sent = W.Text(value=sentence, description="sentence", layout=W.Layout(width="520px"))
    t_word = W.Text(value=sentence.split()[1] if len(sentence.split()) > 1 else "", description="focus word")
    t_rates = W.Text(value="0.1, 0.3, 0.5, 0.9", description="rates")
    t_seed = W.IntText(value=11, description="seed")
    t_custom = W.Textarea(value="", placeholder="optional: one noisy version per line", description="custom",
                          layout=W.Layout(width="520px", height="60px"))
    t_detail = W.Text(value="clean", description="detail", placeholder="variant names, comma separated",
                      layout=W.Layout(width="520px"))
    btn, out = W.Button(description="Run", button_style="primary"), W.Output()

    def go(_):
        out.clear_output()
        with out:
            rates = [float(x) for x in t_rates.value.replace(" ", "").split(",") if x]
            custom = {f"custom {i + 1}": line for i, line in enumerate(t_custom.value.splitlines()) if line.strip()}
            r = trace(t_sent.value, ctx, rates, t_seed.value, t_word.value.strip() or None, custom)
            show(r, [d.strip() for d in t_detail.value.split(",") if d.strip() in r["tokens"]] or ["clean"])
    btn.on_click(go)
    display(W.VBox([t_sent, W.HBox([t_word, t_seed]), t_rates, t_custom, t_detail, btn, out]))
    go(None)
