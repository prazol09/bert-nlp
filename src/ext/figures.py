"""Figures for the Phase 3 extensions (matplotlib PNG). Every figure carries the extension label.

Colours: models keep the Assignment 3 colours (src/neural/figures.py). Operations use a separate validated
three-hue set plus a distinct marker each, so a model colour never means an operation. Layer depth is
ordinal, so it is one blue ramp from light (1 layer) to dark (3 layers), drawn in one panel per model.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..geometry.drift import OP_ORDER
from ..neural.figures import COL, NAME

OP_COL = {"insertion": "#6250d6", "deletion": "#eda100", "substitution": "#008300"}
OP_MARK = {"insertion": "o", "deletion": "s", "substitution": "^"}
DEPTH_COL = {1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}
NOTE = ("Extension beyond the Assignment 3 spec. Preliminary validation results. "
        "Band = lowest to highest of 3 corruption seeds (not a confidence interval).")


def _save(fig, path):
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def _with_clean(summary: pd.DataFrame, op: str, extra: dict) -> pd.DataFrame:
    """One operation's rows plus the shared clean row drawn at p = 0."""
    m = pd.Series(True, index=summary.index)
    for k, v in extra.items():
        m &= summary[k] == v
    g = summary[m & summary.operation.isin([op, "clean"])]
    return g.sort_values("p")


def _xaxis(ax, rates):
    ax.set_xticks(range(len(rates)), [f"{r:g}" for r in rates])
    ax.set_xlabel("noise rate p (0 = clean)")
    ax.grid(alpha=0.3)


def plot_noise_ext(summary: pd.DataFrame, path=None):
    """Rows: loss per noisy target H, true loss per clean target, accuracy. Columns: operation."""
    import matplotlib.pyplot as plt
    rates = sorted(summary.p.unique()); xs = {r: i for i, r in enumerate(rates)}
    rows = [("H", "H: loss per NOISY target (nats)"), ("H_true", "true loss: per CLEAN target (nats)"),
            ("accuracy", "top-1 accuracy")]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharey="row")
    for j, op in enumerate(OP_ORDER):
        for i, (m, lab) in enumerate(rows):
            ax = axes[i, j]
            for key in ("mean", "concat", "ngram"):
                if m == "accuracy" and key == "ngram":
                    continue
                g = _with_clean(summary, op, {"model": key})
                x = [xs[r] for r in g.p]
                ax.plot(x, g[f"{m}_mean"], marker="o", ms=4, lw=2, color=COL[key], label=NAME[key])
                ax.fill_between(x, g[f"{m}_min"], g[f"{m}_max"], color=COL[key], alpha=0.2, lw=0)
            _xaxis(ax, rates)
            if j == 0:
                ax.set_ylabel(lab)
        axes[0, j].set_title(op)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(NOTE + "\nTrue loss divides every condition's summed loss by the same clean target count (35,971).", fontsize=8)
    return _save(fig, path)


def plot_operation_vs_med(summary: pd.DataFrame, geo: pd.DataFrame, noise: pd.DataFrame, path=None):
    """Compare operations at equal ACTUAL damage: x = realized character edit distance per clean character."""
    import matplotlib.pyplot as plt
    med = noise[noise.operation != "clean"].groupby(["operation", "p"]).normalized_med.mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for op in OP_ORDER:
        for ax, key in ((axes[0], "concat"), (axes[1], "ngram")):
            g = summary[(summary.model == key) & (summary.operation == op)].sort_values("p")
            ax.plot([med[(op, p)] for p in g.p], g.H_true_mean, marker=OP_MARK[op], ms=6, lw=2, color=OP_COL[op], label=op)
        g = geo[geo.operation == op].sort_values("rate")
        axes[2].plot([med[(op, p)] for p in g.rate], g.mean_cosine_mean, marker=OP_MARK[op], ms=6, lw=2,
                     color=OP_COL[op], label=op)
    for ax, t, y in ((axes[0], "concatenation model", "true loss per clean target (nats)"),
                     (axes[1], "n-gram control", "true loss per clean target (nats)"),
                     (axes[2], "static vector drift", "mean cosine (lower = more drift)")):
        ax.set_title(t, fontsize=10); ax.set_ylabel(y); ax.set_xlabel("edit distance per clean character")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(NOTE.split(" Band")[0] + " Points are p = 0.01 ... 0.90, placed at the edit distance they actually caused.",
                 fontsize=8)
    return _save(fig, path)


def plot_geometry_ext(geo: pd.DataFrame, path=None):
    import matplotlib.pyplot as plt
    rates = sorted(geo.rate.unique()); xs = {r: i for i, r in enumerate(rates)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = (("mean_cosine", "mean cosine (all valid spans)"), ("mean_distance", "mean Euclidean distance"),
              ("mean_cosine_changed", "mean cosine (changed spans only)"))
    for ax, (m, lab) in zip(axes, panels):
        for op in OP_ORDER:
            g = geo[geo.operation == op].sort_values("rate")
            x = [xs[r] for r in g.rate]
            ax.plot(x, g[f"{m}_mean"], marker=OP_MARK[op], ms=5, lw=2, color=OP_COL[op], label=op)
            ax.fill_between(x, g[f"{m}_min"], g[f"{m}_max"], color=OP_COL[op], alpha=0.2, lw=0)
        ax.set_xticks(range(len(rates)), [f"{r:g}" for r in rates]); ax.set_xlabel("noise rate p")
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(NOTE + "\nMeans over valid pairs; deletion loses spans at high p (see the audit's undefined counts).",
                 fontsize=8)
    return _save(fig, path)


def plot_sweep_training(history: pd.DataFrame, path=None):
    """Loss per epoch for 1/2/3 hidden layers. Solid = clean validation, dashed = training; circle = epoch kept."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, mode in zip(axes[:2], ("mean", "concat")):
        for n, g in history[history.model == mode].groupby("n_layers"):
            ax.plot(g.epoch, g.train_loss, ls="--", lw=1.5, color=DEPTH_COL[n])
            ax.plot(g.epoch, g.val_loss, lw=2, color=DEPTH_COL[n], label=f"{n} hidden layer{'s' if n > 1 else ''}")
            s = g[g.selected].iloc[0]
            ax.scatter([s.epoch], [s.val_loss], s=90, facecolors="none", edgecolors="k", zorder=5)
        ax.set_title(f"{NAME[mode]}: solid = validation, dashed = training", fontsize=9)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss (nats per target)"); ax.grid(alpha=0.3)
    for mode, ls in (("mean", ":"), ("concat", "-")):
        for n, g in history[history.model == mode].groupby("n_layers"):
            axes[2].plot(g.epoch, g.val_accuracy, ls=ls, lw=2, color=DEPTH_COL[n], label=f"{NAME[mode]}, {n} layer(s)")
    axes[2].set_title("clean validation accuracy", fontsize=9); axes[2].set_xlabel("epoch"); axes[2].grid(alpha=0.3)
    axes[0].legend(fontsize=8); axes[2].legend(fontsize=6)
    fig.suptitle("Extension: hidden-layer x epoch sweep, width 64, all other settings as Assignment 3. "
                 "Circle = epoch kept (lowest clean validation loss). Preliminary validation results.", fontsize=8)
    return _save(fig, path)


def plot_sweep_noise(ss: pd.DataFrame, path=None):
    """Does depth change robustness? Increase in true loss over each variant's own clean loss."""
    import matplotlib.pyplot as plt
    rates = sorted(ss.p.unique()); xs = {r: i for i, r in enumerate(rates)}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey=True)
    for i, mode in enumerate(("mean", "concat")):
        for j, op in enumerate(OP_ORDER):
            ax = axes[i, j]
            for n in sorted(ss.n_layers.unique()):
                g = _with_clean(ss, op, {"model": mode, "n_layers": n})
                x = [xs[r] for r in g.p]
                ax.plot(x, g.dH_true_mean, marker="o", ms=4, lw=2, color=DEPTH_COL[n], label=f"{n} layer(s)")
            _xaxis(ax, rates)
            ax.set_title(f"{NAME[mode]} | {op}", fontsize=9)
            if j == 0:
                ax.set_ylabel("true loss increase vs own clean (nats)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(NOTE, fontsize=8)
    return _save(fig, path)


# ---- R1: regularisation ------------------------------------------------------------------------------
REG_NOTE = ("Extension: dropout x weight decay on the 500-record slice, 1 hidden layer of 64, 30 epochs, otherwise as "
            "Assignment 3. Preliminary validation results; selection on validation is optimistic.")


def plot_reg_grid(grid: pd.DataFrame, path=None):
    """Best clean-validation loss for every (dropout, weight decay) cell; darker = higher (worse) loss; box = winner."""
    import matplotlib.pyplot as plt
    import numpy as np
    g1 = grid[grid.n_layers == 1]
    dos, wds = sorted(g1.dropout.unique()), sorted(g1.weight_decay.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, mode in zip(axes, ("mean", "concat")):
        g = g1[g1.model == mode]
        M = np.array([[g[(g.dropout == d) & (g.weight_decay == w)].best_val_loss.iloc[0] for w in wds] for d in dos])
        ax.imshow(M, cmap="Blues", aspect="auto")
        lo, hi = M.min(), M.max()
        for i in range(len(dos)):
            for j in range(len(wds)):
                dark = (M[i, j] - lo) / max(hi - lo, 1e-9) > 0.55
                e = int(g[(g.dropout == dos[i]) & (g.weight_decay == wds[j])].best_epoch.iloc[0])
                ax.text(j, i, f"{M[i, j]:.3f}\nepoch {e}", ha="center", va="center", fontsize=8,
                        color="white" if dark else "#15181e")
        bi, bj = np.unravel_index(M.argmin(), M.shape)
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2.5))
        ax.set_xticks(range(len(wds)), [f"{w:g}" for w in wds]); ax.set_yticks(range(len(dos)), [f"{d:g}" for d in dos])
        ax.set_xlabel("weight decay"); ax.set_ylabel("dropout")
        ax.set_title(f"{NAME[mode]}: best clean validation loss (box = lowest)", fontsize=9)
    fig.suptitle(REG_NOTE, fontsize=7)
    return _save(fig, path)


def plot_reg_curves(history: pd.DataFrame, best: pd.DataFrame, path=None):
    """Baseline (no regularisation) vs the winning setting, per model. Solid = validation, dashed = training."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, mode in zip(axes, ("mean", "concat")):
        w = best[best.model == mode].iloc[0]
        for lab, do, wd, col in (("no regularisation", 0.0, 0.0, "#86b6ef"),
                                 (f"dropout {w.dropout:g}, weight decay {w.weight_decay:g}", w.dropout, w.weight_decay, COL[mode])):
            h = history[(history.model == mode) & (history.n_layers == 1) & (history.dropout == do) & (history.weight_decay == wd)]
            ax.plot(h.epoch, h.val_loss, lw=2, color=col, label=f"{lab}, validation")
            ax.plot(h.epoch, h.train_loss, lw=1.5, ls="--", color=col, label=f"{lab}, training")
            s = h[h.selected].iloc[0]
            ax.scatter([s.epoch], [s.val_loss], s=80, facecolors="none", edgecolors="k", zorder=5)
        ax.set_title(NAME[mode], fontsize=10); ax.set_xlabel("epoch"); ax.set_ylabel("loss (nats per target)")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(REG_NOTE + " Circle = epoch kept.", fontsize=7)
    return _save(fig, path)


def plot_reg_noise(summary: pd.DataFrame, best: pd.DataFrame, path=None):
    """Does regularisation change robustness? True-loss increase over each model's own clean loss, 1 layer."""
    import matplotlib.pyplot as plt
    rates = sorted(summary.p.unique()); xs = {r: i for i, r in enumerate(rates)}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey=True)
    for i, mode in enumerate(("mean", "concat")):
        w = best[best.model == mode].iloc[0]
        for j, op in enumerate(OP_ORDER):
            ax = axes[i, j]
            for lab, do, wd, col in (("no regularisation", 0.0, 0.0, "#86b6ef"),
                                     (f"dropout {w.dropout:g}, wd {w.weight_decay:g}", w.dropout, w.weight_decay, COL[mode])):
                g = _with_clean(summary, op, {"model": mode, "n_layers": 1, "dropout": do, "weight_decay": wd})
                ax.plot([xs[r] for r in g.p], g.dH_true_mean, marker="o", ms=4, lw=2, color=col, label=lab)
            _xaxis(ax, rates); ax.set_title(f"{NAME[mode]} | {op}", fontsize=9)
            if j == 0:
                ax.set_ylabel("true loss increase vs own clean (nats)")
            ax.legend(fontsize=7)
    fig.suptitle(NOTE, fontsize=8)
    return _save(fig, path)


# ---- R2: more data -------------------------------------------------------------------------------------
def plot_scale(history: pd.DataFrame, summary: pd.DataFrame, best: pd.DataFrame, out_dir):
    """scale_training.png: learning curves per setting; scale_noise.png: best networks vs the n-gram on the same chunk."""
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    note = "Extension: 10K-record training chunk. Preliminary validation results; settings chosen on validation (optimistic)."
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, mode in zip(axes, ("mean", "concat")):
        runs = history[history.run.str.startswith(mode + "_")]
        for k, (run, h) in enumerate(runs.groupby("run", sort=False)):
            col = list(DEPTH_COL.values())[k % 3]
            lab = f"dropout {h.dropout.iloc[0]:g}, wd {h.weight_decay.iloc[0]:g}"
            ax.plot(h.epoch, h.val_loss, lw=2, color=col, label=f"{lab}, validation")
            ax.plot(h.epoch, h.train_loss, lw=1.5, ls="--", color=col)
            s = h[h.selected].iloc[0]
            ax.scatter([s.epoch], [s.val_loss], s=80, facecolors="none", edgecolors="k", zorder=5)
        ax.set_title(f"{NAME[mode]} (dashed = training, circle = epoch kept)", fontsize=9)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss (nats per target)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(note, fontsize=8)
    _save(fig, out_dir / "scale_training.png")

    rates = sorted(summary.p.unique()); xs = {r: i for i, r in enumerate(rates)}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey="row")
    lines = [(best[best.model == m].run.iloc[0], m) for m in ("mean", "concat")] + [("ngram", "ngram")]
    for j, op in enumerate(OP_ORDER):
        for i, (metric, lab) in enumerate((("H_true_mean", "true loss per clean target (nats)"),
                                           ("dH_true_mean", "true loss increase vs own clean (nats)"))):
            ax = axes[i, j]
            for run, key in lines:
                g = _with_clean(summary, op, {"run": run})
                ax.plot([xs[r] for r in g.p], g[metric], marker="o", ms=4, lw=2, color=COL[key], label=NAME[key])
            _xaxis(ax, rates)
            if j == 0:
                ax.set_ylabel(lab)
        axes[0, j].set_title(op)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(note + " All three models trained on the same chunk.", fontsize=8)
    _save(fig, out_dir / "scale_noise.png")
