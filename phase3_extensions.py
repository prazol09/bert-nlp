#!/usr/bin/env python
"""Phase 3 extensions (team decisions beyond the Assignment 3 spec). Settings: `extensions:` in configs/phase3.yaml.

    python phase3_extensions.py                      # E1-E3: heavier noise, true loss, layer x epoch sweep
    python phase3_extensions.py --no-sweep           # E1-E2 only (a few minutes)
    python phase3_extensions.py regularise           # R1: dropout x weight-decay grid + winners to convergence (~55 min)
    python phase3_extensions.py scale                # R2: 10K-record training chunk, networks + matched n-gram (~45 min)
    python phase3_extensions.py trace "A cow jumps over the fence" --word cow --rates 0.1 0.3 0.5 0.9
    python phase3_extensions.py trace "The cat sat on the mat" --custom "The cta sat on teh mat"

Results: results/phase3/ext/ (tables, figures, run info) and derived/phase3/{noisy_ext,ext}/.
The Assignment 2/3 artifacts are only read; the run fails if it cannot reproduce them.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--no-sweep", action="store_true", help="skip the hidden-layer x epoch sweep (E3)")
    rg = sub.add_parser("regularise", help="R1: dropout x weight-decay grid on the 500-record slice")
    rg.add_argument("--long-only", action="store_true", help="only re-train the grid winners to convergence (R1b)")
    sc = sub.add_parser("scale", help="R2: re-train on a 10K-record training chunk, with the n-gram rebuilt on the same chunk")
    sc.add_argument("--depth-only", action="store_true", help="R2b: re-train the best 10K settings at 2 and 3 hidden layers")
    tr = sub.add_parser("trace", help="follow one sentence through the pipeline")
    tr.add_argument("sentence")
    tr.add_argument("--word", default=None, help="focus word for the controlled one-edit variants")
    tr.add_argument("--rates", type=float, nargs="*", default=[0.1, 0.3, 0.5, 0.9])
    tr.add_argument("--seed", type=int, default=11)
    tr.add_argument("--custom", nargs="*", default=[], help="your own noisy versions of the sentence")
    tr.add_argument("--detail", nargs="*", default=["clean"], help="variants whose per-token table to print")
    args = ap.parse_args()

    import pandas as pd
    if args.cmd == "trace":
        from src.ext.trace import trace
        r = trace(args.sentence, rates=args.rates, seed=args.seed, focus_word=args.word,
                  custom={f"custom {i + 1}": t for i, t in enumerate(args.custom)})
        with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 50):
            print(r["summary"].round(3).to_string(index=False))
            w = r["words"]
            print("\nper-word drift (changed words):")
            print(w[w["clean word"] != w["noisy word"]].round(3).to_string(index=False))
            for d in args.detail:
                if d in r["tokens"]:
                    print(f"\nper-token predictions: {d}")
                    print(r["tokens"][d].round(2).to_string(index=False))
        return

    if args.cmd == "scale":
        from src.ext.scale import run as run_scale, run_depth
        if args.depth_only:
            print(run_depth()["runs"][["run", "n_layers", "epochs_run", "best_epoch", "best_val_loss", "val_acc_at_best"]].round(4).to_string(index=False))
            return
        out = run_scale()
        print(out["runs"].round(4).to_string(index=False))
        return

    if args.cmd == "regularise":
        from src.ext.regularise import run as run_reg, run_long
        if args.long_only:
            print(run_long()["long"].round(4).to_string(index=False)); return
        out = run_reg()
        run_long()
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(out["grid"][["run", "best_epoch", "best_val_loss", "val_acc_at_best", "val_loss_at_30", "val_rise_after_best"]].round(4).to_string(index=False))
            print("\nwinners:"); print(out["best"][["model", "dropout", "weight_decay", "best_epoch", "best_val_loss"]].round(4).to_string(index=False))
        return

    from src.ext.run import run
    out = run(sweep=not args.no_sweep)
    s = out["summary"]
    print("\ntrue loss (per clean target) at p = 0.20 and 0.90, mean over seeds:")
    print(s[s.p.isin([0.2, 0.9])][["model", "operation", "p", "H_mean", "H_true_mean", "dH_true_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
