"""Phase 3 extensions: deeper networks, true loss, extended noise grid, controlled trace edits."""
import math

import pandas as pd
import pytest
import torch

from src.ext.scoring import add_true_loss, operation_ranking
from src.ext.trace import _controlled, make_variants
from src.neural.model import NextTokenNet
from src.neural.train import resolve_device, train_model


def _net(mode, n_layers, d=4, K=5, h=3, V=20, seed=0):
    torch.manual_seed(seed)
    return NextTokenNet(torch.randn(V, d), mode, K, h, n_layers)


def test_default_depth_keeps_the_assignment3_network():
    torch.manual_seed(0); E = torch.randn(20, 4)
    torch.manual_seed(1); a = NextTokenNet(E, "concat", 5, 3)
    torch.manual_seed(1); b = NextTokenNet(E, "concat", 5, 3, n_layers=1)
    assert list(a.state_dict()) == ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]   # checkpoint keys unchanged
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())  # same initialisation


@pytest.mark.parametrize("n", [2, 3])
def test_extra_layers_parameter_count(n):
    d, K, h = 4, 5, 3
    for mode, m in (("mean", d), ("concat", 2 * d)):
        net = _net(mode, n)
        assert net.counts()["trainable"] == h * m + h + (n - 1) * (h * h + h) + K * h + K
        assert net(torch.tensor([[1, -1]])).shape == (1, K)


def test_zero_layers_rejected():
    with pytest.raises(ValueError):
        _net("mean", 0)


def test_unavailable_gpu_falls_back_to_cpu():
    assert resolve_device(None).type == "cpu" and resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        assert resolve_device("cuda").type == "cpu"


def test_deeper_training_runs_and_keeps_best_epoch():
    torch.manual_seed(0); E = torch.randn(30, 4)
    g = torch.Generator().manual_seed(0)
    ctx, tgt = torch.randint(0, 30, (200, 2), generator=g), torch.randint(0, 6, (200,), generator=g)
    cfg = {"seed": 101, "hidden_size": 8, "n_layers": 3, "learning_rate": 0.01, "weight_decay": 0.0,
           "batch_size": 32, "max_epochs": 4, "eval_chunk": 64}
    out = train_model("concat", E, 6, (ctx, tgt), (ctx[:50], tgt[:50]), cfg)
    v = [h["val_loss"] for h in out["history"]]
    assert out["best_epoch"] == v.index(min(v)) + 1 and out["model"].n_layers == 3


def test_true_loss_uses_the_clean_target_count():
    rows = pd.DataFrame([
        {"model": "m", "condition_id": "clean", "T": 10, "S": 20.0, "H": 2.0},
        {"model": "m", "condition_id": "ins-p0.20-s11", "T": 20, "S": 60.0, "H": 3.0},
    ])
    out = add_true_loss(rows, ["model"]).set_index("condition_id")
    assert out.loc["clean", "H_true"] == 2.0                     # clean: true loss == H
    assert out.loc["ins-p0.20-s11", "H_true"] == 6.0             # 60 nats charged to the 10 clean targets
    assert out.loc["ins-p0.20-s11", "dH"] == 1.0 and out.loc["ins-p0.20-s11", "dH_true"] == 4.0


def test_operation_ranking_direction():
    s = pd.DataFrame({"model": "m", "p": 0.2, "operation": ["insertion", "deletion", "substitution"],
                      "loss": [3.0, 1.0, 2.0], "cos": [0.9, 0.5, 0.7]})
    assert operation_ranking(s, "loss", "high", ["model", "p"]).iloc[0]["order (most > least)"] == "ins > sub > del"
    assert operation_ranking(s, "cos", "low", ["model", "p"]).iloc[0]["most harmful"] == "deletion"


def test_controlled_edits_hit_the_same_letter():
    v = {d["operation"]: d["text"] for d in _controlled("A cow jumps over the fence", "cow", None, [])}
    assert v == {"insertion": "A coow jumps over the fence", "deletion": "A cw jumps over the fence",
                 "substitution": "A cpw jumps over the fence"}


def test_trace_variants_are_deterministic():
    a = make_variants("A cow jumps over the fence", rates=(0.3, 0.9), seed=11, focus_word="cow")
    b = make_variants("A cow jumps over the fence", rates=(0.3, 0.9), seed=11, focus_word="cow")
    assert [x["text"] for x in a] == [x["text"] for x in b]
    assert a[0]["text"] == "A cow jumps over the fence" and len(a) == 1 + 3 + 3 * 2
    assert all(x["text"].count(" ") == 5 for x in a)          # the a2_default rule never edits whitespace


def test_zero_dropout_draws_no_random_numbers():
    net = _net("concat", 2)
    net.train()
    torch.manual_seed(5); before = torch.rand(1)
    torch.manual_seed(5); net(torch.tensor([[1, 2], [3, -1]])); after = torch.rand(1)
    assert torch.equal(before, after)                         # dropout 0 is skipped: RNG stream untouched


def test_dropout_only_in_training_mode():
    torch.manual_seed(0)
    net = NextTokenNet(torch.randn(20, 4), "mean", 5, 16, n_layers=2, dropout=0.5)
    x = torch.tensor([[1, 2], [3, 4]])
    net.eval()
    assert torch.equal(net(x), net(x))                         # evaluation is deterministic
    net.train()
    assert not torch.equal(net(x), net(x))                     # training drops units
    with pytest.raises(ValueError):
        NextTokenNet(torch.randn(20, 4), "mean", 5, 16, dropout=1.0)


def test_patience_stops_early_and_keeps_best():
    torch.manual_seed(0); E = torch.randn(30, 4)
    g = torch.Generator().manual_seed(0)
    ctx, tgt = torch.randint(0, 30, (200, 2), generator=g), torch.randint(0, 6, (200,), generator=g)
    cfg = {"seed": 101, "hidden_size": 32, "learning_rate": 0.05, "weight_decay": 0.0, "batch_size": 16,
           "max_epochs": 40, "eval_chunk": 64, "patience": 2}
    out = train_model("concat", E, 6, (ctx, tgt), (ctx[100:], torch.randint(0, 6, (100,), generator=g)), cfg)
    n = len(out["history"])
    assert n < 40 and n - out["best_epoch"] == 2               # stopped 2 epochs after the last improvement


def test_training_chunk_is_article_complete_and_nested():
    import json
    from src.ext.scale import chunk_records
    from src.phase2_config import DERIVED_P2_DIR
    ch = chunk_records("train_full", 10000)
    full = [json.loads(l) for l in open(DERIVED_P2_DIR / "train_full.jsonl", encoding="utf-8")]
    assert len(ch) >= 10000 and [r["record_id"] for r in ch] == [r["record_id"] for r in full[:len(ch)]]
    assert full[len(ch)]["article_id"] != ch[-1]["article_id"]            # the last article is not cut
    slice_ids = {json.loads(l)["record_id"] for l in open(DERIVED_P2_DIR / "train_a1_slice.jsonl", encoding="utf-8")}
    assert slice_ids <= {r["record_id"] for r in ch}                        # the 500-record slice is inside the chunk
