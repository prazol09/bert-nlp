"""N2: the two prediction networks. Same layers, different input.

    x_mean   = (e1 + e2) / 2          (m = d)       always divide by two, BOS counts as a zero vector
    x_concat = [e1 ; e2]              (m = 2d)
    h = ReLU(W1 x + b1)   (64 hidden)       z = W2 h + b2   (K logits)       softmax is inside the loss

`n_layers` (extension, default 1 = the Assignment 3 network) inserts n_layers - 1 further
ReLU(W h + b) layers of the same width between fc1 and fc2. With the default nothing is added, so the
state-dict keys, the initialisation order and the saved checkpoints are exactly those of Assignment 3.

`dropout` (extension, default 0 = Assignment 3) drops hidden units after every hidden ReLU, during training
only. At 0 the dropout call is skipped entirely, so no random numbers are drawn and training is unchanged.

E is held as a NON-persistent buffer: it is not a Parameter, never reaches the optimizer, and is not
saved with the weights. Only W1, b1, W2, b2 are trainable.
"""
from __future__ import annotations

import torch
from torch import nn


class NextTokenNet(nn.Module):
    def __init__(self, E: torch.Tensor, mode: str, n_outputs: int, hidden: int = 64, n_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        if mode not in ("mean", "concat"):
            raise ValueError(mode)
        if n_layers < 1:
            raise ValueError(n_layers)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(dropout)
        self.mode, self.n_layers, self.dropout = mode, n_layers, float(dropout)
        d = E.shape[1]
        m = d if mode == "mean" else 2 * d
        self.register_buffer("E", E, persistent=False)         # frozen lookup table
        self.fc1 = nn.Linear(m, hidden)
        self.extra = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers - 1))   # empty by default
        self.fc2 = nn.Linear(hidden, n_outputs)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Rows of E for the ids; id -1 (BOS) gives an all-zero vector."""
        return self.E[ids.clamp(min=0)] * (ids >= 0).unsqueeze(-1).to(self.E.dtype)

    def features(self, ctx: torch.Tensor) -> torch.Tensor:
        e1, e2 = self.embed(ctx[:, 0]), self.embed(ctx[:, 1])
        return (e1 + e2) / 2 if self.mode == "mean" else torch.cat([e1, e2], dim=1)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:        # raw logits
        h = self._drop(torch.relu(self.fc1(self.features(ctx))))
        for layer in self.extra:
            h = self._drop(torch.relu(layer(h)))
        return self.fc2(h)

    def _drop(self, h: torch.Tensor) -> torch.Tensor:
        """Dropout after a hidden ReLU while training; skipped (no RNG use) when the rate is 0 or in eval mode."""
        return nn.functional.dropout(h, self.dropout, training=True) if (self.dropout > 0 and self.training) else h

    def counts(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": trainable, "frozen": int(self.E.numel())}
