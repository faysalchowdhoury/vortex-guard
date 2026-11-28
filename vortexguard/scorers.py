"""Online scorers that consume syscall tokens in micro-batches.

All detectors expose ``feed(tokens) -> score`` where ``score`` is the anomaly
logit after the last token. Their deployment cost differs by construction:

* VortexGuard and LSTMIDS carry a fixed-size recurrent state, so the cost of a
  micro-batch is independent of how much history has been seen.
* TransformerIDS has no recurrent state; each update re-encodes the most recent
  ``window`` events (self-attention over the full window).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .models import LSTMIDS, StreamingVortexGuard, TransformerIDS, VortexGuard, masked_pool, FLUX_DIM


class VortexGuardScorer:
    def __init__(self, model: VortexGuard):
        self.s = StreamingVortexGuard(model)

    def reset(self):
        self.s.reset()

    def feed(self, toks) -> float:
        score = float("nan")
        for t in toks:
            score = self.s.step(int(t))
        return score

    def state_bytes(self) -> int:
        n = sum(a.numel() + b.numel() for a, b in self.s.states)
        n += self.s.fbuf.numel() + 4 * self.s.sum_s.numel() + 2 * self.s.sum_f.numel()
        return 4 * n


class LSTMScorer:
    def __init__(self, model: LSTMIDS):
        self.m = model.eval()
        self.reset()

    def reset(self):
        self.hc = None
        H = self.m.lstm.hidden_size
        self.sum = torch.zeros(1, H)
        self.mx = torch.full((1, H), -1e4)
        self.n = 0

    @torch.no_grad()
    def feed(self, toks) -> float:
        x = self.m.emb(torch.as_tensor(np.asarray(toks), dtype=torch.long).unsqueeze(0))
        h, self.hc = self.m.lstm(x, self.hc)
        self.sum += h.sum(1)
        self.mx = torch.maximum(self.mx, h.max(1).values)
        self.n += h.shape[1]
        z = torch.cat([self.sum / self.n, self.mx], -1)
        return float(self.m.head(z))

    def state_bytes(self) -> int:
        H = self.m.lstm.hidden_size
        return 4 * (2 * self.m.lstm.num_layers * H + 2 * H)


class TransformerScorer:
    def __init__(self, model: TransformerIDS, window: int = 1024):
        self.m = model.eval()
        self.window = window
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=self.window)

    @torch.no_grad()
    def feed(self, toks) -> float:
        self.buf.extend(int(t) for t in toks)
        tok = torch.tensor(list(self.buf), dtype=torch.long).unsqueeze(0)
        mask = torch.ones_like(tok, dtype=torch.float32)
        return float(self.m(tok, None, mask)["logit"])

    def state_bytes(self) -> int:
        return 8 * len(self.buf)


def make_scorer(name: str, state_dict=None, window: int = 1024):
    if name == "VortexGuard":
        m = VortexGuard()
        if state_dict is not None:
            m.load_state_dict(state_dict)
        return VortexGuardScorer(m.eval())
    if name == "LSTMIDS":
        m = LSTMIDS()
        if state_dict is not None:
            m.load_state_dict(state_dict)
        return LSTMScorer(m)
    if name == "TransformerIDS":
        m = TransformerIDS()
        if state_dict is not None:
            m.load_state_dict(state_dict)
        return TransformerScorer(m, window)
    raise ValueError(name)
