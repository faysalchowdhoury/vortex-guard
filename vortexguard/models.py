"""VortexGuard detector and baseline host intrusion detectors.

VortexGuard has two branches whose scores are fused by a learned gate:

1. **Selective state-space branch.** A stack of Mamba-style selective SSM
   blocks over syscall tokens. For input u_t the discretised recurrence is
       h_t = Abar_t * h_{t-1} + Bbar_t u_t,   y_t = C_t h_t + D u_t,
   with Abar_t = exp(Delta_t A) and Bbar_t = Delta_t B_t, where Delta_t, B_t and
   C_t are functions of the input (selectivity). The recurrence is O(L) in time
   and O(1) in state per streamed event.
2. **Flux-divergence branch.** Syscalls are edges of a 5-node resource graph
   (process tree P, filesystem F, network N, children C, privilege K). Edge
   fluxes are leaky integrals of event indicators at two time constants; the
   graph divergence div(phi) = B phi gives net outflow per node. A small
   temporal network maps fluxes and divergences to a score.

The fused logit is s = g * s_ssm + (1 - g) * s_flux with g = sigmoid(w^T [z_ssm; z_flux] + b).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import EDGES, VOCAB_SIZE, PAD, edge_lookup, incidence_matrix

TAUS = (16.0, 512.0)              # fast / slow flux time constants (events)
N_EDGE = len(EDGES)
FLUX_DIM = 2 * N_EDGE + 2 * 5     # fluxes + divergences at both time constants

_EDGE_LUT = edge_lookup()
_INC = incidence_matrix()


# --------------------------------------------------------------------------
# Flux-divergence features (parameter free, causal, O(L))
# --------------------------------------------------------------------------
def flux_features(tok: np.ndarray) -> np.ndarray:
    """Return (L, FLUX_DIM) causal flux and divergence features for a token sequence."""
    L = len(tok)
    onehot = np.zeros((L, N_EDGE), dtype=np.float32)
    e = _EDGE_LUT[tok]
    m = e >= 0
    onehot[np.nonzero(m)[0], e[m]] = 1.0
    feats = []
    for tau in TAUS:
        a = math.exp(-1.0 / tau)
        phi = np.zeros_like(onehot)
        acc = np.zeros(N_EDGE, dtype=np.float32)
        for t in range(L):                      # leaky integrator dphi/dt = -phi/tau + x
            acc = a * acc + (1.0 - a) * onehot[t]
            phi[t] = acc
        feats.append(phi)
        feats.append(phi @ _INC.T)              # divergence at every node
    return np.concatenate(feats, axis=1)


class FluxState:
    """O(1) streaming version of :func:`flux_features`."""

    def __init__(self):
        self.acc = [np.zeros(N_EDGE, dtype=np.float32) for _ in TAUS]
        self.a = [math.exp(-1.0 / t) for t in TAUS]

    def step(self, tok: int) -> np.ndarray:
        x = np.zeros(N_EDGE, dtype=np.float32)
        e = _EDGE_LUT[tok]
        if e >= 0:
            x[e] = 1.0
        out = []
        for i, a in enumerate(self.a):
            self.acc[i] = a * self.acc[i] + (1.0 - a) * x
            out.append(self.acc[i])
            out.append(_INC @ self.acc[i])
        return np.concatenate(out)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def masked_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked mean and masked max over time. h:(B,L,D) mask:(B,L)."""
    m = mask.unsqueeze(-1).to(h.dtype)
    mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
    mx = h.masked_fill(m == 0, -1e4).max(1).values
    return torch.cat([mean, mx], dim=-1)


class SelectiveSSM(nn.Module):
    """One Mamba-style block: in_proj -> causal depthwise conv -> selective scan -> gate."""

    def __init__(self, d_model=64, d_state=16, expand=2, d_conv=4, dt_rank=4):
        super().__init__()
        di = expand * d_model
        self.d_inner, self.d_state, self.d_conv, self.dt_rank = di, d_state, d_conv, dt_rank
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * di, bias=False)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, di)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(di, 1)
        self.A_log = nn.Parameter(torch.log(A))          # A = -exp(A_log) < 0 (stable)
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d_model, bias=False)
        with torch.no_grad():                              # dt in [1e-3, 1e-1]
            dt = torch.exp(torch.rand(di) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _ssm_params(self, x):
        dbc = self.x_proj(x)
        dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))               # (..., di)
        A = -torch.exp(self.A_log)                         # (di, N)
        return delta, A, Bm, Cm

    def forward(self, u):                                  # u: (B, L, d_model)
        B_, L, _ = u.shape
        xz = self.in_proj(self.norm(u))
        x, z = xz.chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2)
        x = F.silu(x)
        delta, A, Bm, Cm = self._ssm_params(x)
        dA = torch.exp(delta.unsqueeze(-1) * A)            # Abar: (B, L, di, N)
        dBx = delta.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)   # Bbar u
        h = torch.zeros(B_, self.d_inner, self.d_state, dtype=u.dtype, device=u.device)
        ys = []
        # unbind once: per-step slicing would make autograd allocate a full-size
        # gradient buffer for every step (quadratic in L)
        for a_t, b_t, c_t in zip(dA.unbind(1), dBx.unbind(1), Cm.unsqueeze(2).unbind(1)):
            h = a_t * h + b_t                              # h_t = Abar_t h_{t-1} + Bbar_t u_t
            ys.append((h * c_t).sum(-1))
        y = torch.stack(ys, dim=1) + x * self.D
        y = y * F.silu(z)
        return u + self.out_proj(y)

    # ---- streaming (one event at a time) ----
    def init_state(self, batch=1):
        return (torch.zeros(batch, self.d_inner, self.d_conv - 1),
                torch.zeros(batch, self.d_inner, self.d_state))

    def step(self, u, state):                              # u: (B, d_model)
        conv_buf, h = state
        xz = self.in_proj(self.norm(u))
        x, z = xz.chunk(2, dim=-1)
        window = torch.cat([conv_buf, x.unsqueeze(-1)], dim=-1)          # (B, di, d_conv)
        xc = (window * self.conv.weight.squeeze(1)).sum(-1) + self.conv.bias
        xc = F.silu(xc)
        delta, A, Bm, Cm = self._ssm_params(xc)
        h = torch.exp(delta.unsqueeze(-1) * A) * h + delta.unsqueeze(-1) * Bm.unsqueeze(1) * xc.unsqueeze(-1)
        y = (h * Cm.unsqueeze(1)).sum(-1) + xc * self.D
        y = y * F.silu(z)
        return u + self.out_proj(y), (window[..., 1:], h)


# --------------------------------------------------------------------------
# VortexGuard
# --------------------------------------------------------------------------
class VortexGuard(nn.Module):
    """Dual-branch detector. ``mode`` in {"full", "ssm_only", "flux_only", "no_gate"}."""

    def __init__(self, d_model=64, n_layers=2, d_state=8, flux_hidden=32, mode="full"):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.layers = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.flux_in = nn.Linear(FLUX_DIM, flux_hidden)
        self.flux_conv = nn.Conv1d(flux_hidden, flux_hidden, 5, padding=4)
        self.head_ssm = nn.Linear(2 * d_model, 1)
        self.head_flux = nn.Linear(2 * flux_hidden, 1)
        self.gate = nn.Linear(2 * d_model + 2 * flux_hidden, 1)

    def branch_embeddings(self, tok, flux, mask):
        if self.mode == "flux_only":          # ablation: SSM branch removed entirely
            z_ssm = tok.new_zeros(tok.shape[0], 2 * self.emb.embedding_dim, dtype=torch.float32)
        else:
            h = self.emb(tok)
            for layer in self.layers:
                h = layer(h)
            z_ssm = masked_pool(self.norm_f(h), mask)
        f = F.gelu(self.flux_in(flux))
        L = f.shape[1]
        f = F.gelu(self.flux_conv(f.transpose(1, 2))[..., :L].transpose(1, 2)) + f
        z_flux = masked_pool(f, mask)
        return z_ssm, z_flux

    def fuse(self, z_ssm, z_flux):
        s_ssm = self.head_ssm(z_ssm).squeeze(-1)
        s_flux = self.head_flux(z_flux).squeeze(-1)
        if self.mode == "ssm_only":
            g = torch.ones_like(s_ssm)
        elif self.mode == "flux_only":
            g = torch.zeros_like(s_ssm)
        elif self.mode == "no_gate":
            g = torch.full_like(s_ssm, 0.5)
        else:
            g = torch.sigmoid(self.gate(torch.cat([z_ssm, z_flux], -1)).squeeze(-1))
        return g * s_ssm + (1 - g) * s_flux, s_ssm, s_flux, g

    def forward(self, tok, flux, mask) -> Dict[str, torch.Tensor]:
        z_ssm, z_flux = self.branch_embeddings(tok, flux, mask)
        logit, s_ssm, s_flux, g = self.fuse(z_ssm, z_flux)
        return {"logit": logit, "s_ssm": s_ssm, "s_flux": s_flux, "gate": g}


class StreamingVortexGuard:
    """Event-at-a-time inference with O(1) state, used by the live containment daemon."""

    def __init__(self, model: VortexGuard):
        self.m = model.eval()
        self.reset()

    def reset(self):
        self.states = [l.init_state() for l in self.m.layers]
        self.flux = FluxState()
        hid = self.m.flux_in.out_features
        self.fbuf = torch.zeros(1, hid, 4)
        d = self.m.emb.embedding_dim
        self.sum_s = torch.zeros(1, d); self.max_s = torch.full((1, d), -1e4)
        self.sum_f = torch.zeros(1, hid); self.max_f = torch.full((1, hid), -1e4)
        self.n = 0

    @torch.no_grad()
    def step(self, tok: int) -> float:
        u = self.m.emb(torch.tensor([tok]))
        for i, layer in enumerate(self.m.layers):
            u, self.states[i] = layer.step(u, self.states[i])
        hs = self.m.norm_f(u)
        f0 = F.gelu(self.m.flux_in(torch.from_numpy(self.flux.step(tok)).unsqueeze(0)))
        win = torch.cat([self.fbuf, f0.unsqueeze(-1)], -1)
        fc = F.gelu((win.unsqueeze(1) * self.m.flux_conv.weight.unsqueeze(0)).sum((-1, -2)) + self.m.flux_conv.bias) + f0
        self.fbuf = win[..., 1:]
        self.n += 1
        self.sum_s += hs; self.max_s = torch.maximum(self.max_s, hs)
        self.sum_f += fc; self.max_f = torch.maximum(self.max_f, fc)
        z_ssm = torch.cat([self.sum_s / self.n, self.max_s], -1)
        z_flux = torch.cat([self.sum_f / self.n, self.max_f], -1)
        logit, *_ = self.m.fuse(z_ssm, z_flux)
        return float(logit)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, L, device=None):
        pos = torch.arange(L, device=device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d, 2, device=device).float() * (-math.log(10000.0) / self.d))
        pe = torch.zeros(L, self.d, device=device)
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        return pe


class TransformerIDS(nn.Module):
    """Transformer encoder HIDS over syscall tokens (masked mean+max pooling)."""

    def __init__(self, d_model=64, n_layers=2, n_heads=4, ff=128):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.pe = SinusoidalPE(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(2 * d_model, 1)

    def forward(self, tok, flux, mask):
        h = self.emb(tok) + self.pe(tok.shape[1], tok.device)
        h = self.enc(h, src_key_padding_mask=(mask == 0))
        return {"logit": self.head(masked_pool(h, mask)).squeeze(-1)}


class LSTMIDS(nn.Module):
    """Two-layer LSTM HIDS over syscall tokens."""

    def __init__(self, d_model=64, hidden=64, n_layers=2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.lstm = nn.LSTM(d_model, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, tok, flux, mask):
        h, _ = self.lstm(self.emb(tok))
        return {"logit": self.head(masked_pool(h, mask)).squeeze(-1)}


class StaticSeccompRuleBaseline:
    """A static seccomp-style allowlist profiled from benign training traces.

    The profile is the set of syscalls observed in normal training traces (the
    usual way container seccomp profiles are generated by tracing). The score
    of a trace is the number of distinct syscalls outside the profile; any
    positive score is a violation.
    """

    def fit(self, seqs, labels):
        self.allow = set()
        for s, y in zip(seqs, labels):
            if y == 0:
                self.allow.update(s)
        return self

    def score(self, seq) -> float:
        return float(len(set(seq) - self.allow))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
