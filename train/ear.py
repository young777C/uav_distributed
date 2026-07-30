"""EAR — Explicit Action Reasoner (Stage 1).

A lightweight Transformer that denoises K coarse 3D waypoints (the "action-space
thought"), conditioned on the frozen VLM's layer-L context tokens via cross-
attention (design §3.1 / §6.1). Trained by flow matching (see flow_matching.py).

  input : x_t (B,K,3) noised waypoints, t (B,) flow time, cond (B,M,C_vlm) VLM
          context tokens (+ optional mask)
  output: v_hat (B,K,3) velocity field
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_time(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) in [0,1] -> (B, dim) Fourier features spanning low..high frequency.

    Frequencies geometrically spaced over [1, max_freq] so t is resolvable across
    the whole [0,1] flow interval without aliasing the low-frequency components.
    """
    half = dim // 2
    max_freq = 64.0
    freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), half, device=t.device))
    ang = t[:, None] * freqs[None, :] * (2 * math.pi)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, torch.zeros(t.shape[0], dim - emb.shape[-1], device=t.device)], -1)
    return emb


class _Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cond, cond_kpm):
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.cross_attn(h, cond, cond, key_padding_mask=cond_kpm, need_weights=False)[0]
        x = x + self.ff(self.n3(x))
        return x


class EAR(nn.Module):
    def __init__(self, cond_dim: int, k: int = 3, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4, proprio_dim: int = 0):
        super().__init__()
        self.k = k
        self.wp_in = nn.Linear(3, d_model)
        self.time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.Linear(d_model, d_model))
        self.pos = nn.Parameter(torch.randn(1, k, d_model) * 0.02)
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.proprio_dim = proprio_dim
        if proprio_dim > 0:      # UAV pose (altitude/pitch → depth; velocity → motion)
            self.proprio_mlp = nn.Sequential(nn.Linear(proprio_dim, d_model), nn.GELU(),
                                             nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([_Block(d_model, n_heads) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, 3)
        self.d_model = d_model

    def forward(self, x_t, t, cond, cond_mask=None, proprio=None):
        """x_t (B,K,3), t (B,), cond (B,M,C_vlm), cond_mask (B,M) True=valid,
        proprio (B,P) optional UAV-pose conditioning."""
        h = self.wp_in(x_t) + self.pos
        h = h + self.time_mlp(sinusoidal_time(t, self.d_model))[:, None, :]
        if self.proprio_dim > 0 and proprio is not None:
            h = h + self.proprio_mlp(proprio)[:, None, :]
        c = self.cond_proj(cond)
        kpm = (~cond_mask) if cond_mask is not None else None   # key_padding_mask: True=pad
        for blk in self.blocks:
            h = blk(h, c, kpm)
        return self.out(h)
