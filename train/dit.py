"""AGP fusion + DiT action head + target-id head (Stage 2).

The DiT denoises the 16-step action chunk (flow matching) while interleaving triple
cross-attention over the two reasoners' outputs and the VLM context (design §3.1):

  Q_action ↔ Z^ex   (EAR coarse-waypoint guidance)   — every 2 layers
  Q_action ↔ Z^im   (IAR implicit priors)            — every 2 layers
  Q_action ↔ VLM KV (global semantics)               — every 4 layers

Proprioception (8d) is projected and added to every action token. The target-id
head grounds the instruction to one candidate box (L_target_id), the load-bearing
language signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ear import sinusoidal_time


def cross_attn_schedule(n_layers):
    """Assign each DiT layer a cross-attention context (matches §3.1 cadence)."""
    sched = []
    for i in range(n_layers):
        if i % 4 == 3:
            sched.append("vlm")          # every 4th layer
        elif i % 2 == 0:
            sched.append("ex")           # ~every 2
        else:
            sched.append("im")           # ~every 2
    return sched


class _DiTBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, ctx, ctx_kpm=None):
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.cross_attn(h, ctx, ctx, key_padding_mask=ctx_kpm, need_weights=False)[0]
        x = x + self.ff(self.n3(x))
        return x


class DiT(nn.Module):
    def __init__(self, action_dim, horizon, cond_dim_vlm, d_ex, d_im,
                 proprio_dim=8, d=512, n_layers=32, n_heads=8):
        super().__init__()
        self.horizon, self.action_dim, self.d = horizon, action_dim, d
        self.act_in = nn.Linear(action_dim, d)
        self.pos = nn.Parameter(torch.randn(1, horizon, d) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, d), nn.GELU(), nn.Linear(d, d))
        self.p_ex = nn.Linear(d_ex, d)          # project Z^ex tokens -> d
        self.p_im = nn.Linear(d_im, d)          # project Z^im tokens -> d
        self.p_vlm = nn.Linear(cond_dim_vlm, d)  # project VLM context -> d
        self.sched = cross_attn_schedule(n_layers)
        self.blocks = nn.ModuleList([_DiTBlock(d, n_heads) for _ in range(n_layers)])
        self.out = nn.Linear(d, action_dim)

    def forward(self, a_t, t, z_ex, z_im, vlm_ctx, proprio, vlm_mask=None):
        """a_t (B,H,action_dim), t (B,), z_ex (B,Ke,d_ex), z_im (B,Ki,d_im),
        vlm_ctx (B,M,C_vlm), proprio (B,proprio_dim). Returns v_hat (B,H,action_dim)."""
        h = self.act_in(a_t) + self.pos
        h = h + self.time_mlp(sinusoidal_time(t, self.d))[:, None, :]
        h = h + self.proprio(proprio)[:, None, :]
        ctx = {"ex": self.p_ex(z_ex), "im": self.p_im(z_im), "vlm": self.p_vlm(vlm_ctx)}
        vlm_kpm = (~vlm_mask) if vlm_mask is not None else None
        for blk, which in zip(self.blocks, self.sched):
            kpm = vlm_kpm if which == "vlm" else None
            h = blk(h, ctx[which], kpm)
        return self.out(h)


class TargetIDHead(nn.Module):
    """Score each candidate box as 'the referred target'. Each candidate (query)
    CROSS-ATTENDS to the full VLM context tokens (which have language fused in per
    D2) so it can look up whether the language describes it — replacing the old
    mean-pooled ctx summary that diluted the localized language signal. Trained by
    cross-entropy over candidates (L_target_id)."""

    def __init__(self, cand_dim, ctx_dim, d=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(cand_dim, d)
        self.k = nn.Linear(ctx_dim, d)
        self.v = nn.Linear(ctx_dim, d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d)
        # dropout regularizes this small head: on ~66-episode data its val mis_follow
        # peaked ~ep5 then drifted back toward chance (E1 on mvp_full_v3 overfit).
        self.score = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, cand_feats, vlm_ctx, ctx_mask=None, cand_mask=None):
        """cand_feats (B,N,cand_dim); vlm_ctx (B,M,ctx_dim); ctx_mask (B,M) True=valid.
        Returns logits (B,N)."""
        q = self.q(cand_feats)                                  # (B,N,d)
        k = self.k(vlm_ctx); v = self.v(vlm_ctx)                # (B,M,d)
        kpm = (~ctx_mask) if ctx_mask is not None else None     # key_padding_mask: True=ignore
        a, _ = self.attn(q, k, v, key_padding_mask=kpm)         # each cand attends to ctx tokens
        s = self.score(self.norm(a)).squeeze(-1)                # (B,N)
        if cand_mask is not None:
            s = s.masked_fill(~cand_mask, float("-inf"))
        return s
