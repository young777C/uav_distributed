"""IAR — Implicit Action Reasoner (Stage 2).

Reads the frozen VLM's PER-LAYER context (KV cache) via learnable queries and
distills the "unspeakable" action priors into implicit tokens Z^im, plus auxiliary
predictions the design assigns to IAR (structural-occlusion "about-to-be-occluded"
and target-maneuver detection). Inherited mechanism from ACoT-VLA; UAV-specialized
by the auxiliary heads (design §6.2).

  input : ctx_layers = list of L context token sets, each (B, M, C_vlm)
  output: Z^im (B, n_im, d)  implicit tokens (consumed by AGP cross-attention)
          aux  {occ_structural, maneuver} logits (supervise L_visibility/L_maneuver)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _LayerQuery(nn.Module):
    """One VLM layer -> implicit tokens via learnable-query cross-attention."""

    def __init__(self, cond_dim, d, n_heads, kv_down):
        super().__init__()
        self.kv = nn.Sequential(nn.Linear(cond_dim, kv_down), nn.GELU(), nn.Linear(kv_down, d))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, q, ctx, ctx_mask=None):
        kv = self.kv(ctx)
        kpm = (~ctx_mask) if ctx_mask is not None else None
        return self.norm(self.attn(q, kv, kv, key_padding_mask=kpm, need_weights=False)[0])


class IAR(nn.Module):
    def __init__(self, cond_dim, n_layers_in, d=256, n_im=8, n_heads=4, kv_down=64):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n_im, d) * 0.02)
        self.layers = nn.ModuleList(
            [_LayerQuery(cond_dim, d, n_heads, kv_down) for _ in range(n_layers_in)])
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        self.occ_head = nn.Linear(d, 1)        # structural-occlusion prediction (§6.2 #1)
        self.maneuver_head = nn.Linear(d, 1)   # maneuver-intent detection
        self.d = d

    def forward(self, ctx_layers, ctx_mask=None):
        """ctx_layers: list[(B,M,C_vlm)] (one per cached VLM layer)."""
        b = ctx_layers[0].shape[0]
        q = self.q.expand(b, -1, -1)
        per = [lyr(q, ctx, ctx_mask) for lyr, ctx in zip(self.layers, ctx_layers)]
        z = torch.stack(per, dim=0).mean(0)    # average-pool over layers -> (B, n_im, d)
        z = z + self.mlp(z)
        pooled = z.mean(1)                      # (B, d)
        aux = {
            "occ_structural": self.occ_head(pooled).squeeze(-1),
            "maneuver": self.maneuver_head(pooled).squeeze(-1),
        }
        return z, aux
