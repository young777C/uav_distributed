"""Phase E.1 — run the frozen Qwen3-VL online, per live frame (no disk cache).

Training reads a precomputed context cache (train/backbone_kv.py writes
`ctx_cache_ml_v5/*.ctx.npy` = (F, L, g*g, C) fp16). Closed-loop frames are new
and cannot be precomputed, so the harness must reproduce the SAME tensors online.

This wrapper is a thin, faithful re-assembly of the exact cache code path:

    lang' = language transform (neutral / none / full)           # backbone_kv.py:76-80
    feats = backbone.encode(PIL(frame), lang', [layer])          # acot_probe/backbones.py:144
    ctx   = _pool_grid(feats[layer][0], g)                        # backbone_kv.py:43  -> (g*g, C)

For v5 the cache stores a SINGLE layer (24) at g=16 → M=256 tokens, C=2560
(so `n_layers_in == 1`, and vlm_ctx == the single IAR layer). We keep the
list-of-layers shape so multi-layer caches keep working unchanged.

Import root: repo root must be on sys.path (for `train.*` / `acot_probe.*`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from PIL import Image

from acot_probe.backbones import build_backbone, pool_box
from train.backbone_kv import _pool_grid, neutralize_language

LangMode = Literal["neutral", "none", "full"]

# Ablation string used by the cache builder for the no-language arm (backbone_kv.py:78).
NO_LANGUAGE = "Track the target vehicle."


def apply_language_mode(raw_lang: str, mode: LangMode) -> str:
    """Match train/backbone_kv.py language handling so online == cached distribution.

    v5 was trained with `neutralize_language: true` (identity only), so the M0
    policy must be fed the SAME neutralized instruction at inference. M1 (w/o-lang)
    uses the fixed no-language string.
    """
    if mode == "none":
        return NO_LANGUAGE
    if mode == "neutral":
        return neutralize_language(raw_lang)
    return raw_lang


class OnlineVLM:
    """Frozen VLM, loaded once, encodes one live RGB frame → context tensors.

    Usage:
        vlm = OnlineVLM(cfg, device="cuda:0", language_mode="neutral")
        vlm_ctx_layers, vlm_ctx, grid_last = vlm.encode(rgb_uint8, episode_language)
        cand_feats = vlm.pool_candidates(grid_last, frac_boxes)   # (N, C)
    """

    def __init__(self, cfg: dict, device: str = "cuda:0",
                 language_mode: LangMode = "neutral"):
        b = cfg["backbone"]
        self.layer = int(b.get("layer", 24))
        self.g = int(b.get("ctx_grid", 16))
        self.device = device
        self.language_mode: LangMode = language_mode
        self.bk = build_backbone(
            b["kind"], b["model_id"], device,
            min_pixels=b.get("min_pixels"), max_pixels=b.get("max_pixels"),
        ).load()

    @torch.no_grad()
    def encode(self, rgb: np.ndarray, raw_language: str):
        """rgb: (H,W,3) uint8. Returns (vlm_ctx_layers list[(M,C)], vlm_ctx (M,C), grid_last).

        Tensors are float32 on `self.device` (the trained modules run in float32;
        the fp16 cache is upcast identically by the dataset loader). `grid_last` is
        the raw (Gh,Gw,C) numpy grid kept for candidate-box pooling.
        """
        lang = apply_language_mode(raw_language, self.language_mode)
        img = Image.fromarray(np.asarray(rgb))
        feats = self.bk.encode(img, lang, [self.layer])        # {layer: (grid[Gh,Gw,C], text_vec)}
        grid_last = feats[self.layer][0]                        # (Gh, Gw, C) float32 np
        pooled = _pool_grid(grid_last, self.g).astype(np.float32)  # (g*g, C)
        ctx = torch.from_numpy(pooled).to(self.device)          # (M, C)
        vlm_ctx_layers = [ctx]                                   # n_layers_in == 1 for v5
        return vlm_ctx_layers, ctx, grid_last

    @staticmethod
    def pool_candidates(grid_last: np.ndarray, frac_boxes) -> np.ndarray:
        """Pool the VLM grid within each fractional box → (N, C) cand_feats.

        Same op the dataset uses (dataset_stage2.py:273 via acot_probe.pool_box).
        frac_boxes: iterable of (fu0, fv0, fu1, fv1) in [0,1].
        """
        return np.stack([pool_box(grid_last, fb) for fb in frac_boxes]).astype(np.float32)

    def free(self):
        self.bk.free()
