"""Frozen VLM backbones for the layer-16 grounding probe.

Each backbone loads a pretrained VLM (frozen, bf16), runs ONE forward per frame
with output_hidden_states=True, and returns, for each requested decoder layer:
  - an image-token feature GRID  (Gh, Gw, C)   -- language-fused if the model's
    attention lets image tokens see the instruction (PaliGemma: yes, prefix
    bidirectional; Qwen: text-before-image so image can attend to text under
    causal attention), and
  - a pooled instruction TEXT feature (C,)      -- always provided so the probe
    can combine box+text even when image tokens did not attend to text.

The probe never fine-tunes these weights; it only reads hidden states.

IMPORTANT: the image-token -> 2D grid mapping is model-specific and version
sensitive. It is derived from the processor outputs (image_grid_thw for Qwen;
sqrt(image_seq_length) for PaliGemma). ALWAYS validate with acot_probe/visualize.py
on a few real frames before trusting probe numbers.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def resolve_layers(layers: Sequence[int], n_hidden_states: int) -> list[int]:
    """Map requested layer ids (supporting -1=last) to hidden_states indices.

    hidden_states has length (num_layers + 1); index 0 = embeddings, index L =
    output of decoder layer L. We keep that indexing; -1 -> last.
    """
    out = []
    for L in layers:
        idx = (n_hidden_states - 1) if L == -1 else L
        if 0 <= idx < n_hidden_states:
            out.append(idx)
    return sorted(set(out))


class Backbone:
    """Base wrapper. Subclasses implement `_forward_hidden` and `_image_grid`."""

    def __init__(self, model_id: str, device: str = "cuda:0",
                 min_pixels=None, max_pixels=None):
        self.model_id = model_id
        self.device = device
        self.min_pixels = min_pixels     # force a finer/coarser vision grid (Qwen)
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None

    # -- lifecycle ----------------------------------------------------------
    def load(self):
        raise NotImplementedError

    def free(self):
        import gc
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- core ---------------------------------------------------------------
    def encode(self, image, instruction: str, layers: Sequence[int]) -> dict:
        """Return {layer_idx: (grid[Gh,Gw,C] float32, text_vec[C] float32)}."""
        raise NotImplementedError


class PaliGemmaBackbone(Backbone):
    """google/paligemma2-3b-mix-448  (SigLIP + Gemma-2B, 18 decoder layers)."""

    def load(self):
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._img_token_id = getattr(self.model.config, "image_token_index", None)
        return self

    def encode(self, image, instruction: str, layers):
        import torch
        # PaliGemma expects a task/prompt prefix; the instruction works as prefix.
        inputs = self.processor(
            text=instruction, images=image, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hs = out.hidden_states                       # tuple[(1,S,C)]
        idxs = resolve_layers(layers, len(hs))
        ids = inputs["input_ids"][0]
        img_mask = (ids == self._img_token_id) if self._img_token_id is not None \
            else torch.zeros_like(ids, dtype=torch.bool)
        if img_mask.sum() == 0:                      # fallback: assume image-first block
            n_img = int(getattr(self.processor, "image_seq_length", 1024))
            img_pos = torch.arange(n_img, device=ids.device)
        else:
            img_pos = img_mask.nonzero(as_tuple=True)[0]
        txt_pos = torch.arange(len(ids), device=ids.device)[~torch.isin(
            torch.arange(len(ids), device=ids.device), img_pos)]
        n_img = len(img_pos)
        g = int(round(n_img ** 0.5))
        result = {}
        for idx in idxs:
            h = hs[idx][0]                           # (S, C)
            grid = h[img_pos][: g * g].reshape(g, g, -1).float().cpu().numpy()
            text_vec = h[txt_pos].mean(0).float().cpu().numpy() if len(txt_pos) else \
                grid.reshape(-1, grid.shape[-1]).mean(0)
            result[idx] = (grid, text_vec)
        return result


class Qwen3VLBackbone(Backbone):
    """Qwen/Qwen3-VL-4B-Instruct  (deep ~36-layer decoder, dynamic resolution)."""

    def load(self):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as _Model
        except Exception:                            # older/newer naming
            from transformers import AutoModelForImageTextToText as _Model
        pkw = {}
        if self.min_pixels:
            pkw["min_pixels"] = int(self.min_pixels)
        if self.max_pixels:
            pkw["max_pixels"] = int(self.max_pixels)
        self.processor = AutoProcessor.from_pretrained(self.model_id, **pkw)
        self.model = _Model.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self._img_token_id = getattr(cfg, "image_token_id", None) \
            or getattr(cfg, "image_token_index", None)
        return self

    def encode(self, image, instruction: str, layers):
        import torch
        # Put TEXT BEFORE IMAGE so image tokens can attend to the instruction
        # under causal attention (makes the fused image features language-aware).
        messages = [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image"},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hs = out.hidden_states
        idxs = resolve_layers(layers, len(hs))
        ids = inputs["input_ids"][0]
        img_mask = (ids == self._img_token_id)
        img_pos = img_mask.nonzero(as_tuple=True)[0]
        # grid dims from image_grid_thw (merged patch grid)
        if "image_grid_thw" in inputs:
            thw = inputs["image_grid_thw"][0].tolist()          # [t, h, w] in patches
            merge = int(getattr(self.model.config.vision_config, "spatial_merge_size", 2))
            gh, gw = thw[1] // merge, thw[2] // merge
        else:
            n = int(img_mask.sum())
            gh = gw = int(round(n ** 0.5))
        n_img = gh * gw
        txt_pos = (~img_mask).nonzero(as_tuple=True)[0]
        result = {}
        for idx in idxs:
            h = hs[idx][0]
            grid = h[img_pos][:n_img].reshape(gh, gw, -1).float().cpu().numpy()
            text_vec = h[txt_pos].mean(0).float().cpu().numpy()
            result[idx] = (grid, text_vec)
        return result


def build_backbone(kind: str, model_id: str, device: str,
                   min_pixels=None, max_pixels=None) -> Backbone:
    kind = kind.lower()
    if kind == "paligemma":
        return PaliGemmaBackbone(model_id, device, min_pixels, max_pixels)
    if kind in ("qwen3vl", "qwen"):
        return Qwen3VLBackbone(model_id, device, min_pixels, max_pixels)
    raise ValueError(f"unknown backbone kind: {kind}")


# ---------------------------------------------------------------------------
# Pooling: grid feature within a fractional box -> vector
# ---------------------------------------------------------------------------
def pool_box(grid: np.ndarray, frac_box) -> np.ndarray:
    """Mean-pool grid cells (Gh,Gw,C) covered by fractional box (fu0,fv0,fu1,fv1)."""
    Gh, Gw = grid.shape[:2]
    fu0, fv0, fu1, fv1 = frac_box
    c0 = int(np.floor(fu0 * Gw)); c1 = max(c0 + 1, int(np.ceil(fu1 * Gw)))
    r0 = int(np.floor(fv0 * Gh)); r1 = max(r0 + 1, int(np.ceil(fv1 * Gh)))
    c0, c1 = np.clip([c0, c1], 0, Gw); r0, r1 = np.clip([r0, r1], 0, Gh)
    if c1 <= c0 or r1 <= r0:
        return grid.reshape(-1, grid.shape[-1]).mean(0)
    return grid[r0:r1, c0:c1].reshape(-1, grid.shape[-1]).mean(0)
