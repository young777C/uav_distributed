"""Precompute + cache the frozen VLM's layer-L context tokens per frame.

EAR/IAR read the frozen backbone's per-layer representation via cross-attention.
Per design principle 1, we run the VLM ONCE over the dataset and cache the layer-L
image tokens to disk, so Stage-1/2 training never forwards the 3B VLM in the loop.

Reuses acot_probe.backbones (run from the uav-acot-track/ root so the import
resolves). Only run AFTER data generation is complete and a working GPU is present.

    python -m train.backbone_kv --config train/config.yaml
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import yaml


_POS_RE = re.compile(
    r"\b(directly |straight )?(ahead|in front|front[- ]?left|front[- ]?right|front[- ]?center|"
    r"to the (?:left|right)|on the (?:left|right)|in the front[- ]?(?:left|right|center))\b",
    re.I)


def neutralize_language(lang: str) -> str:
    """Ablation A: strip TIME-VARYING clauses (spatial bearing, intent, distance) from
    the per-episode instruction, keeping only the time-invariant target IDENTITY. e.g.
    'Keep tracking the yellow Jeep Wrangler directly ahead, which is about to turn
    right, maintaining a tracking distance of about 50 m.' -> 'Keep tracking the
    yellow Jeep Wrangler.'  Everything after the first comma (intent/distance) is
    dropped; leading positional phrases are removed. Identity (color+make) survives."""
    s = lang.split(",")[0]                       # drop ", which is about to ..., maintaining ..."
    s = _POS_RE.sub("", s)                        # drop "directly ahead" / "in the front-left" ...
    s = re.sub(r"\s{2,}", " ", s).strip().rstrip(".").strip()
    return s + "."


def _pool_grid(grid: np.ndarray, g: int) -> np.ndarray:
    """(Gh,Gw,C) -> (g*g, C) by average pooling to a g×g grid (cache-size control)."""
    Gh, Gw, C = grid.shape
    if Gh <= g and Gw <= g:
        return grid.reshape(-1, C)
    import torch
    t = torch.from_numpy(grid).permute(2, 0, 1)[None]         # (1,C,Gh,Gw)
    t = torch.nn.functional.adaptive_avg_pool2d(t, (g, g))
    return t[0].permute(1, 2, 0).reshape(-1, C).numpy()


def precompute(cfg, shard=0, num_shards=1, device=None):
    import h5py
    from acot_probe.backbones import build_backbone

    b = cfg["backbone"]
    layer = int(b.get("layer", 16))
    g = int(b.get("ctx_grid", 8))
    stride = int(cfg["data"].get("frame_stride", 3))
    cap = int(cfg["data"].get("max_frames_per_episode", 200))
    out_dir = Path(cfg["data"]["context_cache"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bk = build_backbone(b["kind"], b["model_id"], (device or b.get("device", "cuda:0")),
                        min_pixels=b.get("min_pixels"), max_pixels=b.get("max_pixels")).load()
    try:
        for ep in sorted(glob.glob(cfg["data"]["episodes_glob"]))[shard::num_shards]:
            out = out_dir / (Path(ep).stem + ".npz")
            if out.exists():
                continue
            with h5py.File(ep, "r") as f:
                n = f["rgb"].shape[0]
                lang = f.attrs.get("language", "")
                lang = lang.decode() if isinstance(lang, bytes) else str(lang)
                if bool(b.get("no_language", False)):
                    lang = "Track the target vehicle."   # ablation: strip ALL target identity
                elif bool(b.get("neutralize_language", False)):
                    lang = neutralize_language(lang)
                frames = list(range(0, n, stride))[:cap]
                ctx = []
                from PIL import Image
                for i in frames:
                    img = Image.fromarray(np.asarray(f["rgb"][i]))
                    feats = bk.encode(img, lang, [layer])
                    grid = next(iter(feats.values()))[0]        # (Gh,Gw,C)
                    ctx.append(_pool_grid(grid, g))
            np.savez_compressed(out, frames=np.array(frames, np.int64),
                                ctx=np.stack(ctx).astype(np.float32))
            print(f"[ctx] {out.name}: {len(frames)} frames, tokens={ctx[0].shape}")
    finally:
        bk.free()


def precompute_multilayer(cfg, layers, shard=0, num_shards=1, device=None):
    """Stage-2 cache: store SEVERAL layers per frame for IAR.
    Output npz: frames, grid=(g,g), layers, ctx=(F, L, g*g, C)."""
    import h5py
    from acot_probe.backbones import build_backbone

    b = cfg["backbone"]
    g = int(b.get("ctx_grid", 8))
    stride = int(cfg["data"].get("frame_stride", 3))
    cap = int(cfg["data"].get("max_frames_per_episode", 200))
    out_dir = Path(cfg["data"]["stage2_context_cache"])
    out_dir.mkdir(parents=True, exist_ok=True)

    bk = build_backbone(b["kind"], b["model_id"], (device or b.get("device", "cuda:0")),
                        min_pixels=b.get("min_pixels"), max_pixels=b.get("max_pixels")).load()
    try:
        from PIL import Image
        for ep in sorted(glob.glob(cfg["data"]["episodes_glob"]))[shard::num_shards]:
            out = out_dir / (Path(ep).stem + ".npz")
            if out.exists():
                continue
            with h5py.File(ep, "r") as f:
                n = f["rgb"].shape[0]
                lang = f.attrs.get("language", "")
                lang = lang.decode() if isinstance(lang, bytes) else str(lang)
                if bool(b.get("no_language", False)):
                    lang = "Track the target vehicle."   # ablation: strip ALL target identity
                elif bool(b.get("neutralize_language", False)):
                    lang = neutralize_language(lang)
                frames = list(range(0, n, stride))[:cap]
                per_frame, used = [], None
                for i in frames:
                    feats = bk.encode(Image.fromarray(np.asarray(f["rgb"][i])), lang, layers)
                    used = sorted(feats)
                    per_frame.append(np.stack([_pool_grid(feats[L][0], g) for L in used]))
            if not per_frame or used is None:
                continue
            ctx_arr = np.stack(per_frame).astype(np.float16)         # fp16: half disk, VLM feats tolerate it
            np.save(out.with_name(out.stem + ".ctx.npy"), ctx_arr)   # uncompressed -> mmap-ready
            np.savez(out, frames=np.array(frames, np.int64),         # slim meta (no ctx)
                     grid=np.array([g, g], np.int64),
                     layers=np.array(used, np.int64))
            print(f"[ctx-ml] {out.name}: {len(frames)}f × {len(used)}L, tokens={g*g} (+mmap npy)")
    finally:
        bk.free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="multi-layer Stage-2 cache (e.g. --layers 4 8 12 16)")
    ap.add_argument("--shard", type=int, default=0, help="this process's shard index")
    ap.add_argument("--num-shards", type=int, default=1, help="total shards (GPUs)")
    ap.add_argument("--device", default=None, help="override backbone device, e.g. cuda:2")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.layers:
        precompute_multilayer(cfg, args.layers, args.shard, args.num_shards, args.device)
    else:
        precompute(cfg, args.shard, args.num_shards, args.device)


if __name__ == "__main__":
    main()
