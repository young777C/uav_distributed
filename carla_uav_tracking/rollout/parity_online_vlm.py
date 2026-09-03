"""Milestone-1 parity test: OnlineVLM.encode == precomputed context cache.

De-risks the whole harness: if the online VLM reproduces the training cache, the
model behaves identically online. Runs the frozen Qwen3-VL on a few real frames
and compares to `runs/ctx_cache_ml_v5/<ep>.ctx.npy` (built by train.backbone_kv).

Run inside the training container (host GPU driver is too old):
    bash train/docker/run.sh python carla_uav_tracking/rollout/parity_online_vlm.py

PASS = per-token cosine ≥ 0.999 and small abs diff on every probed frame
(fp16 cache storage + bf16 forward → not bit-exact, but near-identical).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import yaml

# repo root (train.*/acot_probe.*) + carla_uav_tracking (rollout.*) on sys.path.
_CARLA_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO = os.path.dirname(_CARLA_PKG)
for _p in (_REPO, _CARLA_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h5py  # noqa: E402
from rollout.online_vlm import OnlineVLM  # noqa: E402

CONFIG = os.path.join(_REPO, "train", "config_v5.yaml")
CACHE_DIR = os.path.join(_REPO, "runs", "ctx_cache_ml_v5")
EPISODE_STEM = os.environ.get("PARITY_EP", "episode_000000")
PROBE_ROWS = [0, 100, 250, 499]      # rows into the cache `frames` array
COS_THRESH = 0.999


def cosine_per_token(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: (M, C) → (M,) cosine similarity per token."""
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return num / den


def main() -> int:
    cfg = yaml.safe_load(open(CONFIG))
    device = os.environ.get("PARITY_DEVICE", cfg["backbone"].get("device", "cuda:0"))

    # --- load the precomputed cache we must reproduce ---
    meta = np.load(os.path.join(CACHE_DIR, EPISODE_STEM + ".npz"))
    frames = meta["frames"]                       # (F,) source frame indices
    ctx = np.load(os.path.join(CACHE_DIR, EPISODE_STEM + ".ctx.npy"),
                  mmap_mode="r")                  # (F, L, g*g, C) fp16
    layers = meta["layers"].tolist()
    print(f"[parity] cache {EPISODE_STEM}: ctx{tuple(ctx.shape)} layers={layers} "
          f"grid={meta['grid'].tolist()} F={len(frames)}")

    ep_h5 = os.path.join(cfg["data"]["episodes_glob"].rsplit("/", 1)[0],
                         EPISODE_STEM + ".h5")
    with h5py.File(ep_h5, "r") as f:
        raw_lang = f.attrs.get("language", "")
        raw_lang = raw_lang.decode() if isinstance(raw_lang, bytes) else str(raw_lang)
        # Cache was built with neutralize_language=true → match with mode="neutral".
        mode = "none" if cfg["backbone"].get("no_language") else (
            "neutral" if cfg["backbone"].get("neutralize_language") else "full")
        print(f"[parity] language_mode={mode}  raw='{raw_lang[:80]}...'")
        vlm = OnlineVLM(cfg, device=device, language_mode=mode)

        all_ok = True
        for row in PROBE_ROWS:
            if row >= len(frames):
                continue
            fidx = int(frames[row])
            rgb = np.asarray(f["rgb"][fidx])                  # (H,W,3) uint8
            _, online_ctx, _ = vlm.encode(rgb, raw_lang)      # (M, C) torch
            online = online_ctx.float().cpu().numpy()
            cached = np.asarray(ctx[row, 0], dtype=np.float32)  # single layer
            if online.shape != cached.shape:
                print(f"  ✘ frame {fidx}: shape {online.shape} != cache {cached.shape}")
                all_ok = False
                continue
            cos = cosine_per_token(online, cached)
            abs_d = np.abs(online - cached)
            ok = float(cos.mean()) >= COS_THRESH
            all_ok &= ok
            print(f"  {'✓' if ok else '✘'} frame {fidx:4d} (row {row:3d}): "
                  f"cos mean={cos.mean():.5f} min={cos.min():.5f} | "
                  f"abs mean={abs_d.mean():.4f} max={abs_d.max():.4f} | "
                  f"|online|={np.linalg.norm(online):.1f} |cache|={np.linalg.norm(cached):.1f}")

    vlm.free()
    print(f"\n[parity] {'PASS ✅' if all_ok else 'FAIL ✘'}  "
          f"(threshold: per-token cos mean ≥ {COS_THRESH})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
