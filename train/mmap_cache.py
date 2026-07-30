"""Convert an existing Stage-2 multilayer cache (fat .npz with `ctx`) to the
mmap-ready format: a slim .npz (frames/grid/layers) + an uncompressed
`<stem>.ctx.npy` (the big (F,L,T,C) array, memory-mappable and shared across
processes via the OS page cache). Idempotent; skips already-slim shards.

    python -m train.mmap_cache runs/ctx_cache_ml_neutral [more_dirs...]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np


def convert_dir(d: str) -> None:
    files = sorted(glob.glob(str(Path(d) / "episode_*.npz")))
    done = 0
    for f in files:
        p = Path(f)
        ctx_npy = p.with_name(p.stem + ".ctx.npy")
        z = np.load(f)
        if "ctx" not in z.files:            # already slim
            z.close(); continue
        frames = np.asarray(z["frames"]); grid = np.asarray(z["grid"])
        layers = np.asarray(z["layers"]) if "layers" in z.files else np.array([], np.int64)
        if not ctx_npy.exists():
            np.save(ctx_npy, np.asarray(z["ctx"], dtype=np.float32))
        z.close()
        np.savez(f, frames=frames, grid=grid, layers=layers)   # rewrite slim (reclaim space)
        done += 1
    print(f"[mmap_cache] {d}: converted {done} / {len(files)} shards")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m train.mmap_cache <cache_dir> [more...]")
    for d in sys.argv[1:]:
        convert_dir(d)
