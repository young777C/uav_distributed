"""Camera-inside-building check (2026-10-11; read-only, offline — needs no CARLA server).

The kinematic drone has no collision and scenario mvp_moving_fast(_small) runs without uav.geometry_avoidance, so the
camera sometimes passes through buildings (whole frame = concrete / facade while the geometric labels still give a
target box). Measured on mvp_v5_fov90_fast: 0.77 % of frames, concentrated in Town05 / Town10HD.

Building boxes = CARLA world.get_level_bbs(CityObjectLabel.Buildings) exported once per town to
/nvidia/hque/data/carla_data/building_bbs/<town>.json ([cx, cy, cz, ex, ey, ez, yaw_deg] per box). A frame is "inside"
when the recorded camera position (state/cam_x, cam_y, cam_z) lies in any oriented box. Verified visually (seed 355:
inside a parking structure, seed 84: through a facade / building under construction).

    from recording.building_mask import building_keep_mask, train_keep_mask
    keep = train_keep_mask("episode_000000.h5")     # unjam (in-view relocations) AND building mask, ±1 s each
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

from recording.unjam_mask import _opened, unjam_keep_mask

BBS_DIR = os.environ.get("BUILDING_BBS_DIR", "/nvidia/hque/data/carla_data/building_bbs")
MAX_INSIDE_FRAC = 0.05          # QC v3: episodes with > 5 % camera-inside-building frames are rejected


@lru_cache(maxsize=None)
def load_boxes(town: str, kind: str = "Buildings") -> np.ndarray:
    B = np.asarray(json.load(open(os.path.join(BBS_DIR, f"{town}.json")))[kind], dtype=float).reshape(-1, 7)
    B[:, 3:6] = np.abs(B[:, 3:6])        # codex R1-3: some exported level boxes carry negative extents
    return B


def camera_inside(P: np.ndarray, town: str, margin: float = 0.0) -> np.ndarray:
    """P (T, 3) world camera positions → bool (T,) inside any oriented building box"""
    B = load_boxes(town)
    P = np.asarray(P, dtype=float).reshape(-1, 3)
    out = np.zeros(len(P), dtype=bool)
    for i in range(0, len(B), 2000):
        b = B[i:i + 2000]
        d = P[:, None, :] - b[None, :, :3]
        c, s = np.cos(np.radians(b[:, 6])), np.sin(np.radians(b[:, 6]))
        lx = d[..., 0] * c + d[..., 1] * s
        ly = -d[..., 0] * s + d[..., 1] * c
        out |= ((np.abs(lx) <= b[:, 3] + margin) & (np.abs(ly) <= b[:, 4] + margin)
                & (np.abs(d[..., 2]) <= b[:, 5] + margin)).any(1)
    return out


def episode_inside(h5) -> np.ndarray:
    with _opened(h5) as f:
        town = str(f.attrs["town"])
        P = np.stack([np.asarray(f[k]) for k in ("state/cam_x", "state/cam_y", "state/cam_z")], 1)
    return camera_inside(P, town)


def building_keep_mask(h5, pad_s: float = 1.0, fps: int = 10) -> np.ndarray:
    """bool[n_frames]: False for frames with the camera inside a building, dilated by ±pad_s"""
    m = episode_inside(h5)
    pad = int(round(pad_s * fps))
    if pad and m.any():
        m = np.convolve(m.astype(int), np.ones(2 * pad + 1, dtype=int), mode="same") > 0
    return ~m


def train_keep_mask(h5, pad_s: float = 1.0) -> np.ndarray:
    """frames usable for training = not near an in-view unjam relocation AND camera not (near) inside a building"""
    with _opened(h5) as f:
        return unjam_keep_mask(f, pad_s=pad_s) & building_keep_mask(f, pad_s=pad_s)
