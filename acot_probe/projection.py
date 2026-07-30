"""Offline projection of 3D vehicle centers -> image pixels, using the camera
pose stored per frame by the recorder (state/cam_*), reproducing CARLA's camera
convention (see carla_uav_tracking/recording/recorder.py:_point_in_frame).

This lets the layer-16 grounding probe build its candidate set (the target +
every distractor as a box in the image) WITHOUT running CARLA — we only need the
3D positions and camera pose already in the HDF5 file.

Convention (matches CARLA):
  - camera local frame: x=forward, y=right, z=up
  - a world point p projects to  u = f*(y/x) + W/2 ,  v = f*(-z/x) + H/2
    with f = W / (2*tan(fov/2)),  and is visible iff x>~0 and (u,v) in frame.
  - camera world->local matrix = inverse of CARLA's Transform.get_matrix().

NOTE: the data stores 3D CENTERS only (no true 2D bbox extents), so each vehicle
becomes a distance-scaled square pseudo-box. If you later record true actor
bounding boxes at generation time, store them as annotation/cand_uv + extents and
`extract_candidates` will prefer them (see `_candidates_from_annotation`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# CARLA transform matrix
# ---------------------------------------------------------------------------
def carla_matrix(x, y, z, pitch, yaw, roll=0.0) -> np.ndarray:
    """CARLA Transform.get_matrix() (local->world), angles in DEGREES."""
    p, yw, r = math.radians(pitch), math.radians(yaw), math.radians(roll)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(yw), math.sin(yw)
    cr, sr = math.cos(r), math.sin(r)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = cp * cy
    m[0, 1] = cy * sp * sr - sy * cr
    m[0, 2] = -cy * sp * cr - sy * sr
    m[0, 3] = x
    m[1, 0] = cp * sy
    m[1, 1] = sy * sp * sr + cy * cr
    m[1, 2] = -sy * sp * cr + cy * sr
    m[1, 3] = y
    m[2, 0] = sp
    m[2, 1] = -cp * sr
    m[2, 2] = cp * cr
    m[2, 3] = z
    return m


def inverse_pose_matrix(x, y, z, pitch, yaw, roll=0.0) -> np.ndarray:
    """World->local (CARLA get_inverse_matrix())."""
    return np.linalg.inv(carla_matrix(x, y, z, pitch, yaw, roll))


@dataclass
class Candidate:
    """One projected vehicle in a frame."""
    u: float          # pixel center (in ORIGINAL image space, e.g. 336)
    v: float
    w: float          # pseudo-box width/height (px)
    h: float
    depth: float      # forward distance (m)
    is_target: bool
    idx: int          # -1 for target, else distractor index

    def frac_box(self, W: int, H: int) -> tuple[float, float, float, float]:
        """(fu0, fv0, fu1, fv1) fractional box in [0,1], clipped to frame."""
        fu0 = max(0.0, (self.u - self.w / 2) / W)
        fv0 = max(0.0, (self.v - self.h / 2) / H)
        fu1 = min(1.0, (self.u + self.w / 2) / W)
        fv1 = min(1.0, (self.v + self.h / 2) / H)
        return fu0, fv0, fu1, fv1


def project_point(world_xyz, campose, W, H, fov_deg=90.0):
    """Project a world point. campose=(x,y,z,pitch,yaw[,roll]).

    Returns (u, v, depth) in pixels, or None if behind the camera.
    """
    if len(campose) == 5:
        campose = (*campose, 0.0)
    M = inverse_pose_matrix(*campose)
    q = M @ np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    depth = q[0]
    if depth <= 0.1:
        return None
    f = W / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    u = f * (q[1] / depth) + W / 2.0
    v = f * (-q[2] / depth) + H / 2.0
    return u, v, depth


def _pseudo_box(depth, W, fov_deg, vehicle_size_m, box_min_px, box_max_px):
    """Distance-scaled square box (px) for a vehicle of ~vehicle_size_m."""
    f = W / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    size = f * vehicle_size_m / max(depth, 0.1)
    return float(np.clip(size, box_min_px, box_max_px))


def extract_candidates(ep, i: int, cfg: dict) -> list[Candidate]:
    """Build the candidate list for frame `i` of an open h5 file `ep`.

    Target from target/{tx,ty,tz}; distractors from distractors/positions[i] (D,6).
    Only in-frame, in-front candidates are returned. Requires the target to be
    in-frame (else returns [] — that frame is unusable for selection).
    """
    img = cfg["image"]
    W, H, fov = img["width"], img["height"], img["fov_deg"]
    campose = (
        float(ep["state/cam_x"][i]), float(ep["state/cam_y"][i]),
        float(ep["state/cam_z"][i]), float(ep["state/cam_pitch"][i]),
        float(ep["state/cam_yaw"][i]),
    )

    def _mk(world_xyz, is_target, idx):
        pr = project_point(world_xyz, campose, W, H, fov)
        if pr is None:
            return None
        u, v, depth = pr
        if not (0 <= u < W and 0 <= v < H):
            return None
        s = _pseudo_box(depth, W, fov, img["vehicle_size_m"],
                        img["box_min_px"], img["box_max_px"])
        return Candidate(u, v, s, s, depth, is_target, idx)

    target = _mk(
        (float(ep["target/tx"][i]), float(ep["target/ty"][i]), float(ep["target/tz"][i])),
        True, -1,
    )
    if target is None:                       # target off-screen -> unusable frame
        return []

    cands = [target]
    if "distractors/positions" in ep:
        dpos = ep["distractors/positions"][i]     # (D, 6)
        for d in range(dpos.shape[0]):
            c = _mk((dpos[d, 0], dpos[d, 1], dpos[d, 2]), False, d)
            if c is not None:
                cands.append(c)
    return cands
