"""Post-processing: compute occlusion levels, 2D bboxes, EAR waypoint GT, and search_mode labels.

Run after raw data collection to add training annotations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# Correct offline projection (ported verbatim from acot_probe/projection.py so the
# training labels match the layer-16 grounding probe). Uses the per-frame camera
# pose stored by the recorder (state/cam_*) incl. pitch & yaw, and the real image
# size — replaces the old buggy 224-hardcoded / yaw-only projection.
# ---------------------------------------------------------------------------
_FOV_DEG = 90.0
_VEHICLE_SIZE_M = 4.0        # pseudo-box scale (data stores 3D centres only)
_BOX_MIN_PX = 8.0
_BOX_MAX_PX = 140.0


def _carla_matrix(x, y, z, pitch, yaw, roll=0.0) -> np.ndarray:
    """CARLA Transform.get_matrix() (local->world), angles in DEGREES."""
    p, yw, r = math.radians(pitch), math.radians(yaw), math.radians(roll)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(yw), math.sin(yw)
    cr, sr = math.cos(r), math.sin(r)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = cp * cy; m[0, 1] = cy * sp * sr - sy * cr; m[0, 2] = -cy * sp * cr - sy * sr; m[0, 3] = x
    m[1, 0] = cp * sy; m[1, 1] = sy * sp * sr + cy * cr; m[1, 2] = -sy * sp * cr + cy * sr; m[1, 3] = y
    m[2, 0] = sp;      m[2, 1] = -cp * sr;               m[2, 2] = cp * cr;                 m[2, 3] = z
    return m


def _project(world_xyz, campose, W, H, fov=_FOV_DEG):
    """Project a world point with campose=(x,y,z,pitch,yaw). Returns (u,v,depth) in
    pixels, or None if behind the camera."""
    M = np.linalg.inv(_carla_matrix(*campose))
    q = M @ np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    depth = q[0]
    if depth <= 0.1:
        return None
    f = W / (2.0 * math.tan(math.radians(fov) / 2.0))
    return f * (q[1] / depth) + W / 2.0, f * (-q[2] / depth) + H / 2.0, depth


def _pseudo_box(depth, W, fov=_FOV_DEG) -> float:
    f = W / (2.0 * math.tan(math.radians(fov) / 2.0))
    return float(np.clip(f * _VEHICLE_SIZE_M / max(depth, 0.1), _BOX_MIN_PX, _BOX_MAX_PX))


# Threshold on structural occlusion above which the target counts as "not usefully
# visible" (majority of target→camera rays blocked) → mask the ACTION imitation loss.
_OCCLUDED_THRESH = 0.5


def compute_occlusion_labels(bbox_u, bbox_v, bbox_w, occ_raycast, W, H,
                             occluded_thresh: float = _OCCLUDED_THRESH):
    """Split occlusion into THREE non-overlapping signals (A1c ②), so the IAR
    occlusion head learns structural occlusion and NOT "out of frame":

      occ_structural [0,1] : target IS in frame but hidden by geometry
                             (occ_raycast gated to in-frame). 0 when off-screen.
      off_screen     {0,1} : target is behind the camera OR projected outside the
                             image — a DISTINCT signal (drives search/re-acquisition).
      occluded       {0,1} : "target not usefully visible" mask =
                             (occ_structural >= thresh) OR off_screen. Use it to MASK
                             the action imitation loss (expert used privileged GT the
                             policy can't see); the EAR waypoint loss stays UNmasked so
                             the model still learns to predict the re-emergence point.

    All inputs/outputs are 1-D arrays of length n; outputs are float32 (0/1 or [0,1]).
    """
    bbox_u = np.asarray(bbox_u, dtype=np.float32)
    bbox_v = np.asarray(bbox_v, dtype=np.float32)
    bbox_w = np.asarray(bbox_w, dtype=np.float32)
    behind = bbox_w == 0.0                                   # (-1,-1,0,0) sentinel
    in_frame = (bbox_u >= 0) & (bbox_u < W) & (bbox_v >= 0) & (bbox_v < H) & ~behind
    off_screen = (~in_frame).astype(np.float32)              # behind OR outside frame
    ray = (np.asarray(occ_raycast, dtype=np.float32)
           if occ_raycast is not None else np.zeros_like(bbox_u))
    occ_structural = np.where(in_frame, ray, 0.0).astype(np.float32)
    occluded = ((occ_structural >= occluded_thresh) | (off_screen > 0)).astype(np.float32)
    return occ_structural, off_screen, occluded


class PostProcessor:
    """Add training annotations to raw recording data."""

    def __init__(self, output_dir: Path, fov: float = _FOV_DEG):
        self._output_dir = Path(output_dir)
        # §10: the sensor FOV was narrowed 90→70; bbox/off_screen projection MUST use the
        # SAME fov as the camera (else off_screen under-reports — the target leaves the real
        # 70° frame but a 90° projection still counts it in-frame). Caller passes output.fov.
        self._fov = float(fov)

    def process_episode(self, episode_path: Path) -> None:
        """Add annotations to one episode HDF5 file."""
        with h5py.File(episode_path, "r+") as f:
            n_steps = f["rgb"].shape[0]

            # 1. Correct-projection 2D bboxes: TARGET + ALL DISTRACTORS
            bboxes = self._compute_bboxes(f, n_steps, self._fov)               # (n, 4)
            dbboxes = self._compute_distractor_bboxes(f, n_steps, self._fov)   # (n, D, 4)

            # 2. Occlusion (from the correct target bbox)
            occlusions = self._compute_occlusions(f, n_steps, bboxes)

            # 2b. Separated occlusion labels (A1c ②): structural (in-frame, hidden by
            #     geometry) vs off-screen vs the "occluded" action-loss mask.
            H, W = int(f["rgb"].shape[1]), int(f["rgb"].shape[2])
            ray = (f["state/occ_raycast"][:n_steps].astype(np.float32)
                   if "state/occ_raycast" in f else None)
            occ_struct, off_screen, occluded = compute_occlusion_labels(
                bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], ray, W, H)

            # 3. Compute EAR waypoint ground truth (unchanged — no recompute needed)
            waypoints, wp_sigmas = self._compute_waypoints(f, n_steps)

            # 4. Compute search_mode ground truth (from corrected occlusion)
            search_modes = self._compute_search_modes(occlusions, n_steps)

            # Write annotations
            self._write_dataset(f, "annotation/bbox_u", bboxes[:, 0])
            self._write_dataset(f, "annotation/bbox_v", bboxes[:, 1])
            self._write_dataset(f, "annotation/bbox_w", bboxes[:, 2])
            self._write_dataset(f, "annotation/bbox_h", bboxes[:, 3])
            self._write_dataset(f, "annotation/occlusion", occlusions)
            # A1c ② — separated, non-overlapping occlusion signals.
            self._write_dataset(f, "annotation/occ_structural", occ_struct)
            self._write_dataset(f, "annotation/off_screen", off_screen)
            self._write_dataset(f, "annotation/occluded", occluded)
            self._write_dataset(f, "annotation/search_mode", search_modes)
            # Distractor 2D bboxes (n, D, 4) [u,v,w,h] for Stage-2 target-ID loss.
            self._write_dataset(f, "distractors/bbox", dbboxes)

            # Waypoints: shape (n_steps, 3, 3) → (n_steps, 9)
            wp_flat = waypoints.reshape(n_steps, -1)
            self._write_dataset(f, "annotation/waypoints", wp_flat)
            self._write_dataset(f, "annotation/waypoint_sigma", wp_sigmas)

            # Finalize the language with the target's ACTUAL early motion (accurate
            # "about to turn left / stop / accelerate"), derived from the trajectory.
            self._finalize_language(f, n_steps)

    # ------------------------------------------------------------------
    # Annotation computation
    # ------------------------------------------------------------------

    @staticmethod
    def _campose(f, i):
        return (float(f["state/cam_x"][i]), float(f["state/cam_y"][i]),
                float(f["state/cam_z"][i]), float(f["state/cam_pitch"][i]),
                float(f["state/cam_yaw"][i]))

    @staticmethod
    def _compute_bboxes(f: h5py.File, n: int, fov: float = _FOV_DEG) -> np.ndarray:
        """TARGET 2D pseudo-bbox [u, v, w, h] (px) via the CORRECT projection (real
        camera pose incl. pitch/yaw + actual W,H + sensor fov). [-1,-1,0,0] if behind."""
        H, W = int(f["rgb"].shape[1]), int(f["rgb"].shape[2])
        tx, ty, tz = f["target/tx"][:], f["target/ty"][:], f["target/tz"][:]
        out = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            pr = _project((tx[i], ty[i], tz[i]), PostProcessor._campose(f, i), W, H, fov)
            if pr is None:
                out[i] = (-1, -1, 0, 0)
                continue
            u, v, depth = pr
            s = _pseudo_box(depth, W, fov)
            out[i] = (u, v, s, s)
        return out

    @staticmethod
    def _compute_distractor_bboxes(f: h5py.File, n: int, fov: float = _FOV_DEG) -> np.ndarray:
        """ALL distractors' 2D pseudo-bbox → (n, D, 4) [u,v,w,h] (px) for Stage-2
        L_target_id. [-1,-1,0,0] where a distractor is behind the camera."""
        if "distractors/positions" not in f:
            return np.zeros((n, 0, 4), dtype=np.float32)
        H, W = int(f["rgb"].shape[1]), int(f["rgb"].shape[2])
        dpos = f["distractors/positions"][:, :, :3]      # (n, D, 3)
        D = dpos.shape[1]
        out = np.zeros((n, D, 4), dtype=np.float32)
        out[..., 0:2] = -1.0
        for i in range(n):
            cp = PostProcessor._campose(f, i)
            for d in range(D):
                pr = _project((dpos[i, d, 0], dpos[i, d, 1], dpos[i, d, 2]), cp, W, H, fov)
                if pr is None:
                    continue
                u, v, depth = pr
                s = _pseudo_box(depth, W, fov)
                out[i, d] = (u, v, s, s)
        return out

    @staticmethod
    def _compute_occlusions(f: h5py.File, n: int, bboxes: np.ndarray) -> np.ndarray:
        """Occlusion in [0,1], combining:
          - view geometry from the correct bbox: 1.0 behind camera (w==0), 0.5 in-front
            but out-of-frame, 0.0 in-frame; and
          - STRUCTURAL occlusion (state/occ_raycast): graded fraction of target→camera
            rays blocked by buildings/tunnel roof while the target is still in frame
            (A1b — the label §6.2/§6.4/H2 need). Absent on pre-A1b data → geometry only.
        """
        H, W = int(f["rgb"].shape[1]), int(f["rgb"].shape[2])
        u, v, w = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2]
        behind = w == 0.0                       # (-1,-1,0,0) sentinel
        in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H) & ~behind
        occ = np.zeros(n, dtype=np.float32)
        occ[behind] = 1.0
        occ[~in_frame & ~behind] = 0.5
        if "state/occ_raycast" in f:
            occ = np.maximum(occ, f["state/occ_raycast"][:n].astype(np.float32))
        return occ

    @staticmethod
    def _compute_waypoints(f: h5py.File, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute EAR waypoint ground truth: future target positions at +2s, +4s, +6s.

        Returns:
            waypoints: (n, 3, 3) → [t+2s, t+4s, t+6s] × [x, y, z]
            sigmas: (n, 3) → uncertainty per waypoint
        """
        fps = 10
        offsets = [2 * fps, 4 * fps, 6 * fps]  # 20, 40, 60 steps

        waypoints = np.zeros((n, 3, 3), dtype=np.float32)
        sigmas = np.zeros((n, 3), dtype=np.float32)

        for i in range(n):
            for j, offset in enumerate(offsets):
                future_idx = min(i + offset, n - 1)
                tx = f["target/tx"][future_idx]
                ty = f["target/ty"][future_idx]
                tz = f["target/tz"][future_idx]

                # Add Kalman-style smoothing
                if i > 0 and future_idx < n - 1:
                    window = f["target/tx"][i:future_idx+1]
                    tx = float(np.mean(window))
                    window = f["target/ty"][i:future_idx+1]
                    ty = float(np.mean(window))
                    window = f["target/tz"][i:future_idx+1]
                    tz = float(np.mean(window))

                waypoints[i, j] = [tx, ty, tz]
                # Uncertainty grows with prediction horizon
                sigmas[i, j] = 1.0 + 2.0 * (j + 1)

        return waypoints, sigmas

    @staticmethod
    def _compute_search_modes(occlusions: np.ndarray, n: int) -> np.ndarray:
        """Compute search_mode ground truth from occlusion level + time-since-lost.

        search_mode ∈ [0, 1]:
          0 = precise tracking (target fully visible)
          1 = wide-area search (target lost for long duration)
        """
        search_modes = np.zeros(n, dtype=np.float32)
        time_since_lost = 0.0

        for i in range(n):
            if occlusions[i] > 0.8:
                time_since_lost += 0.1  # dt = 0.1s
            else:
                time_since_lost = max(0.0, time_since_lost - 0.2)  # decay

            search_modes[i] = float(np.clip(
                occlusions[i] * (1.0 + time_since_lost / 5.0), 0.0, 1.0
            ))

        return search_modes

    @staticmethod
    def _detect_maneuvers(f: h5py.File, n: int) -> list[dict]:
        """Detect maneuver events from target velocity/acceleration changes."""
        events = []
        tspeed = f["target/tspeed"][:]
        tyaw = f["target/tyaw"][:]

        for i in range(1, n - 1):
            accel = tspeed[i] - tspeed[i - 1]
            yaw_change = abs(tyaw[i] - tyaw[i - 1])

            if accel > 3.0:
                events.append({"t": i * 0.1, "type": "ACCELERATION", "mag": float(accel)})
            elif accel < -5.0:  # stronger threshold for braking
                events.append({"t": i * 0.1, "type": "SUDDEN_STOP", "mag": float(accel)})
            elif yaw_change > 15.0:  # deg/s
                events.append({"t": i * 0.1, "type": "SHARP_TURN", "mag": float(yaw_change)})

        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finalize_language(f, n: int) -> None:
        """Rewrite attrs['language'] using the target's actual early trajectory so the
        motion clause ('about to turn left', 'about to stop', ...) is accurate."""
        import random as _r
        from scene.language_generator import (
            LanguageGenerator, spatial_and_distance, motion_phrase)
        a = f.attrs
        desc = a.get("target_desc", "")
        if not desc:
            return
        num_similar = int(a.get("num_similar", 0))
        w = min(50, n - 1)                          # ~5s @ 10Hz
        start = 10 if w > 15 else 0                 # skip spawn-alignment jitter
        try:
            motion = motion_phrase(f["target/tyaw"][start:w + 1],
                                   f["target/tspeed"][start:w + 1])
        except Exception:
            motion = ""
        try:
            spatial, dist_m = spatial_and_distance(
                float(f["state/uav_x"][0]), float(f["state/uav_y"][0]),
                float(f["state/uav_yaw"][0]),
                float(f["target/tx"][0]), float(f["target/ty"][0]),
                uz=float(f["state/uav_z"][0]), tz=float(f["target/tz"][0]))
        except Exception:
            spatial, dist_m = "", None
        seed_str = str(a.get("seed", "0"))
        seed = int(seed_str) if seed_str.lstrip("-").isdigit() else 0
        a["language"] = LanguageGenerator().build_instruction(
            desc, num_similar, spatial=spatial, motion=motion,
            distance_m=dist_m, rng=_r.Random(seed))

    @staticmethod
    def _write_dataset(f: h5py.File, name: str, data: np.ndarray) -> None:
        if name in f:
            del f[name]
        f.create_dataset(name, data=data.astype(np.float32), compression="gzip",
                         compression_opts=4)
