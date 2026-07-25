"""Post-processing: compute occlusion levels, 2D bboxes, EAR waypoint GT, and search_mode labels.

Run after raw data collection to add training annotations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d


class PostProcessor:
    """Add training annotations to raw recording data."""

    def __init__(self, output_dir: Path):
        self._output_dir = Path(output_dir)

    def process_episode(self, episode_path: Path) -> None:
        """Add annotations to one episode HDF5 file."""
        with h5py.File(episode_path, "r+") as f:
            n_steps = f["rgb"].shape[0]

            # 1. Compute 2D bboxes from 3D target positions + camera params
            bboxes = self._compute_bboxes(f, n_steps)

            # 2. Compute occlusion levels
            occlusions = self._compute_occlusions(f, n_steps)

            # 3. Compute EAR waypoint ground truth
            waypoints, wp_sigmas = self._compute_waypoints(f, n_steps)

            # 4. Compute search_mode ground truth
            search_modes = self._compute_search_modes(occlusions, n_steps)

            # 5. Detect maneuver events from target velocity changes
            maneuver_events = self._detect_maneuvers(f, n_steps)

            # Write annotations
            self._write_dataset(f, "annotation/bbox_u", bboxes[:, 0])
            self._write_dataset(f, "annotation/bbox_v", bboxes[:, 1])
            self._write_dataset(f, "annotation/bbox_w", bboxes[:, 2])
            self._write_dataset(f, "annotation/bbox_h", bboxes[:, 3])
            self._write_dataset(f, "annotation/occlusion", occlusions)
            self._write_dataset(f, "annotation/search_mode", search_modes)

            # Waypoints: shape (n_steps, 3, 3) → (n_steps, 9)
            wp_flat = waypoints.reshape(n_steps, -1)
            self._write_dataset(f, "annotation/waypoints", wp_flat)
            self._write_dataset(f, "annotation/waypoint_sigma", wp_sigmas)

    # ------------------------------------------------------------------
    # Annotation computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_bboxes(f: h5py.File, n: int) -> np.ndarray:
        """Compute 2D bounding boxes by projecting 3D target position + extent.

        Simplified: assumes a fixed target size and projects to image plane.
        Returns (n, 4): [u_center, v_center, width, height].
        """
        bboxes = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            tx = f["target/tx"][i]
            ty = f["target/ty"][i]
            tz = f["target/tz"][i]
            uav_x = f["state/uav_x"][i]
            uav_y = f["state/uav_y"][i]
            uav_z = f["state/uav_z"][i]
            uav_yaw = f["state/uav_yaw"][i]

            # Simple pinhole projection (assuming camera at UAV, facing forward-down)
            # Transform target to UAV body frame
            dx = tx - uav_x
            dy = ty - uav_y
            dz = tz - uav_z

            yaw_rad = np.radians(uav_yaw)
            cos_y = np.cos(-yaw_rad)
            sin_y = np.sin(-yaw_rad)

            # Target in camera frame (forward = x, right = y, down = z)
            cam_x = dx * cos_y - dy * sin_y
            cam_y = dx * sin_y + dy * cos_y
            cam_z = -dz  # UAV looks down

            if cam_x < 0.1:  # behind camera
                bboxes[i] = [-1, -1, 0, 0]
                continue

            # Perspective projection (224×224, 90° FOV)
            fov = np.radians(90.0)
            f_px = 112.0 / np.tan(fov / 2)

            u = 112.0 + f_px * cam_y / cam_x
            v = 112.0 + f_px * cam_z / cam_x

            # Fixed-size bbox scaled by distance
            size = 50.0 / max(cam_x, 0.1)  # 50px at 1m distance
            bboxes[i] = [u, v, size, size]

        return bboxes

    @staticmethod
    def _compute_occlusions(f: h5py.File, n: int) -> np.ndarray:
        """Estimate occlusion level from bbox validity + heuristics.

        0 = fully visible, 1 = fully occluded.
        Simplified: uses bbox validity as proxy (negative → behind camera).
        For production: replace with raycast results (requires CARLA world access).
        """
        bboxes = PostProcessor._compute_bboxes(f, n)
        occlusions = np.zeros(n, dtype=np.float32)
        # When bbox_u < 0 → target is behind camera
        occlusions[bboxes[:, 0] < 0] = 1.0
        # When bbox is outside image bounds → partially occluded
        in_frame = (
            (bboxes[:, 0] >= 0) & (bboxes[:, 0] < 224) &
            (bboxes[:, 1] >= 0) & (bboxes[:, 1] < 224)
        )
        occlusions[~in_frame] = np.maximum(occlusions[~in_frame], 0.5)
        return np.clip(occlusions, 0.0, 1.0)

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
    def _write_dataset(f: h5py.File, name: str, data: np.ndarray) -> None:
        if name in f:
            del f[name]
        f.create_dataset(name, data=data.astype(np.float32), compression="gzip",
                         compression_opts=4)
