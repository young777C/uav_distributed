"""Occlusion detection via raycast from target to UAV."""

from __future__ import annotations

import math

import carla
import numpy as np


class OcclusionDetector:
    """Raycast-based occlusion detection between target and UAV.

    Projects target bounding box corners as ray origins toward the UAV camera.
    If ≥ threshold fraction of rays are blocked → target is occluded.

    Usage:
        detector = OcclusionDetector(world, num_rays=8, threshold=0.5)
        is_occluded, level = detector.check(target_actor, uav_position)
    """

    def __init__(
        self,
        world: carla.World,
        num_rays: int = 8,
        hit_threshold: float = 0.5,
    ):
        self._world = world
        self._num_rays = num_rays
        self._hit_threshold = hit_threshold

    def check(
        self,
        target_actor: carla.Actor,
        uav_camera_position: np.ndarray,
    ) -> tuple[bool, float]:
        """Check if target is occluded from UAV camera.

        Args:
            target_actor: The CARLA vehicle/walker actor
            uav_camera_position: [x, y, z] of UAV camera in world coords

        Returns:
            (is_occluded: bool, occlusion_level: float [0, 1])
            occlusion_level = fraction of rays blocked
        """
        bbox = target_actor.bounding_box
        target_transform = target_actor.get_transform()

        # Get bbox corner positions in world coordinates
        corners = self._get_bbox_corners(bbox, target_transform)

        # Cast rays from corners toward UAV
        uav_loc = carla.Location(
            x=uav_camera_position[0],
            y=uav_camera_position[1],
            z=uav_camera_position[2],
        )

        blocked = 0
        for corner in corners:
            c = carla.Location(x=float(corner[0]), y=float(corner[1]), z=float(corner[2]))
            d_cam = c.distance(uav_loc)
            # cast_ray returns a LIST of labelled points where the ray crosses geometry;
            # a corner is blocked if something lies strictly between it and the camera
            # (2 m buffers skip the target's own body and the camera end).
            for h in self._world.cast_ray(c, uav_loc):
                if 2.0 < c.distance(h.location) < d_cam - 2.0:
                    blocked += 1
                    break

        level = blocked / max(len(corners), 1)
        return level >= self._hit_threshold, level

    @staticmethod
    def _get_bbox_corners(
        bbox: carla.BoundingBox,
        transform: carla.Transform,
        num_samples: int = 8,
    ) -> list[np.ndarray]:
        """Sample points on the bounding box surface in world coordinates.

        Uses 8 corners + edge midpoints for better coverage.
        """
        ext = bbox.extent
        loc = transform.location
        rot = transform.rotation

        # Local-space corners
        lx, ly, lz = ext.x, ext.y, ext.z
        local_corners = [
            np.array([+lx, +ly, +lz]),
            np.array([+lx, +ly, -lz]),
            np.array([+lx, -ly, +lz]),
            np.array([+lx, -ly, -lz]),
            np.array([-lx, +ly, +lz]),
            np.array([-lx, +ly, -lz]),
            np.array([-lx, -ly, +lz]),
            np.array([-lx, -ly, -lz]),
        ]

        # Rotate to world frame
        yaw_rad = math.radians(rot.yaw)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        # Simplified 2D rotation (UAV scenario: pitch/roll ≈ 0)
        world_corners = []
        for lc in local_corners[:num_samples]:
            wx = loc.x + lc[0] * cos_y - lc[1] * sin_y
            wy = loc.y + lc[0] * sin_y + lc[1] * cos_y
            wz = loc.z + lc[2]
            world_corners.append(np.array([wx, wy, wz]))

        return world_corners
