"""Kinematic UAV model for CARLA data generation.

No physics simulation — the UAV is a freely moving 3D camera platform.
Rationale: we need visually reasonable tracking trajectories for VLA
training, not aerodynamics validation. Instant movement via set_transform()
maximizes frame rate and eliminates physics-related bugs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import carla
import numpy as np


@dataclass
class DroneState:
    """Full UAV state at a timestep."""
    position: np.ndarray    # [x, y, z] world coordinates
    velocity: np.ndarray    # [vx, vy, vz] world frame
    yaw: float              # degrees
    pitch: float            # degrees (always 0 for kinematic drone)
    energy: float           # normalized [0, 1]


class KinematicDrone:
    """A freely-movable camera platform in CARLA 3D space.

    Uses CARLA's spectator as the physical representation. The spectator
    is moved instantaneously via set_transform() — no physics, no latency.

    Usage:
        drone = KinematicDrone(world, start_transform)
        drone.step(dx=2.0, dy=-1.0, dz=0.5, dyaw=3.0)
        state = drone.get_state()
    """

    # Physical constraints (configurable)
    MAX_ALTITUDE = 200.0   # meters above ground
    MIN_ALTITUDE = 5.0     # meters above ground
    PITCH_LIMIT = 30.0     # degrees (enforced but not simulated)

    def __init__(
        self,
        world: carla.World,
        start_transform: carla.Transform,
        initial_energy: float = 1.0,
    ):
        self._world = world
        self._spectator = world.get_spectator()
        self._spectator.set_transform(start_transform)

        self.transform = start_transform
        self.energy = initial_energy

        # Previous state for velocity computation
        self._prev_location = start_transform.location
        self.velocity = np.zeros(3)

        # Energy model parameters
        self._energy_consumption_rate = 0.0001  # per meter flown

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, dx: float, dy: float, dz: float, dyaw: float) -> None:
        """Apply a 4-DoF displacement in world coordinates.

        Args:
            dx, dy, dz: displacement in meters (world frame)
            dyaw: yaw change in degrees
        """
        loc = self.transform.location
        rot = self.transform.rotation

        new_loc = carla.Location(
            x=loc.x + dx,
            y=loc.y + dy,
            z=max(self.MIN_ALTITUDE, min(self.MAX_ALTITUDE, loc.z + dz)),
        )
        new_rot = carla.Rotation(
            pitch=0.0,
            yaw=rot.yaw + dyaw,
            roll=0.0,
        )
        new_transform = carla.Transform(new_loc, new_rot)

        # Update velocity estimate
        displacement = np.array([dx, dy, dz])
        self.velocity = displacement / 0.1  # dt = 0.1s at 10Hz

        # Update energy
        distance = float(np.linalg.norm(displacement))
        self.energy = max(0.0, self.energy - distance * self._energy_consumption_rate)

        # Apply
        self._spectator.set_transform(new_transform)
        self._prev_location = loc
        self.transform = new_transform

    def get_state(self) -> DroneState:
        """Return current UAV state vector."""
        loc = self.transform.location
        return DroneState(
            position=np.array([loc.x, loc.y, loc.z]),
            velocity=self.velocity.copy(),
            yaw=self.transform.rotation.yaw,
            pitch=self.transform.rotation.pitch,
            energy=self.energy,
        )

    def get_camera_transform(self) -> carla.Transform:
        """Return the transform where an RGB sensor should be attached.

        The sensor is attached slightly below and behind the spectator
        to simulate a realistic UAV camera mount.
        """
        yaw_rad = math.radians(self.transform.rotation.yaw)
        forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
        # Camera is 0.5m behind and 0.3m below the spectator origin
        cam_loc = carla.Location(
            x=self.transform.location.x - forward[0] * 0.5,
            y=self.transform.location.y - forward[1] * 0.5,
            z=self.transform.location.z - 0.3,
        )
        cam_rot = carla.Rotation(
            pitch=-30.0,  # shallow angle — target is close and slightly ahead
            yaw=self.transform.rotation.yaw,
            roll=0.0,
        )
        return carla.Transform(cam_loc, cam_rot)

    def look_at(self, world_point: np.ndarray) -> float:
        """Compute the yaw needed to face a world-space point.

        Returns:
            yaw_error: degrees between current yaw and desired yaw
        """
        loc = self.transform.location
        desired_yaw = math.degrees(
            math.atan2(world_point[1] - loc.y, world_point[0] - loc.x)
        )
        current_yaw = self.transform.rotation.yaw
        return _normalize_angle(desired_yaw - current_yaw)


def _normalize_angle(angle_deg: float) -> float:
    """Wrap angle to [-180, 180]."""
    return ((angle_deg + 180.0) % 360.0) - 180.0
