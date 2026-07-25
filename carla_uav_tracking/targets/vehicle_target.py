"""Target agents for UAV tracking — vehicles with CARLA road network navigation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import carla
import numpy as np


@dataclass
class TargetState:
    position: np.ndarray
    velocity: np.ndarray
    yaw: float
    speed: float
    visible: bool
    bbox_2d: tuple[float, float, float, float] | None


class VehicleTarget:
    """CARLA vehicle following road network via live waypoint queries.

    Each frame: query the nearest road waypoint from the vehicle's current
    position, then look ahead ~45m along the road for smooth steering.
    At intersections, randomly choose direction with low probability.
    """

    def __init__(
        self,
        world: carla.World,
        blueprint_name: str,
        spawn_point: carla.Transform,
        speed_kmh: float = 30.0,
        carla_client: carla.Client | None = None,
    ):
        bp = world.get_blueprint_library().find(blueprint_name)
        if bp is None:
            raise ValueError(f"Vehicle blueprint not found: {blueprint_name}")
        bp.set_attribute("role_name", "target_vehicle")

        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        if spawn_points:
            pt = random.choice(spawn_points)
            pt.location.x += random.uniform(-0.5, 0.5)
            pt.location.y += random.uniform(-0.5, 0.5)
        else:
            pt = spawn_point

        self._vehicle = None
        for attempt in range(5):
            try:
                self._vehicle = world.spawn_actor(bp, pt)
                break
            except RuntimeError:
                if attempt == 4: raise
                pt.location.x += random.uniform(-3, 3)
                pt.location.y += random.uniform(-3, 3)

        self._world = world
        self._map = carla_map
        self._target_speed_ms = random.uniform(6.0, 12.0)  # moderate speed — stay on road
        self._is_turning = False
        self._direction = self._compute_direction(pt.rotation.yaw)
        self._stuck_counter = 0
        self._total_distance = 0.0
        self._turn_cooldown = 0.0
        self._steer = 0.0

        self._cached_transform = pt
        self._cached_velocity = np.zeros(3)
        self._cached_speed = 0.0

        # CARLA autopilot — follows road network, obeys traffic lights
        if carla_client is not None:
            tm = carla_client.get_trafficmanager(8000)
            tm.set_synchronous_mode(True)
            tm.global_percentage_speed_difference(
                -random.uniform(10, 40)  # 10-40% above speed limit
            )
        self._vehicle.set_autopilot(True, 8000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        t = self._vehicle.get_transform()
        v = self._vehicle.get_velocity()
        self._cached_transform = t
        self._cached_velocity = np.array([v.x, v.y, v.z])
        self._cached_speed = float(np.linalg.norm(self._cached_velocity))
        return (
            np.array([t.location.x, t.location.y, t.location.z]),
            self._cached_velocity.copy(),
            t.rotation.yaw,
            self._cached_speed,
        )

    def step(self, dt: float = 0.1) -> None:
        """Autopilot handles everything — just track state."""
        speed = self._cached_speed
        self._total_distance += speed * dt

        # Detect turning from yaw change
        current_yaw = self._vehicle.get_transform().rotation.yaw
        if hasattr(self, '_prev_yaw'):
            yaw_diff = abs(current_yaw - self._prev_yaw)
            self._is_turning = yaw_diff > 3.0
        self._prev_yaw = current_yaw

    @property
    def is_turning(self) -> bool:
        return getattr(self, '_is_turning', False)

    @property
    def is_turning(self) -> bool:
        return self._is_turning

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def route_info(self) -> dict:
        return {"distance": self._total_distance, "stuck": self._stuck_counter > 10}

    @property
    def id(self) -> int:
        return self._vehicle.id

    @property
    def bounding_box(self) -> carla.BoundingBox:
        return self._vehicle.bounding_box

    @property
    def transform(self) -> carla.Transform:
        return self._vehicle.get_transform()

    def destroy(self) -> None:
        if self._vehicle is not None and self._vehicle.is_alive:
            self._vehicle.destroy()

    def is_alive(self) -> bool:
        return self._vehicle.is_alive

    @staticmethod
    def _compute_direction(yaw: float) -> str:
        yaw = yaw % 360
        if yaw < 45 or yaw > 315: return "east"
        elif yaw < 135: return "north"
        elif yaw < 225: return "west"
        else: return "south"


class PedestrianTarget:
    """CARLA walker with AI controller."""

    def __init__(
        self,
        world: carla.World,
        blueprint_name: str,
        spawn_point: carla.Transform,
        walk_speed: float = 2.0,
    ):
        bp = world.get_blueprint_library().find(blueprint_name)
        if bp is None:
            raise ValueError(f"Walker blueprint not found: {blueprint_name}")
        bp.set_attribute("role_name", "target_pedestrian")

        self._walker = None
        for attempt in range(5):
            pt = spawn_point
            if attempt > 0:
                pt = carla.Transform(
                    carla.Location(
                        x=spawn_point.location.x + random.uniform(-5, 5),
                        y=spawn_point.location.y + random.uniform(-5, 5),
                        z=spawn_point.location.z + random.uniform(1, 3),
                    ),
                    spawn_point.rotation,
                )
            try:
                self._walker = world.spawn_actor(bp, pt)
                break
            except RuntimeError:
                if attempt == 4: raise

        self._world = world
        controller_bp = world.get_blueprint_library().find("controller.ai.walker")
        self._controller = world.spawn_actor(controller_bp, carla.Transform(), self._walker)
        self._controller.start()
        self._controller.go_to_location(world.get_random_location_from_navigation())
        self._controller.set_max_speed(walk_speed)

    def get_state(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        t = self._walker.get_transform()
        v = self._walker.get_velocity()
        velocity = np.array([v.x, v.y, v.z])
        return (
            np.array([t.location.x, t.location.y, t.location.z]),
            velocity,
            t.rotation.yaw,
            float(np.linalg.norm(velocity)),
        )

    @property
    def id(self) -> int: return self._walker.id
    @property
    def bounding_box(self) -> carla.BoundingBox: return self._walker.bounding_box
    @property
    def transform(self) -> carla.Transform: return self._walker.get_transform()

    def destroy(self) -> None:
        if self._controller is not None and self._controller.is_alive:
            self._controller.stop(); self._controller.destroy()
        if self._walker is not None and self._walker.is_alive:
            self._walker.destroy()

    def is_alive(self) -> bool: return self._walker.is_alive
