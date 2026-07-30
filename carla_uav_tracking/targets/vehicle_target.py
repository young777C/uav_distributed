"""Target agents for UAV tracking — vehicles with CARLA road network navigation."""

from __future__ import annotations

import math
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
        color: str | None = None,
        role_name: str = "target_vehicle",
        tm_port: int = 8000,
        speed_diff: float | None = None,
    ):
        bp = world.get_blueprint_library().find(blueprint_name)
        if bp is None:
            raise ValueError(f"Vehicle blueprint not found: {blueprint_name}")
        bp.set_attribute("role_name", role_name)
        # Optional color control (for same-shape/different-color similarity).
        if color is not None and bp.has_attribute("color"):
            try:
                bp.set_attribute("color", color)
            except (RuntimeError, ValueError):
                pass

        carla_map = world.get_map()
        # P0-a fix: RESPECT the requested spawn point (snap to the nearest driving
        # lane so it stays on-road), instead of teleporting to a random map spawn
        # point. This is what keeps target + distractors co-located and co-visible.
        wp = carla_map.get_waypoint(
            spawn_point.location, project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        base = wp.transform if wp is not None else spawn_point

        self._vehicle = None
        for attempt in range(8):
            jitter = 1.5 * attempt  # first attempt = exact requested point
            pt = carla.Transform(
                carla.Location(
                    x=base.location.x + random.uniform(-jitter, jitter),
                    y=base.location.y + random.uniform(-jitter, jitter),
                    z=base.location.z + 0.3,
                ),
                base.rotation,
            )
            try:
                self._vehicle = world.spawn_actor(bp, pt)
                break
            except RuntimeError:
                if attempt == 7:
                    raise

        self._world = world
        self._map = carla_map
        self._target_speed_ms = random.uniform(6.0, 12.0)  # moderate speed — stay on road
        self._is_turning = False
        self._direction = self._compute_direction(base.rotation.yaw)
        self._stuck_counter = 0
        self._total_distance = 0.0
        self._turn_cooldown = 0.0
        self._steer = 0.0

        self._cached_transform = base
        self._cached_velocity = np.zeros(3)
        self._cached_speed = 0.0

        # CARLA autopilot — follows road network, obeys traffic lights.
        # Per-vehicle speed difference (negative = faster than the limit).
        if carla_client is not None:
            tm = carla_client.get_trafficmanager(tm_port)
            tm.set_synchronous_mode(True)
            try:
                if speed_diff is not None:
                    tm.vehicle_percentage_speed_difference(self._vehicle, float(speed_diff))
                # Stay in-lane & orderly (no weaving); recycling provides the dynamics.
                tm.auto_lane_change(self._vehicle, False)
                tm.distance_to_leading_vehicle(self._vehicle, 2.5)
            except (RuntimeError, AttributeError):
                pass
        self._vehicle.set_autopilot(True, tm_port)

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
            try:
                self._vehicle.set_autopilot(False)  # let the TM drop it first
            except RuntimeError:
                pass
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
    """CARLA walker driven by manual WalkerControl (no AI controller / navmesh).

    The walker AI controller + navigation mesh segfaults the client in this
    offscreen build, so we move the walker directly: apply WalkerControl(direction,
    speed) every step and let it stroll along the road with gentle heading drift.
    """

    def __init__(
        self,
        world: carla.World,
        blueprint_name: str,
        spawn_point: carla.Transform,
        walk_speed: float = 1.6,
        carla_client: carla.Client | None = None,
        snap_to_road: bool = True,
        role_name: str = "target_pedestrian",
    ):
        bp = world.get_blueprint_library().find(blueprint_name)
        if bp is None:
            raise ValueError(f"Walker blueprint not found: {blueprint_name}")
        bp.set_attribute("role_name", role_name)

        if snap_to_road:
            carla_map = world.get_map()
            wp = carla_map.get_waypoint(
                spawn_point.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            base = wp.transform if wp is not None else spawn_point
        else:
            base = spawn_point  # e.g. a navmesh location for ambient sidewalk walkers

        self._walker = None
        for attempt in range(8):
            jitter = 1.0 * attempt
            pt = carla.Transform(
                carla.Location(
                    x=base.location.x + random.uniform(-jitter, jitter),
                    y=base.location.y + random.uniform(-jitter, jitter),
                    z=base.location.z + 1.0,
                ),
                base.rotation,
            )
            try:
                self._walker = world.spawn_actor(bp, pt)
                break
            except RuntimeError:
                if attempt == 7:
                    raise

        self._world = world
        self._speed = float(walk_speed)
        self._heading = base.rotation.yaw

    def step(self, dt: float = 0.1) -> None:
        # Stroll forward with a gentle random heading drift.
        self._heading += random.uniform(-3.0, 3.0)
        yaw = math.radians(self._heading)
        self._walker.apply_control(carla.WalkerControl(
            direction=carla.Vector3D(x=math.cos(yaw), y=math.sin(yaw), z=0.0),
            speed=self._speed, jump=False,
        ))

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
        if self._walker is not None and self._walker.is_alive:
            self._walker.destroy()

    def is_alive(self) -> bool: return self._walker.is_alive
