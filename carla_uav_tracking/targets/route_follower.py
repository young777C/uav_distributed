"""Pure Pursuit route following for target vehicles.

Generates a random route along CARLA's road network waypoints,
then follows it using a simple pure-pursuit controller with
speed variation, intersection slowing, and random maneuvers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import carla
import numpy as np


@dataclass
class RouteConfig:
    num_waypoints: tuple[int, int] = (5, 15)
    min_turns: int = 2
    target_speed_ms: tuple[float, float] = (8.0, 20.0)
    lookahead_distance: float = 15.0            # increased for smoother steering
    turn_deceleration: float = 0.6
    steer_gain: float = 0.5                     # was 2.0 — too aggressive
    maneuver_interval: tuple[float, float] = (15.0, 30.0)
    maneuver_prob: float = 0.3


class RouteFollower:
    """Drives a CARLA vehicle along a randomly generated road network route.

    Usage:
        follower = RouteFollower(vehicle, world, config)
        for each tick:
            control = follower.get_control()
            vehicle.apply_control(control)
    """

    def __init__(
        self,
        vehicle: carla.Vehicle,
        world: carla.World,
        config: RouteConfig | None = None,
    ):
        self._vehicle = vehicle
        self._world = world
        self._map = world.get_map()
        self._cfg = config or RouteConfig()

        # Generate route
        self._target_speed = random.uniform(*self._cfg.target_speed_ms)
        self._waypoints = self._generate_route()
        self._current_wp_idx = 1  # skip spawn waypoint (vehicle is already there)

        # Maneuver state
        self._next_maneuver_time = random.uniform(*self._cfg.maneuver_interval)
        self._active_maneuver: Optional[tuple[str, float]] = None  # (type, remaining)
        self._maneuver_history: list[dict] = []

        # Stats
        self._elapsed = 0.0
        self._n_turns_completed = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_control(self, dt: float = 0.1) -> carla.VehicleControl:
        """Return vehicle control for this frame."""
        self._elapsed += dt

        # Check maneuver expiry
        if self._active_maneuver is not None:
            mtype, remaining = self._active_maneuver
            remaining -= dt
            if remaining <= 0:
                self._active_maneuver = None
            else:
                self._active_maneuver = (mtype, remaining)
                return self._maneuver_control(mtype)

        # Check if should start new maneuver
        if self._elapsed >= self._next_maneuver_time:
            self._next_maneuver_time = self._elapsed + random.uniform(
                *self._cfg.maneuver_interval
            )
            if random.random() < self._cfg.maneuver_prob:
                return self._start_random_maneuver()

        # Normal route following
        return self._pure_pursuit_control()

    @property
    def current_waypoint_index(self) -> int:
        return self._current_wp_idx

    @property
    def total_waypoints(self) -> int:
        return len(self._waypoints)

    @property
    def target_speed(self) -> float:
        return self._target_speed

    @property
    def is_turning(self) -> bool:
        """True if vehicle is at an intersection / turning."""
        if self._current_wp_idx >= len(self._waypoints) - 1:
            return False
        wp_current = self._waypoints[self._current_wp_idx]
        wp_next = self._waypoints[min(self._current_wp_idx + 1,
                                       len(self._waypoints) - 1)]
        yaw_diff = abs(wp_next.transform.rotation.yaw - wp_current.transform.rotation.yaw)
        yaw_diff = min(yaw_diff, 360 - yaw_diff)
        return yaw_diff > 30.0

    @property
    def maneuver_history(self) -> list[dict]:
        return self._maneuver_history

    def get_current_direction(self) -> str:
        """Return cardinal direction for language generation."""
        if not self._waypoints:
            return "unknown"
        yaw = self._vehicle.get_transform().rotation.yaw
        yaw = yaw % 360
        if yaw < 45 or yaw > 315: return "east"
        elif yaw < 135: return "north"
        elif yaw < 225: return "west"
        else: return "south"

    # ------------------------------------------------------------------
    # Internal — Route Generation
    # ------------------------------------------------------------------

    def _generate_route(self) -> list[carla.Waypoint]:
        """Generate a random route along CARLA road network.

        Ensures the route has at least cfg.min_turns intersections.
        """
        spawn_points = self._map.get_spawn_points()
        if not spawn_points:
            return []

        # Start from a random spawn point
        start = random.choice(spawn_points)
        start_wp = self._map.get_waypoint(start.location)

        # Walk along the road network
        num_wp = random.randint(*self._cfg.num_waypoints)
        waypoints = [start_wp]

        for _ in range(num_wp - 1):
            # Choose next waypoint (prefer going straight, sometimes turn)
            next_wps = waypoints[-1].next(15.0)  # 15m ahead

            if not next_wps:
                break

            if len(next_wps) > 1 and random.random() < 0.3:
                # Make a turn at intersection
                next_wp = random.choice(next_wps[1:]) if len(next_wps) > 1 else next_wps[0]
            else:
                next_wp = next_wps[0]  # go straight

            waypoints.append(next_wp)

        # Count turns
        turns = 0
        for i in range(1, len(waypoints)):
            prev_yaw = waypoints[i-1].transform.rotation.yaw
            curr_yaw = waypoints[i].transform.rotation.yaw
            diff = abs(curr_yaw - prev_yaw)
            if min(diff, 360 - diff) > 30:
                turns += 1

        # Regenerate if not enough turns
        if turns < self._cfg.min_turns and len(spawn_points) > 1:
            return self._generate_route()

        return waypoints

    # ------------------------------------------------------------------
    # Internal — Controls
    # ------------------------------------------------------------------

    def _pure_pursuit_control(self) -> carla.VehicleControl:
        """Standard pure pursuit: steer toward lookahead point on path."""
        if not self._waypoints:
            return carla.VehicleControl(throttle=0.5)

        vehicle_loc = self._vehicle.get_location()
        vehicle_yaw = np.radians(self._vehicle.get_transform().rotation.yaw)

        # Advance waypoint index if close enough
        target_loc = self._waypoints[self._current_wp_idx].transform.location
        dist_to_wp = vehicle_loc.distance(target_loc)
        if dist_to_wp < 5.0 and self._current_wp_idx < len(self._waypoints) - 1:
            self._current_wp_idx += 1
            self._n_turns_completed += 1

        # Find lookahead point
        lookahead_point = self._find_lookahead_point(vehicle_loc)

        # Compute steering
        dx = lookahead_point.x - vehicle_loc.x
        dy = lookahead_point.y - vehicle_loc.y
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = desired_yaw - vehicle_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))  # normalize
        steer = np.clip(yaw_error * self._cfg.steer_gain, -1.0, 1.0)

        # Speed — reduce at turns
        speed = self._target_speed
        if self.is_turning:
            speed *= self._cfg.turn_deceleration

        # Throttle based on current speed
        current_vel = self._vehicle.get_velocity()
        current_speed = np.sqrt(current_vel.x**2 + current_vel.y**2)
        throttle = 0.8 if current_speed < speed else 0.3

        return carla.VehicleControl(
            throttle=throttle, steer=float(steer), brake=0.0
        )

    def _find_lookahead_point(self, vehicle_loc: carla.Location) -> carla.Location:
        """Find point on route at lookahead distance."""
        accumulated = 0.0
        prev_loc = vehicle_loc

        for i in range(self._current_wp_idx, len(self._waypoints)):
            wp_loc = self._waypoints[i].transform.location
            seg_dist = prev_loc.distance(wp_loc)

            if accumulated + seg_dist >= self._cfg.lookahead_distance:
                # Interpolate on this segment
                remaining = self._cfg.lookahead_distance - accumulated
                frac = remaining / max(seg_dist, 0.01)
                return carla.Location(
                    x=prev_loc.x + (wp_loc.x - prev_loc.x) * frac,
                    y=prev_loc.y + (wp_loc.y - prev_loc.y) * frac,
                    z=wp_loc.z,
                )

            accumulated += seg_dist
            prev_loc = wp_loc

        # Return last waypoint
        return self._waypoints[-1].transform.location

    # ------------------------------------------------------------------
    # Internal — Maneuvers
    # ------------------------------------------------------------------

    def _start_random_maneuver(self) -> carla.VehicleControl:
        mtype = random.choice(["sudden_stop", "hard_accel", "sharp_turn", "lane_change"])
        duration = random.uniform(1.5, 4.0)
        self._active_maneuver = (mtype, duration)
        self._maneuver_history.append({
            "time": self._elapsed, "type": mtype, "duration": duration,
        })
        return self._maneuver_control(mtype)

    def _maneuver_control(self, mtype: str) -> carla.VehicleControl:
        if mtype == "sudden_stop":
            return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        elif mtype == "hard_accel":
            return carla.VehicleControl(throttle=1.0, brake=0.0, steer=0.0)
        elif mtype == "sharp_turn":
            return carla.VehicleControl(
                throttle=0.4, brake=0.0,
                steer=random.choice([-1.0, 1.0]),
            )
        elif mtype == "lane_change":
            steer = random.uniform(-0.5, 0.5)
            return carla.VehicleControl(throttle=0.5, brake=0.0, steer=steer)
        return carla.VehicleControl(throttle=0.5)
