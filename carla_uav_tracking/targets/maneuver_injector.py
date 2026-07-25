"""Inject maneuver events into target behavior during tracking episodes.

Creates diverse scenarios: sudden stops, sharp turns, accelerations, U-turns.
These are the events that make tracking non-trivial — without them, tracking
is just "follow a car in a straight line."
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import carla


class ManeuverType(Enum):
    NONE = auto()
    SUDDEN_STOP = auto()
    SHARP_TURN = auto()
    ACCELERATION = auto()
    U_TURN = auto()


@dataclass
class ManeuverConfig:
    sudden_stop_prob: float = 0.05
    sharp_turn_prob: float = 0.03
    acceleration_prob: float = 0.02
    u_turn_prob: float = 0.01
    min_interval: float = 5.0  # seconds between maneuvers


@dataclass
class ManeuverEvent:
    type: ManeuverType
    timestamp: float
    duration: float           # how long it lasts
    metadata: dict[str, Any]  # e.g., {"from_speed": 15.0, "to_speed": 0.0}


class ManeuverInjector:
    """Inject random maneuvers into a CARLA vehicle target.

    Usage:
        injector = ManeuverInjector(config)
        for each frame:
            maneuver = injector.maybe_inject(vehicle, timestamp)
            if maneuver:
                record_maneuver_event(maneuver)
    """

    def __init__(self, config: ManeuverConfig | None = None):
        self._cfg = config or ManeuverConfig()
        self._last_maneuver_time = -999.0
        self._active_maneuver: ManeuverEvent | None = None
        self._maneuver_history: list[ManeuverEvent] = []

    def reset(self) -> None:
        self._last_maneuver_time = -999.0
        self._active_maneuver = None
        self._maneuver_history = []

    def maybe_inject(
        self,
        vehicle: carla.Vehicle,
        timestamp: float,
    ) -> ManeuverEvent | None:
        """Check if a new maneuver should start. Returns event if so.

        Call this every frame. Currently-active maneuvers are managed internally.
        """
        # Don't inject if one is already active
        if self._active_maneuver is not None:
            if timestamp - self._active_maneuver.timestamp > self._active_maneuver.duration:
                # Maneuver complete — restore normal autopilot
                self._restore_autopilot(vehicle)
                self._active_maneuver = None
            return None

        # Check cooldown
        if timestamp - self._last_maneuver_time < self._cfg.min_interval:
            return None

        # Roll for each maneuver type
        roll = random.random()
        cumulative = 0.0

        # Sudden stop
        cumulative += self._cfg.sudden_stop_prob
        if roll < cumulative:
            return self._execute_stop(vehicle, timestamp)

        # Sharp turn
        cumulative += self._cfg.sharp_turn_prob
        if roll < cumulative:
            return self._execute_sharp_turn(vehicle, timestamp)

        # Acceleration
        cumulative += self._cfg.acceleration_prob
        if roll < cumulative:
            return self._execute_acceleration(vehicle, timestamp)

        # U-turn
        cumulative += self._cfg.u_turn_prob
        if roll < cumulative:
            return self._execute_u_turn(vehicle, timestamp)

        return None

    # ------------------------------------------------------------------
    # Maneuver implementations
    # ------------------------------------------------------------------

    def _execute_stop(self, vehicle: carla.Vehicle, t: float) -> ManeuverEvent:
        vehicle.set_autopilot(False)
        control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        vehicle.apply_control(control)
        event = ManeuverEvent(
            type=ManeuverType.SUDDEN_STOP,
            timestamp=t,
            duration=random.uniform(1.5, 4.0),
            metadata={"action": "emergency_brake"},
        )
        self._record(event)
        return event

    def _execute_sharp_turn(self, vehicle: carla.Vehicle, t: float) -> ManeuverEvent:
        vehicle.set_autopilot(False)
        steer = random.uniform(-1.0, 1.0)
        control = carla.VehicleControl(throttle=0.4, brake=0.0, steer=steer)
        vehicle.apply_control(control)
        event = ManeuverEvent(
            type=ManeuverType.SHARP_TURN,
            timestamp=t,
            duration=random.uniform(1.0, 3.0),
            metadata={"steer": steer},
        )
        self._record(event)
        return event

    def _execute_acceleration(self, vehicle: carla.Vehicle, t: float) -> ManeuverEvent:
        vehicle.set_autopilot(False)
        control = carla.VehicleControl(throttle=1.0, brake=0.0, steer=0.0)
        vehicle.apply_control(control)
        event = ManeuverEvent(
            type=ManeuverType.ACCELERATION,
            timestamp=t,
            duration=random.uniform(1.5, 3.0),
            metadata={"throttle": 1.0},
        )
        self._record(event)
        return event

    def _execute_u_turn(self, vehicle: carla.Vehicle, t: float) -> ManeuverEvent:
        vehicle.set_autopilot(False)
        control = carla.VehicleControl(
            throttle=0.4, brake=0.0,
            steer=random.choice([-1.0, 1.0]),
        )
        vehicle.apply_control(control)
        event = ManeuverEvent(
            type=ManeuverType.U_TURN,
            timestamp=t,
            duration=random.uniform(3.0, 6.0),
            metadata={"direction": "left" if control.steer < 0 else "right"},
        )
        self._record(event)
        return event

    def _restore_autopilot(self, vehicle: carla.Vehicle) -> None:
        """Return vehicle to autopilot after maneuver ends."""
        vehicle.set_autopilot(True, 8000)

    def _record(self, event: ManeuverEvent) -> None:
        self._active_maneuver = event
        self._last_maneuver_time = event.timestamp
        self._maneuver_history.append(event)

    @property
    def history(self) -> list[ManeuverEvent]:
        return self._maneuver_history
