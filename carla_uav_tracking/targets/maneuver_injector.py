"""Inject SMOOTH, rule-abiding maneuver events into the target via the
TrafficManager (never raw control / never autopilot-off).

The old version turned autopilot OFF and applied random steering, which made the
target swerve, break traffic rules, and collide. This version keeps the target on
autopilot the whole time and only modulates its TM speed / lane, so it stays on the
road and obeys rules. Motion-intent language labels are derived post-hoc from the
actual trajectory, so we don't need violent control to get 'about to turn/stop'.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class ManeuverType(Enum):
    NONE = auto()
    SUDDEN_STOP = auto()      # smooth slow-down (TM speed → ~10%)
    ACCELERATION = auto()     # speed up (TM speed → +60%)
    LANE_CHANGE = auto()      # TM-forced lane change (smooth)


@dataclass
class ManeuverConfig:
    sudden_stop_prob: float = 0.0
    sharp_turn_prob: float = 0.0        # kept for config back-compat → lane change
    acceleration_prob: float = 0.0
    u_turn_prob: float = 0.0            # kept for config back-compat → ignored
    min_interval: float = 6.0
    base_speed_diff: float = 15.0       # calm cruising speed to restore to (% slower)


@dataclass
class ManeuverEvent:
    type: ManeuverType
    timestamp: float
    duration: float
    metadata: dict[str, Any]


class ManeuverInjector:
    """Modulate a TM-driven target's speed/lane to create smooth maneuver diversity."""

    def __init__(self, config: ManeuverConfig | None = None, traffic_manager=None):
        self._cfg = config or ManeuverConfig()
        self._tm = traffic_manager
        self._last = -999.0
        self._active: ManeuverEvent | None = None
        self._history: list[ManeuverEvent] = []

    def reset(self) -> None:
        self._last = -999.0
        self._active = None
        self._history = []

    def maybe_inject(self, vehicle, timestamp: float) -> ManeuverEvent | None:
        if self._tm is None or vehicle is None:
            return None
        # End an active maneuver → restore calm cruising speed.
        if self._active is not None:
            if timestamp - self._active.timestamp > self._active.duration:
                self._safe(lambda: self._tm.vehicle_percentage_speed_difference(
                    vehicle, float(self._cfg.base_speed_diff)))
                self._active = None
            return None
        if timestamp - self._last < self._cfg.min_interval:
            return None

        roll = random.random()
        cum = self._cfg.sudden_stop_prob
        if roll < cum:
            return self._start(vehicle, timestamp, ManeuverType.SUDDEN_STOP,
                               lambda: self._tm.vehicle_percentage_speed_difference(vehicle, 92.0),
                               random.uniform(2.0, 4.0))
        cum += self._cfg.acceleration_prob
        if roll < cum:
            return self._start(vehicle, timestamp, ManeuverType.ACCELERATION,
                               lambda: self._tm.vehicle_percentage_speed_difference(vehicle, -60.0),
                               random.uniform(2.0, 4.0))
        cum += self._cfg.sharp_turn_prob + self._cfg.u_turn_prob
        if roll < cum:
            direction = random.choice([True, False])
            return self._start(vehicle, timestamp, ManeuverType.LANE_CHANGE,
                               lambda: self._tm.force_lane_change(vehicle, direction),
                               random.uniform(1.0, 2.0))
        return None

    def force_evade(self, vehicle, timestamp: float, duration: float) -> ManeuverEvent | None:
        """Deliberately induce a strong deviation (speed up + lane change) for `duration`s.
        Used by the LossEventScheduler to push the target out of frame during a blind
        window. Speed is auto-restored by maybe_inject once the duration elapses."""
        if self._tm is None or vehicle is None:
            return None
        self._safe(lambda: self._tm.vehicle_percentage_speed_difference(vehicle, -60.0))
        self._safe(lambda: self._tm.force_lane_change(vehicle, random.choice([True, False])))
        ev = ManeuverEvent(type=ManeuverType.ACCELERATION, timestamp=timestamp,
                           duration=duration, metadata={"forced": True})
        self._active = ev
        self._last = timestamp
        self._history.append(ev)
        return ev

    # ------------------------------------------------------------------
    def _start(self, vehicle, t, mtype, action, duration) -> ManeuverEvent | None:
        if not self._safe(action):
            return None
        ev = ManeuverEvent(type=mtype, timestamp=t, duration=duration, metadata={})
        self._active = ev
        self._last = t
        self._history.append(ev)
        return ev

    @staticmethod
    def _safe(fn) -> bool:
        try:
            fn()
            return True
        except (RuntimeError, AttributeError):
            return False

    @property
    def history(self) -> list[ManeuverEvent]:
        return self._history
