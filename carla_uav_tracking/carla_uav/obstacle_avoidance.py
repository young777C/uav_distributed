"""Lightweight obstacle avoidance for UAV tracking.

Simple altitude-floor strategy: UAV stays above a minimum height
to avoid buildings and trees. No per-frame raycasting — relies on
starting altitude being above obstacle height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import carla
import numpy as np


@dataclass
class AvoidanceConfig:
    min_altitude_agl: float = 10.0    # meters above ground


@dataclass
class ObstacleEvent:
    timestamp: float
    action_taken: str         # "climb", "none"
    violation_magnitude: float  # how far below minimum


class ObstacleAvoider:
    """Altitude-floor obstacle avoidance for kinematic UAV.

    Ensures UAV stays above min_altitude_agl. For kinematic drones
    that teleport, this is sufficient — buildings are rarely above 30m
    in CARLA towns.
    """

    def __init__(self, config: AvoidanceConfig | None = None):
        self._cfg = config or AvoidanceConfig()
        self._events: list[ObstacleEvent] = []

    def reset(self) -> None:
        self._events = []

    def check(
        self,
        action: tuple[float, float, float, float],
        uav_pos: np.ndarray,
        timestamp: float,
    ) -> tuple[tuple[float, float, float, float], Optional[ObstacleEvent]]:
        """Ensure altitude stays above minimum.

        Returns:
            (possibly_modified_action, obstacle_event_or_None)
        """
        dx, dy, dz, dyaw = action
        predicted_z = uav_pos[2] + dz

        if predicted_z < self._cfg.min_altitude_agl:
            dz = self._cfg.min_altitude_agl - uav_pos[2] + 2.0
            event = ObstacleEvent(
                timestamp=timestamp,
                action_taken="climb",
                violation_magnitude=self._cfg.min_altitude_agl - predicted_z,
            )
            self._events.append(event)
            return (dx, dy, dz, dyaw), event

        return action, None

    @property
    def events(self) -> list[ObstacleEvent]:
        return self._events
