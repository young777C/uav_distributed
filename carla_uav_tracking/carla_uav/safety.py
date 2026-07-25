"""Safety constraints for the kinematic UAV.

Checks performed each step:
  1. Ground collision (altitude check)
  2. Energy depletion
  3. Out-of-bounds (map boundary)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class SafetyEvent(Enum):
    NONE = auto()
    LOW_ALTITUDE = auto()     # < min safe altitude
    ENERGY_LOW = auto()       # < return-home threshold
    ENERGY_CRITICAL = auto()  # imminent depletion
    OUT_OF_BOUNDS = auto()    # outside map


@dataclass
class SafetyConfig:
    min_altitude: float = 3.0         # meters above ground
    max_altitude: float = 200.0       # meters
    energy_return_threshold: float = 0.15  # must return home below this
    energy_critical_threshold: float = 0.05


@dataclass
class SafetyResult:
    event: SafetyEvent = SafetyEvent.NONE
    is_safe: bool = True
    message: str = ""


class SafetyMonitor:
    """Episode-level safety monitor — terminates episode on violations."""

    def __init__(self, config: SafetyConfig | None = None):
        self._cfg = config or SafetyConfig()
        self.violation_count = 0

    def reset(self) -> None:
        self.violation_count = 0

    def check(
        self,
        uav_position: np.ndarray,
        ground_altitude: float,
        energy: float,
        map_bounds: tuple[float, float, float, float] | None = None,
    ) -> SafetyResult:
        """Check all safety constraints.

        Args:
            uav_position: [x, y, z] in world coords
            ground_altitude: z-coordinate of ground at this location
            energy: [0, 1] remaining energy fraction
            map_bounds: (min_x, max_x, min_y, max_y) or None

        Returns:
            SafetyResult with violation details
        """
        altitude_agl = uav_position[2] - ground_altitude

        # Ground collision / low altitude
        if altitude_agl < self._cfg.min_altitude:
            self.violation_count += 1
            return SafetyResult(
                event=SafetyEvent.LOW_ALTITUDE,
                is_safe=False,
                message=f"Altitude {altitude_agl:.1f}m < {self._cfg.min_altitude}m minimum",
            )

        # Critical energy
        if energy < self._cfg.energy_critical_threshold:
            self.violation_count += 1
            return SafetyResult(
                event=SafetyEvent.ENERGY_CRITICAL,
                is_safe=False,
                message=f"Energy {energy:.2%} < {self._cfg.energy_critical_threshold:.0%} critical",
            )

        # Low energy warning (non-fatal in simulation, recorded as event)
        if energy < self._cfg.energy_return_threshold:
            return SafetyResult(
                event=SafetyEvent.ENERGY_LOW,
                is_safe=True,   # still safe to continue
                message=f"Energy {energy:.2%} < return threshold",
            )

        # Map bounds
        if map_bounds is not None:
            min_x, max_x, min_y, max_y = map_bounds
            px, py = uav_position[0], uav_position[1]
            if not (min_x <= px <= max_x and min_y <= py <= max_y):
                self.violation_count += 1
                return SafetyResult(
                    event=SafetyEvent.OUT_OF_BOUNDS,
                    is_safe=False,
                    message=f"Position ({px:.0f}, {py:.0f}) outside map bounds",
                )

        return SafetyResult()
