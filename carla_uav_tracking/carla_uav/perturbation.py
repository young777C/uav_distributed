"""APF-style perturbation injection for UAV tracking.

Periodically pushes the UAV off its optimal tracking trajectory
to simulate tracking failures (occlusion, wind gust, sensor error).
The PID controller then recovers — producing natural recovery trajectories
in the training data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np


class PerturbationType(Enum):
    NONE = auto()
    PUSH_ASIDE = auto()    # lateral push — UAV displaced sideways
    LAG_BEHIND = auto()    # UAV decelerates — falls far behind
    OVERSHOOT = auto()     # UAV overshoots — passes the target
    ALTITUDE_DROP = auto() # UAV loses altitude temporarily


@dataclass
class PerturbationConfig:
    probability: float = 0.03          # ~1 per 3s at 10Hz
    min_interval: float = 8.0          # seconds between perturbations
    push_aside_range: tuple[float, float] = (20.0, 50.0)  # meters
    lag_behind_range: tuple[float, float] = (30.0, 80.0)
    overshoot_range: tuple[float, float] = (20.0, 40.0)
    altitude_drop_range: tuple[float, float] = (10.0, 25.0)
    recovery_time_range: tuple[float, float] = (2.0, 8.0)  # seconds to recover


@dataclass
class PerturbationEvent:
    type: PerturbationType
    timestamp: float
    magnitude: float         # meters of displacement
    duration: float          # expected recovery time
    resolved_at: Optional[float] = None  # when UAV re-acquired target


class PerturbationInjector:
    """Injects random tracking perturbations into expert policy actions.

    The perturbation overrides the PID output for one frame,
    then the PID naturally recovers over the following seconds.
    This creates training data with embedded recovery trajectories.

    Usage:
        injector = PerturbationInjector(config)
        for each frame:
            action = pid_output
            perturbed, event = injector.maybe_perturb(action, uav_pos, timestamp)
            if perturbed:
                action = perturbed
    """

    def __init__(self, config: PerturbationConfig | None = None):
        self._cfg = config or PerturbationConfig()
        self._last_perturbation_at = -999.0
        self._active: Optional[PerturbationEvent] = None
        self._history: list[PerturbationEvent] = []

    def reset(self) -> None:
        self._last_perturbation_at = -999.0
        self._active = None
        self._history = []

    def maybe_perturb(
        self,
        action: tuple[float, float, float, float],  # [dx, dy, dz, dyaw]
        uav_pos: np.ndarray,
        target_pos: np.ndarray,
        timestamp: float,
    ) -> tuple[tuple[float, float, float, float], Optional[PerturbationEvent]]:
        """Check if a perturbation should be injected this frame.

        Returns:
            (perturbed_action, event_or_None)
            If no perturbation, returns the original action.
        """
        # Don't perturb if one is already active
        if self._active is not None:
            return action, self._active

        # Check cooldown
        if timestamp - self._last_perturbation_at < self._cfg.min_interval:
            return action, None

        # Random trigger
        if random.random() > self._cfg.probability:
            return action, None

        # Choose perturbation type
        ptype = random.choice(list(PerturbationType))
        if ptype == PerturbationType.NONE:
            return action, None

        # Compute perturbed action
        perturbed = self._apply_perturbation(action, uav_pos, target_pos, ptype)

        duration = random.uniform(*self._cfg.recovery_time_range)
        event = PerturbationEvent(
            type=ptype, timestamp=timestamp,
            magnitude=self._perturbation_magnitude(perturbed, action),
            duration=duration,
        )
        self._active = event
        self._last_perturbation_at = timestamp
        self._history.append(event)

        # Active for only 1 frame — PID recovery starts next frame
        self._active = None

        return perturbed, event

    @property
    def history(self) -> list[PerturbationEvent]:
        return self._history

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_perturbation(
        self,
        action: tuple[float, float, float, float],
        uav_pos: np.ndarray,
        target_pos: np.ndarray,
        ptype: PerturbationType,
    ) -> tuple[float, float, float, float]:
        dx, dy, dz, dyaw = action

        # Direction from UAV to target (for lateral pushes)
        to_target = target_pos[:2] - uav_pos[:2]
        dist = float(np.linalg.norm(to_target))
        if dist > 0.01:
            to_target /= dist
        else:
            to_target = np.array([1.0, 0.0])

        # Perpendicular to target direction
        perp = np.array([-to_target[1], to_target[0]])

        if ptype == PerturbationType.PUSH_ASIDE:
            mag = random.uniform(*self._cfg.push_aside_range)
            side = random.choice([-1, 1])
            dx += perp[0] * mag * side
            dy += perp[1] * mag * side

        elif ptype == PerturbationType.LAG_BEHIND:
            mag = random.uniform(*self._cfg.lag_behind_range)
            dx -= to_target[0] * mag
            dy -= to_target[1] * mag

        elif ptype == PerturbationType.OVERSHOOT:
            mag = random.uniform(*self._cfg.overshoot_range)
            dx += to_target[0] * mag
            dy += to_target[1] * mag

        elif ptype == PerturbationType.ALTITUDE_DROP:
            dz -= random.uniform(*self._cfg.altitude_drop_range)

        return (dx, dy, dz, dyaw)

    @staticmethod
    def _perturbation_magnitude(
        perturbed: tuple, original: tuple,
    ) -> float:
        return float(np.sqrt(
            (perturbed[0] - original[0])**2 +
            (perturbed[1] - original[1])**2 +
            (perturbed[2] - original[2])**2
        ))
