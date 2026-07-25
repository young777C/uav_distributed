"""Expert policy wrapping PID visual servoing with noise injection.

Produces diverse "expert styles" by randomizing:
  - PID gains per episode
  - Action noise per step
  - Viewing height offset per episode

This diversity is critical — it teaches the VLA that there are multiple
valid ways to track a target, preventing overfitting to a single style.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from carla_uav.pid_controller import (
    PIDGains,
    PIDVisualServo,
    PIDVisualServoConfig,
    VelocityLimits,
)


@dataclass
class ExpertStyle:
    """A specific expert "personality" for one episode."""
    pid_config: PIDVisualServoConfig
    noise_scale: float         # std of Gaussian noise added to actions
    viewing_offset: float       # meters above target


class ExpertPolicy:
    """PID-based expert with per-episode style randomization.

    Usage:
        expert = ExpertPolicy(config_dict)
        expert.reset_episode()  # randomize style for new episode
        for each step:
            action = expert.act(target_pos, uav_pos, uav_yaw, target_visible, t)
    """

    def __init__(self, config: dict[str, Any]):
        self._cfg = config
        self._pid: PIDVisualServo | None = None
        self._current_style: ExpertStyle | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Randomize expert style for a new episode."""
        pid = self._cfg.get("pid_params", {})

        def _sample(key: str, default: float) -> float:
            val = pid.get(key, [default, default])
            if isinstance(val, (int, float)):
                return float(val)
            return random.uniform(val[0], val[1])

        limits_cfg = self._cfg.get("velocity_limits", {})
        limits = VelocityLimits(
            xy_max=limits_cfg.get("xy_max", 15.0),
            z_max=limits_cfg.get("z_max", 5.0),
            yaw_rate_max=limits_cfg.get("yaw_rate_max", 60.0),
        )

        pid_config = PIDVisualServoConfig(
            xy=PIDGains(kp=_sample("kp_xy", 2.0), ki=_sample("ki_xy", 0.05), kd=_sample("kd_xy", 0.3)),
            z=PIDGains(kp=_sample("kp_z", 1.2), ki=_sample("ki_z", 0.05), kd=_sample("kd_z", 0.2), output_max=5.0),
            yaw=PIDGains(kp=_sample("kp_yaw", 1.5), ki=_sample("ki_yaw", 0.02), kd=_sample("kd_yaw", 0.1), output_max=60.0),
            limits=limits,
            viewing_height_offset=random.uniform(20.0, 30.0),  # above buildings, still visible
            dead_reckon_timeout=self._cfg.get("dead_reckon_timeout", 5.0),
        )

        noise_range = self._cfg.get("noise_scale", [0.02, 0.10])
        noise_scale = random.uniform(noise_range[0], noise_range[1])

        self._current_style = ExpertStyle(
            pid_config=pid_config,
            noise_scale=noise_scale,
            viewing_offset=pid_config.viewing_height_offset,
        )

        self._pid = PIDVisualServo(pid_config)

    def act(
        self,
        target_pos: np.ndarray | None,
        uav_pos: np.ndarray,
        uav_yaw: float,
        target_visible: bool,
        timestamp: float,
    ) -> tuple[float, float, float, float]:
        """Compute expert action = PID output + Gaussian noise.

        Returns:
            (dx, dy, dz, dyaw) in world frame
        """
        assert self._pid is not None, "Call reset_episode() first"
        assert self._current_style is not None

        # Get PID base action
        if target_pos is None and not target_visible:
            dx, dy, dz, dyaw = self._pid.compute(
                None, uav_pos, uav_yaw, False, timestamp
            )
        else:
            dx, dy, dz, dyaw = self._pid.compute(
                target_pos, uav_pos, uav_yaw, target_visible, timestamp
            )

        # Add Gaussian noise for diversity
        noise = np.random.normal(0, self._current_style.noise_scale, 4)
        dx += noise[0]
        dy += noise[1]
        dz += noise[2]
        dyaw += noise[3] * 5.0  # scale yaw noise slightly higher

        return (dx, dy, dz, dyaw)

    @property
    def current_style(self) -> ExpertStyle | None:
        return self._current_style
