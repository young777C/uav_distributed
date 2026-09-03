"""PID visual servoing controller for UAV target tracking.

Three independent PID loops:
  - XY: horizontal position tracking → keeps target at forward projection center
  - Z:  altitude control → maintains desired viewing height above target
  - Yaw: orientation control → faces the target

With dead-reckoning fallback when target is temporarily lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PIDGains:
    """PID gain triplet with anti-windup clamping."""
    kp: float
    ki: float
    kd: float
    integral_max: float = 5.0   # anti-windup clamp
    output_max: float = 15.0     # m/s or deg/s


@dataclass
class PIDState:
    """Per-loop PID state (reset each episode)."""
    prev_error: float = 0.0
    integral: float = 0.0


@dataclass
class VelocityLimits:
    """Physical velocity constraints for realistic UAV behavior."""
    xy_max: float = 15.0       # m/s horizontal
    z_max: float = 5.0         # m/s vertical
    yaw_rate_max: float = 60.0 # deg/s


@dataclass
class PIDVisualServoConfig:
    """Full configuration for the PID visual servoing controller."""
    xy: PIDGains = field(default_factory=lambda: PIDGains(kp=2.0, ki=0.05, kd=0.3))
    z:  PIDGains = field(default_factory=lambda: PIDGains(kp=1.2, ki=0.05, kd=0.2, output_max=5.0))
    yaw: PIDGains = field(default_factory=lambda: PIDGains(kp=1.5, ki=0.02, kd=0.1, output_max=60.0))
    limits: VelocityLimits = field(default_factory=VelocityLimits)
    viewing_height_offset: float = 15.0  # meters above target
    dt: float = 0.1                      # control period (10 Hz)
    dead_reckon_timeout: float = 5.0     # seconds before abandoning prediction


class PIDVisualServo:
    """Three-layer PID controller for UAV visual tracking.

    Usage:
        pid = PIDVisualServo(config)
        for each timestep:
            action = pid.compute(target_position, uav_position, uav_yaw, target_visible, timestamp)
            drone.step(*action)
    """

    def __init__(self, config: PIDVisualServoConfig):
        self._cfg = config
        self._state_xy = PIDState()
        self._state_z = PIDState()
        self._state_yaw = PIDState()
        self._last_known_target: Optional[np.ndarray] = None
        self._last_known_velocity: np.ndarray = np.zeros(3)
        self._last_seen_at: Optional[float] = None
        self._target_lost_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all PID states for a new episode."""
        self._state_xy = PIDState()
        self._state_z = PIDState()
        self._state_yaw = PIDState()
        self._last_known_target = None
        self._last_known_velocity = np.zeros(3)
        self._last_seen_at = None
        self._target_lost_at = None

    def compute(
        self,
        target_pos: np.ndarray,     # [x, y, z] world POSITION setpoint (may trail the target)
        uav_pos: np.ndarray,        # [x, y, z] world coords
        uav_yaw: float,             # degrees
        target_visible: bool,
        timestamp: float,
        look_pos: np.ndarray | None = None,   # world point the CAMERA faces (yaw); default target_pos
    ) -> tuple[float, float, float, float]:
        """Compute control action: (dx, dy, dz, dyaw) in world frame.

        Returns zero-vector action if no target has ever been seen.
        """
        if target_visible and target_pos is not None:
            # Update dead-reckoning state
            if self._last_known_target is not None:
                self._last_known_velocity = (
                    (target_pos - self._last_known_target) / self._cfg.dt
                )
            self._last_known_target = target_pos.copy()
            self._last_seen_at = timestamp
            self._target_lost_at = None
            tracking_pos = target_pos
        elif self._last_known_target is not None:
            # Measure elapsed time from when target was last SEEN (not first reported lost)
            if self._target_lost_at is None:
                self._target_lost_at = timestamp
            elapsed = timestamp - (self._last_seen_at
                                  if self._last_seen_at is not None
                                  else self._target_lost_at)
            if elapsed > self._cfg.dead_reckon_timeout:
                # Give up — return zero action (coast)
                return (0.0, 0.0, 0.0, 0.0)
            # Decay velocity over time (target probably slowed down)
            decay = max(0.0, 1.0 - elapsed / self._cfg.dead_reckon_timeout)
            tracking_pos = self._last_known_target + self._last_known_velocity * elapsed * decay
        else:
            # No target has ever been seen
            return (0.0, 0.0, 0.0, 0.0)

        # --- Layer 1: XY horizontal tracking ---
        error_xy = tracking_pos[:2] - uav_pos[:2]
        desired_xy_vel = self._pid_step(
            np.linalg.norm(error_xy),
            self._cfg.xy,
            self._state_xy,
        )
        # Convert scalar speed to direction vector
        if np.linalg.norm(error_xy) > 0.01:
            direction_xy = error_xy / np.linalg.norm(error_xy)
        else:
            direction_xy = np.zeros(2)
        vx_des = direction_xy[0] * desired_xy_vel
        vy_des = direction_xy[1] * desired_xy_vel

        # --- Layer 2: Altitude control ---
        desired_z = tracking_pos[2] + self._cfg.viewing_height_offset
        error_z = desired_z - uav_pos[2]
        vz_des = self._pid_step(error_z, self._cfg.z, self._state_z)

        # --- Layer 3: Yaw control ---
        # Only yaw when there's a meaningful HORIZONTAL offset. Near-nadir (target almost
        # directly under the UAV, as when it flies overhead to re-acquire) the atan2 heading
        # is ill-conditioned — tiny target motion flips it wildly → the camera shakes. Hold
        # the yaw inside that deadband instead of chasing the singular heading.
        yaw_ref = look_pos if (look_pos is not None and target_visible) else tracking_pos
        horiz = float(np.hypot(yaw_ref[0] - uav_pos[0], yaw_ref[1] - uav_pos[1]))
        if horiz < 5.0:
            yaw_rate = 0.0
        else:
            desired_yaw = np.degrees(
                np.arctan2(yaw_ref[1] - uav_pos[1], yaw_ref[0] - uav_pos[0])
            )
            error_yaw = _normalize_angle(desired_yaw - uav_yaw)
            yaw_rate = self._pid_step(error_yaw, self._cfg.yaw, self._state_yaw)

        # --- Convert velocities to displacements ---
        dx = np.clip(vx_des * self._cfg.dt, -self._cfg.limits.xy_max * self._cfg.dt,
                      self._cfg.limits.xy_max * self._cfg.dt)
        dy = np.clip(vy_des * self._cfg.dt, -self._cfg.limits.xy_max * self._cfg.dt,
                      self._cfg.limits.xy_max * self._cfg.dt)
        dz = np.clip(vz_des * self._cfg.dt, -self._cfg.limits.z_max * self._cfg.dt,
                      self._cfg.limits.z_max * self._cfg.dt)
        dyaw = np.clip(yaw_rate * self._cfg.dt, -self._cfg.limits.yaw_rate_max * self._cfg.dt,
                        self._cfg.limits.yaw_rate_max * self._cfg.dt)

        return (float(dx), float(dy), float(dz), float(dyaw))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _pid_step(error: float, gains: PIDGains, state: PIDState) -> float:
        """Single PID step with anti-windup clamping.

        Returns: control output (velocity or rate)
        """
        # Proportional
        p = gains.kp * error

        # Integral with anti-windup
        state.integral += error
        state.integral = np.clip(state.integral, -gains.integral_max, gains.integral_max)
        i = gains.ki * state.integral

        # Derivative on measurement (not error) to avoid derivative kick
        d = gains.kd * (error - state.prev_error)
        state.prev_error = error

        # Clamp total output
        return float(np.clip(p + i + d, -gains.output_max, gains.output_max))


def _normalize_angle(angle_deg: float) -> float:
    """Wrap angle to [-180, 180]."""
    return ((angle_deg + 180.0) % 360.0) - 180.0
