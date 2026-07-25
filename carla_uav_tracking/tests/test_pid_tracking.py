"""Unit tests for PID visual servoing — runs without CARLA.

Usage:
    python -m pytest tests/test_pid_tracking.py -v
"""

import numpy as np
import pytest

from carla_uav.pid_controller import (
    PIDGains,
    PIDVisualServo,
    PIDVisualServoConfig,
    VelocityLimits,
    _normalize_angle,
)


@pytest.fixture
def default_config() -> PIDVisualServoConfig:
    return PIDVisualServoConfig(
        xy=PIDGains(kp=2.0, ki=0.05, kd=0.3),
        z=PIDGains(kp=1.2, ki=0.05, kd=0.2, output_max=5.0),
        yaw=PIDGains(kp=1.5, ki=0.02, kd=0.1, output_max=60.0),
        limits=VelocityLimits(xy_max=15.0, z_max=5.0, yaw_rate_max=60.0),
        viewing_height_offset=15.0,
        dt=0.1,
        dead_reckon_timeout=5.0,
    )


class TestNormalizeAngle:
    def test_zero(self):
        assert _normalize_angle(0.0) == 0.0

    def test_wrap_positive(self):
        assert _normalize_angle(370.0) == 10.0

    def test_wrap_negative(self):
        assert _normalize_angle(-190.0) == 170.0

    def test_in_range(self):
        assert _normalize_angle(45.0) == 45.0


class TestPIDConvergence:
    """Test that PID converges to a stationary target."""

    def test_stationary_target_converges(self, default_config):
        pid = PIDVisualServo(default_config)
        pid.reset()

        target = np.array([100.0, 50.0, 10.0])  # stationary target
        uav = np.array([0.0, 0.0, 25.0])        # start 100m away
        uav_yaw = 0.0

        positions = [uav.copy()]
        for t in np.arange(0, 30.0, 0.1):  # 30 seconds
            dx, dy, dz, dyaw = pid.compute(
                target, uav, uav_yaw, target_visible=True, timestamp=t
            )
            uav = uav + np.array([dx, dy, dz])
            uav_yaw += dyaw
            positions.append(uav.copy())

        # After 30s, UAV should be near desired position
        desired = target + np.array([0, 0, default_config.viewing_height_offset])
        final_distance = np.linalg.norm(uav - desired)
        assert final_distance < 10.0, f"PID failed to converge: distance={final_distance:.1f}m"

    def test_z_convergence(self, default_config):
        """Test altitude control converges independently."""
        pid = PIDVisualServo(default_config)
        pid.reset()

        target = np.array([50.0, 50.0, 10.0])
        uav = np.array([50.0, 50.0, 80.0])  # too high
        uav_yaw = 45.0

        errors_z = []
        for t in np.arange(0, 20.0, 0.1):
            dx, dy, dz, dyaw = pid.compute(
                target, uav, uav_yaw, target_visible=True, timestamp=t
            )
            uav[2] += dz
            errors_z.append(abs(uav[2] - (target[2] + default_config.viewing_height_offset)))

        # Error should decrease over time
        assert errors_z[-1] < errors_z[0] * 0.5, \
            f"Altitude error not decreasing: {errors_z[0]:.1f} -> {errors_z[-1]:.1f}"


class TestDeadReckoning:
    """Test dead-reckoning behavior when target is lost."""

    def test_dead_reckon_pursues_predicted_position(self, default_config):
        """When target is lost, dead-reckoning predicts where it should be
        and the PID continues to pursue that predicted position."""
        pid = PIDVisualServo(default_config)
        pid.reset()

        target = np.array([10.0, 0.0, 5.0])
        uav = np.array([0.0, 0.0, 20.0])
        uav_yaw = 0.0

        # First, see the target moving east at ~10 m/s for 3 frames
        for t in [0.0, 0.1, 0.2]:
            pid.compute(target + np.array([t * 10, 0, 0]),
                        uav, uav_yaw, target_visible=True, timestamp=t)

        # Now lose it — UAV was at origin, last known target at [12, 0, 5]
        # Dead-reckoning predicts target continues eastward (velocity ~[10, 0, 0])
        positions = [uav.copy()]
        for t in np.arange(0.3, 3.0, 0.1):
            dx, dy, dz, dyaw = pid.compute(
                None, uav, uav_yaw, target_visible=False, timestamp=t
            )
            uav = uav + np.array([dx, dy, dz])
            positions.append(uav.copy())

        # UAV should have moved eastward, chasing the dead-reckoning prediction
        total_displacement = positions[-1] - positions[0]
        assert total_displacement[0] > 5.0, (
            f"UAV should pursue eastward under dead-reckoning: "
            f"displacement x={total_displacement[0]:.1f}m"
        )

    def test_dead_reckon_returns_zero_after_timeout(self, default_config):
        """After dead_reckon_timeout, returns zero action to coast."""
        pid = PIDVisualServo(default_config)
        pid.reset()

        target = np.array([10.0, 0.0, 5.0])
        uav = np.array([0.0, 0.0, 20.0])
        uav_yaw = 0.0

        # See the target once
        pid.compute(target, uav, uav_yaw, target_visible=True, timestamp=0.0)

        # Lost for longer than timeout
        dx, dy, dz, dyaw = pid.compute(
            None, uav, uav_yaw, target_visible=False, timestamp=10.0
        )
        assert dx == 0.0 and dy == 0.0 and dz == 0.0 and dyaw == 0.0, (
            f"Should coast after timeout, got dx={dx:.2f}"
        )


class TestYawControl:
    """Test yaw tracking behavior."""

    def test_yaw_faces_target(self, default_config):
        """UAV starts facing east (0°). Target is north → yaw should converge to ~90°.

        CARLA yaw convention: 0°=east, 90°=north, ±180°=west, -90°=south.
        arctan2(dy, dx) with target north of UAV → 90°.
        """
        pid = PIDVisualServo(default_config)
        pid.reset()

        # UAV at origin, facing east (0°), target 100m north
        target = np.array([0.0, 100.0, 5.0])
        uav = np.array([0.0, 0.0, 20.0])
        uav_yaw = 0.0  # facing east

        for t in np.arange(0, 10.0, 0.1):
            dx, dy, dz, dyaw = pid.compute(
                target, uav, uav_yaw, target_visible=True, timestamp=t
            )
            uav_yaw += dyaw

        # Should face approximately north (90°) after tracking
        yaw_error = abs(_normalize_angle(uav_yaw - 90.0))
        assert yaw_error < 30.0, (
            f"Yaw should face north (~90°), got {uav_yaw:.1f}°, error={yaw_error:.1f}°"
        )


class TestNoTargetEverSeen:
    """Test behavior when no target has ever been observed."""

    def test_returns_zero(self, default_config):
        pid = PIDVisualServo(default_config)
        pid.reset()

        uav = np.array([0.0, 0.0, 20.0])
        dx, dy, dz, dyaw = pid.compute(
            None, uav, 45.0, target_visible=False, timestamp=0.0
        )

        assert dx == 0.0 and dy == 0.0 and dz == 0.0 and dyaw == 0.0, \
            "Should return zero when no target has ever been seen"
