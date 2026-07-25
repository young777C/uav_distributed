"""carla_uav — Core UAV control modules for CARLA-based tracking data generation."""

# Lazy imports to allow unit testing without CARLA installed.
# Modules that require CARLA (drone, safety) are only imported when used.

from carla_uav.pid_controller import (
    PIDVisualServo,
    PIDVisualServoConfig,
    PIDGains,
    VelocityLimits,
)
from carla_uav.expert_policy import ExpertPolicy, ExpertStyle

__all__ = [
    "PIDVisualServo", "PIDVisualServoConfig", "PIDGains", "VelocityLimits",
    "ExpertPolicy", "ExpertStyle",
]

# CARLA-dependent modules — import explicitly when needed:
#   from carla_uav.drone import KinematicDrone, DroneState
#   from carla_uav.safety import SafetyMonitor, SafetyConfig, SafetyEvent, SafetyResult
