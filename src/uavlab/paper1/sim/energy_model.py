"""
Energy model aligned with ``UAV_GCS_能量建模计算说明.md``.

Physical (SI):
  E_fly = P_fly * d / v_c   (v_c = **average** cruise speed, not instantaneous speed every step)
  E_hov = P_hov * t_hov
  E_safe = eta * E_0

Simulation uses normalized remaining energy in [0, 1] with E_0 as full scale:
  energy_per_meter = (P_fly / v_c) / E_0_J
  energy_hover_per_s = P_hov / E_0_J
  energy_safe_margin = eta

Per simulation step (``Paper1Env.step_fast``), exactly one mode applies:
  - speed < hover_speed_threshold_mps (default 0.5 m/s): E_hov = P_hov * dt
  - else: E_fly = (P_fly / v_c) * traveled_m
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

WH_TO_J = 3600.0

Point2D = Tuple[float, float]


def _dist(a: Point2D, b: Point2D) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def resolve_energy_coefficients(*, env: Dict[str, Any], energy: Dict[str, Any]) -> tuple[float, float, float]:
    """
  Return ``(energy_per_meter, energy_hover_per_s, energy_safe_margin)`` for Paper1SimConfig.

  When ``energy.use_power_speed_model`` is true (default), derive from §8 physical parameters.
  Otherwise use explicit ``energy_per_meter`` / ``energy_hover_per_s`` / ``energy_safe_margin``.
    """

    use_power = bool(energy.get("use_power_speed_model", True))
    if use_power:
        e0_wh = float(energy.get("E_0_Wh", 263.2))
        p_fly = float(energy.get("P_fly_W", 400.0))
        p_hov = float(energy.get("P_hov_W", 450.0))
        eta = float(energy.get("eta_safe", 0.20))
        v_c = float(env.get("v_xy_cruise", env.get("v_xy_max", 10.0)))
        e0_j = e0_wh * WH_TO_J
        if e0_j <= 0.0 or v_c <= 0.0:
            raise ValueError("E_0_Wh and env.v_xy_cruise must be positive for power-speed energy model.")
        e_per_m = (p_fly / v_c) / e0_j
        e_hover = p_hov / e0_j
        margin = eta
        return float(e_per_m), float(e_hover), float(margin)

    e_per_m = float(energy.get("energy_per_meter", 6e-5))
    e_hover = float(energy.get("energy_hover_per_s", 3e-5))
    margin = float(energy.get("energy_safe_margin", energy.get("eta_safe", 0.05)))
    return e_per_m, e_hover, margin


def energy_need_for_poi_visit(
    *,
    pos_ne: Point2D,
    poi_ne: Point2D,
    gcs_ne: Point2D,
    energy_per_meter: float,
    energy_hover_per_s: float,
    energy_safe_margin: float,
    t_hov_s: float,
) -> float:
    """Doc §6: E_go + E_hov + E_ret + E_safe (normalized)."""

    d_to = _dist(pos_ne, poi_ne)
    d_home = _dist(poi_ne, gcs_ne)
    e_go = float(energy_per_meter) * d_to
    e_ret = float(energy_per_meter) * d_home
    e_hov = float(energy_hover_per_s) * float(max(0.0, t_hov_s))
    return float(e_go + e_ret + e_hov + float(energy_safe_margin))


def energy_return_need(
    *,
    pos_ne: Point2D,
    gcs_ne: Point2D,
    energy_per_meter: float,
    energy_safe_margin: float,
) -> float:
    """Doc §11.2 fast return: E_ret + E_safe from current position."""

    d_home = _dist(pos_ne, gcs_ne)
    return float(energy_per_meter) * d_home + float(energy_safe_margin)


def energy_need_from_env(env: Any, *, poi_ne: Point2D, t_hov_s: float) -> float:
    """Convenience wrapper using ``Paper1Env`` / ``Paper1SimConfig``."""

    cfg = env.cfg
    return energy_need_for_poi_visit(
        pos_ne=env.pos_ne,
        poi_ne=poi_ne,
        gcs_ne=cfg.gcs_ne,
        energy_per_meter=cfg.energy_per_meter,
        energy_hover_per_s=cfg.energy_hover_per_s,
        energy_safe_margin=cfg.energy_safe_margin,
        t_hov_s=t_hov_s,
    )


def energy_return_need_from_env(env: Any) -> float:
    cfg = env.cfg
    return energy_return_need(
        pos_ne=env.pos_ne,
        gcs_ne=cfg.gcs_ne,
        energy_per_meter=cfg.energy_per_meter,
        energy_safe_margin=cfg.energy_safe_margin,
    )


def deduct_step_energy(
    *,
    remaining_energy: float,
    speed_mps: float,
    dt: float,
    traveled_m: float,
    energy_per_meter: float,
    energy_hover_per_s: float,
    hover_speed_threshold_mps: float,
) -> float:
    """
    Apply one fast-step energy debit (flight vs hover by instantaneous speed).

    Hover when ``speed_mps < hover_speed_threshold_mps``; otherwise flight by distance.
    """

    thr = float(max(0.0, hover_speed_threshold_mps))
    if float(speed_mps) < thr:
        debit = float(energy_hover_per_s) * float(max(0.0, dt))
    else:
        debit = float(energy_per_meter) * float(max(0.0, traveled_m))
    return float(max(0.0, float(remaining_energy) - debit))
