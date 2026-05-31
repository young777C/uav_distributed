from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class EnergyBudgetParams:
    """
    Paper-1 Eq. (17)–(19) parameters (normalized fractions of ``E_0``).

    Physical form (``UAV_GCS_能量建模计算说明.md``): ``e_per_meter = (P_fly/v_c)/E_0_J``,
    ``p_hover = P_hov/E_0_J``. Use :func:`uavlab.paper1.sim.energy_model.resolve_energy_coefficients`.
    """

    e_per_meter: float = 6e-5  # e_d  (= P_fly/v_c / E_0 when calibrated)
    p_hover: float = 3e-5  # p_hov (energy per second, normalized)
    t_safe_s: float = 0.0  # t_safe
    t_obs_s: float = 10.0  # t_obs / t_i^hov


def fly_energy(*, distance_m: float, params: EnergyBudgetParams) -> float:
    """
    Paper-1 Eq. (17): E_ij^fly = e_d * d_ij.
    """

    return float(params.e_per_meter) * float(max(0.0, distance_m))


def hover_energy(*, t_obs_s: float, params: EnergyBudgetParams) -> float:
    """
    Paper-1 Eq. (18): E_i^hov = p_hov * (t_obs + t_safe).
    """

    return float(params.p_hover) * float(max(0.0, params.t_obs_s) + max(0.0, params.t_safe_s))


def sequence_energy(
    *,
    edges: Iterable[Tuple[float, int]],
    waits: Iterable[Tuple[float, int]],
    params: EnergyBudgetParams,
) -> float:
    """
    Paper-1 Eq. (19): E_t^seq = Σ E_ij^fly z_ij + Σ E_i^hov y_i.

    Inputs are simplified:
    - edges: iterable of (distance_m, z_ij) where z_ij is 0/1
    - waits: iterable of (t_obs_s, y_i) where y_i is 0/1
    """

    e_fly = 0.0
    for d, z in edges:
        if int(z) != 0:
            e_fly += fly_energy(distance_m=float(d), params=params)

    e_hov = 0.0
    for t_obs, y in waits:
        if int(y) != 0:
            e_hov += hover_energy(t_obs_s=float(t_obs), params=params)

    return float(e_fly + e_hov)

