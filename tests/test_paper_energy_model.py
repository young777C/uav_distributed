"""Energy model alignment with UAV_GCS_能量建模计算说明.md §8."""

from __future__ import annotations

from uavlab.common.config import load_config
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.energy_model import (
    WH_TO_J,
    deduct_step_energy,
    energy_need_for_poi_visit,
    resolve_energy_coefficients,
)


def test_resolve_energy_from_physical_params() -> None:
    env = {"v_xy_cruise": 10.0, "v_xy_max": 20.0}
    energy = {
        "use_power_speed_model": True,
        "E_0_Wh": 263.2,
        "P_fly_W": 400.0,
        "P_hov_W": 450.0,
        "eta_safe": 0.20,
    }
    e_m, e_h, eta = resolve_energy_coefficients(env=env, energy=energy)
    e0_j = 263.2 * WH_TO_J
    assert abs(e_m - (400.0 / 10.0) / e0_j) < 1e-12
    assert abs(e_h - 450.0 / e0_j) < 1e-12
    assert eta == 0.20


def test_doc_section_10_example_need_energy() -> None:
    """§10 numeric example: E_need ≈ 81.67 Wh / 263.2 Wh ≈ 0.310 normalized."""

    env = {"v_xy_cruise": 10.0}
    energy = {
        "use_power_speed_model": True,
        "E_0_Wh": 263.2,
        "P_fly_W": 400.0,
        "P_hov_W": 450.0,
        "eta_safe": 0.20,
    }
    e_m, e_h, eta = resolve_energy_coefficients(env=env, energy=energy)
    need = energy_need_for_poi_visit(
        pos_ne=(0.0, 0.0),
        poi_ne=(1000.0, 0.0),
        gcs_ne=(1000.0, 1500.0),
        energy_per_meter=e_m,
        energy_hover_per_s=e_h,
        energy_safe_margin=eta,
        t_hov_s=10.0,
    )
    need_wh = need * 263.2
    assert abs(need_wh - 81.67) < 0.15


def test_deduct_step_energy_hover_vs_flight() -> None:
    e_m, e_h, _ = resolve_energy_coefficients(
        env={"v_xy_cruise": 10.0},
        energy={"use_power_speed_model": True, "E_0_Wh": 236.8, "P_fly_W": 400.0, "P_hov_W": 450.0},
    )
    dt = 0.2
    after_hover = deduct_step_energy(
        remaining_energy=1.0,
        speed_mps=0.0,
        dt=dt,
        traveled_m=0.0,
        energy_per_meter=e_m,
        energy_hover_per_s=e_h,
        hover_speed_threshold_mps=0.5,
    )
    assert abs(after_hover - (1.0 - e_h * dt)) < 1e-12

    after_fly = deduct_step_energy(
        remaining_energy=1.0,
        speed_mps=10.0,
        dt=dt,
        traveled_m=2.0,
        energy_per_meter=e_m,
        energy_hover_per_s=e_h,
        hover_speed_threshold_mps=0.5,
    )
    assert abs(after_fly - (1.0 - e_m * 2.0)) < 1e-12

    # At threshold speed (0.5) use flight branch (strictly below threshold for hover).
    at_thr = deduct_step_energy(
        remaining_energy=1.0,
        speed_mps=0.5,
        dt=dt,
        traveled_m=1.0,
        energy_per_meter=e_m,
        energy_hover_per_s=e_h,
        hover_speed_threshold_mps=0.5,
    )
    assert abs(at_thr - (1.0 - e_m * 1.0)) < 1e-12


def test_base_yaml_loads_doc_energy() -> None:
    cfg = load_config("configs/base.yaml")
    sim = from_resolved_config(cfg, {"poi_list": [], "gcs_ne": [1250.0, 1250.0]})
    assert sim.poi_dwell_s == 5.0
    assert sim.hover_speed_threshold_mps == 0.5
    assert abs(sim.energy_safe_margin - 0.20) < 1e-9
    e0_j = 236.8 * WH_TO_J
    assert abs(sim.energy_per_meter - 40.0 / e0_j) < 1e-10
    assert abs(sim.energy_hover_per_s - 450.0 / e0_j) < 1e-10
