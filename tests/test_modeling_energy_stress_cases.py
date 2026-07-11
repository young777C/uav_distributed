"""Modeling-axis energy stress cases must not enter struct/coupling sweeps."""

from __future__ import annotations

from pathlib import Path

from uavlab.common.config import load_resolved_config

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "configs/experiments/paper1/cases"


def _struct_axis_case_paths() -> list[Path]:
    return sorted(CASES.glob("*.yaml"))


def test_energy_stress_cases_not_in_struct_glob():
    stress = {
        p.resolve()
        for p in (CASES / "modeling_energy_stress").glob("*.yaml")
    }
    struct_cases = {p.resolve() for p in _struct_axis_case_paths()}
    assert stress.isdisjoint(struct_cases)
    assert len(struct_cases) == 8


def test_c2_energy_stress_loads_and_overrides_energy():
    cfg = load_resolved_config(
        "configs/experiments/paper1/cases/modeling_energy_stress/c2_g2_m2_E210.yaml"
    )
    assert float(cfg["energy"]["E_0_Wh"]) == 210.0
    assert float(cfg["energy"]["P_fly_W"]) == 440.0
    assert cfg["experiment"]["env_case"]["comm_case"] == "C2"
    assert cfg["experiment"]["env_case"]["conflict_level"] == "M2"
    assert cfg["experiment"]["env_case"]["energy_stress_tier"] == "E210"
    assert cfg["scene_file"] == "configs/scenes/g2_cluster_m2.yaml"
