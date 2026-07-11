"""§4.2.2 modeling axis: comm / energy / joint under fixed FDLC + full_coupling."""

from __future__ import annotations

from uavlab.common.config import _deep_merge, load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig, normalize_paper1_struct

_CASE = "configs/experiments/paper1/cases/c2_g2_m2.yaml"
_STRUCT_FDLC = "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml"

_MODELING_YAMLS = (
    "configs/experiments/paper1/system/modelling_comm_aware_decision.yaml",
    "configs/experiments/paper1/system/modelling_energy_aware_decision.yaml",
    "configs/experiments/paper1/system/modelling_comm_energy_aware_decision.yaml",
)


def _cfg_sweep(modeling_rel: str) -> dict:
    """Mirror scripts/sweep.py: case + system (system extends struct FDLC)."""
    case = load_resolved_config(_CASE)
    modeling = load_resolved_config(modeling_rel)
    return apply_experiment_presets(_deep_merge(case, modeling))


def _contract_sweep(modeling_rel: str) -> Paper1ContractConfig:
    return Paper1ContractConfig.from_cfg(_cfg_sweep(modeling_rel), waypoint_delta_max_m=5.0)


def _contract_struct_fdlc() -> Paper1ContractConfig:
    case = load_resolved_config(_CASE)
    struct = load_resolved_config(_STRUCT_FDLC)
    return Paper1ContractConfig.from_cfg(
        apply_experiment_presets(_deep_merge(case, struct)),
        waypoint_delta_max_m=5.0,
    )


def test_modeling_axis_fixed_struct_and_coupling():
    for rel in _MODELING_YAMLS:
        c = _contract_sweep(rel)
        assert normalize_paper1_struct(c.structure) == "fdlc"
        assert c.coupling_mode == "full_coupling"
        assert c.enable_fast_mode_switch is True
        assert c.enable_event_feedback is True
        assert bool(c.fast_to_slow.get("send_backlog")) is True


def test_comm_aware_decision_paper_operationalization():
    c = _contract_sweep("configs/experiments/paper1/system/modelling_comm_aware_decision.yaml")
    assert c.use_comm_in_slow is True
    assert c.use_energy_in_slow is False
    assert c.use_comm_in_fast is True
    assert c.use_energy_in_fast is False
    assert c.comm_objective is True
    assert c.comm_path_quality_mask is True
    assert c.energy_budget_constraint is False
    assert c.energy_hard_constraint is False
    assert bool(c.struct_profile.use_struct_comm_profile) is True
    assert float(c.struct_profile.comm_ret_objective_weight) > 0.0
    assert float(c.struct_profile.energy_plan_margin_frac) == 0.0
    assert c.return_policy.enable_backlog_gates is True


def test_energy_aware_decision_paper_operationalization():
    c = _contract_sweep("configs/experiments/paper1/system/modelling_energy_aware_decision.yaml")
    assert c.use_comm_in_slow is False
    assert c.use_energy_in_slow is True
    assert c.use_comm_in_fast is False
    assert c.use_energy_in_fast is True
    assert c.comm_objective is False
    assert c.comm_path_quality_mask is False
    assert c.energy_budget_constraint is True
    assert c.energy_hard_constraint is True
    assert bool(c.struct_profile.use_struct_comm_profile) is False
    assert float(c.struct_profile.comm_ret_objective_weight) == 0.0
    assert int(c.struct_profile.candidate_degrade_max_level) == 0
    assert c.return_policy.enable_backlog_gates is False
    assert c.return_policy.enable_upload_stuck_recovery is False


def test_comm_energy_joint_b1_return_home_on_modeling_axis():
    joint = _contract_sweep("configs/experiments/paper1/system/modelling_comm_energy_aware_decision.yaml")
    fdlc = _contract_struct_fdlc()
    assert joint.use_comm_in_slow is True
    assert joint.use_energy_in_slow is True
    assert joint.use_comm_in_fast is True
    assert joint.use_energy_in_fast is True
    assert joint.energy_hard_constraint is True
    assert joint.energy_budget_constraint is False
    assert fdlc.energy_budget_constraint is True
    assert joint.return_policy.enable_backlog_gates == fdlc.return_policy.enable_backlog_gates is True
    assert joint.struct_profile == fdlc.struct_profile


def test_struct_coupling_axis_still_uses_full_energy_constraint():
    fdlc = _contract_struct_fdlc()
    assert fdlc.energy_budget_constraint is True
    assert fdlc.energy_hard_constraint is True


def test_modeling_sweep_yaml_only_differs_on_modeling_keys():
    comm_cfg = _cfg_sweep("configs/experiments/paper1/system/modelling_comm_aware_decision.yaml")
    energy_cfg = _cfg_sweep("configs/experiments/paper1/system/modelling_energy_aware_decision.yaml")
    joint_cfg = _cfg_sweep("configs/experiments/paper1/system/modelling_comm_energy_aware_decision.yaml")
    assert comm_cfg["experiment"]["paper1"]["modeling"] == "comm_aware_decision"
    assert energy_cfg["experiment"]["paper1"]["modeling"] == "energy_aware_decision"
    assert joint_cfg["paper1_loops"]["modeling"]["energy_constraint"] == "return_home"
    assert comm_cfg["experiment"]["paper1"]["struct"] == "fdlc"
    assert comm_cfg["experiment"]["paper1"]["coupling"] == "full_coupling"
    assert str(comm_cfg["env"]["fast_upload_mode"]) == "policy"
