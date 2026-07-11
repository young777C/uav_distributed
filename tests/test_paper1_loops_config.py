from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from uavlab.common.config import _deep_merge, load_resolved_config
from uavlab.experiments.presets import (
    apply_experiment_presets,
    normalize_paper1_modeling,
    normalize_paper1_struct,
)
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import SlowPlan
from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.slow.slow_loop import SlowLoop
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config


def test_normalize_paper1_modeling_legacy():
    assert normalize_paper1_modeling("full_model") == "comm_energy_aware_decision"
    assert normalize_paper1_modeling("task_comm") == "comm_aware_decision"
    assert normalize_paper1_modeling("task_energy") == "energy_aware_decision"


def test_contract_comm_energy_aware_decision_sanity():
    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/sanity/c1_g1_comm_energy_aware_decision.yaml")
    )
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert c.structure == "fdlc"
    assert c.use_comm_in_slow is True
    assert c.use_energy_in_slow is True
    assert c.use_comm_in_fast is True
    assert c.use_energy_in_fast is True
    assert c.comm_path_quality_mask is True
    assert c.energy_hard_constraint is True
    assert c.energy_budget_constraint is False


def test_contract_comm_aware_decision_semantics():
    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/sanity/c1_g1_comm_aware_decision.yaml")
    )
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert c.use_comm_in_slow is True
    assert c.use_energy_in_slow is False
    assert c.use_comm_in_fast is True
    assert c.use_energy_in_fast is False
    assert c.comm_path_quality_mask is True
    assert c.energy_budget_constraint is False
    assert c.energy_hard_constraint is False
    assert bool(c.struct_profile.use_struct_comm_profile) is True
    assert float(c.struct_profile.energy_plan_margin_frac) == 0.0


def test_contract_energy_aware_decision_semantics():
    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/sanity/c1_g1_energy_aware_decision.yaml")
    )
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert c.use_comm_in_slow is False
    assert c.use_energy_in_slow is True
    assert c.use_comm_in_fast is False
    assert c.use_energy_in_fast is True
    assert c.comm_path_quality_mask is False
    assert c.energy_budget_constraint is True
    assert bool(c.struct_profile.use_struct_comm_profile) is False
    assert float(c.struct_profile.comm_ret_objective_weight) == 0.0
    assert c.return_policy.enable_backlog_gates is False


def test_semantics_disable_event_feedback():
    cfg = {
        "paper1_loops": {
            "semantics": {"enable_event_feedback": False},
            "coupling_mode": "full_coupling",
            "fast_to_slow": {
                "send_completion": True,
                "send_link_stats": "full",
                "send_backlog": True,
                "send_mode": True,
                "send_safety_events": True,
            },
        }
    }
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert c.enable_event_feedback is False


def test_slow_loop_importable():
    assert SlowLoop is not None


def test_structure_cdsl_operationalization():
    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/system/struct_centralized_single_loop.yaml")
    )
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=0.0)
    assert normalize_paper1_struct(c.structure) == "cdsl"
    assert c.slow_policy == "periodic_or_event_replan"
    assert c.coupling_mode == "periodic_goal"
    assert c.enable_event_feedback is False
    assert c.enable_fast_mode_switch is False
    assert c.use_comm_in_fast is False
    assert c.use_energy_in_fast is False
    assert c.use_comm_in_slow is True
    assert c.use_energy_in_slow is True
    assert c.comm_objective is True
    assert c.energy_budget_constraint is True
    assert c.allow_waypoint_delta is False
    assert c.allow_mode_switching is False
    assert str(c.slow_loop_triggers.get("replan_trigger_policy")) == "periodic"
    assert str(c.slow_loop_triggers.get("periodic_replan_scope")) == "repeat"
    assert c.struct_profile.control_cmd_min_ratio == 0.95
    assert c.struct_profile.use_data_return_hard_mask is True


def test_structure_wcdl_operationalization():
    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/system/struct_decoupled_dual_loop.yaml")
    )
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert normalize_paper1_struct(c.structure) == "wcdl"
    assert c.slow_policy == "periodic_or_event_replan"
    assert c.enable_event_feedback is True
    assert c.enable_fast_mode_switch is True
    assert c.use_comm_in_fast is False
    assert c.use_energy_in_fast is False
    assert str(c.slow_loop_triggers.get("replan_trigger_policy")) == "hybrid"
    assert str(c.slow_loop_triggers.get("periodic_replan_scope")) == "repeat"


def test_struct_sweep_wcdl_fast_semantics_not_overwritten_by_full_modeling():
    """Mirror struct-axis sweep: base + case + system YAML with fixed comm_energy modeling."""
    base = load_resolved_config("configs/base.yaml")
    case = load_resolved_config("configs/experiments/paper1/cases/c1_g2_m2.yaml")
    sys = load_resolved_config("configs/experiments/paper1/system/struct_decoupled_dual_loop.yaml")
    cfg = apply_experiment_presets(_deep_merge(_deep_merge(base, case), sys))
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert normalize_paper1_struct(c.structure) == "wcdl"
    assert c.use_comm_in_fast is False
    assert c.use_energy_in_fast is False
    assert c.use_comm_in_slow is True
    assert c.use_energy_in_slow is True


def test_struct_sweep_fdlc_fast_semantics():
    base = load_resolved_config("configs/base.yaml")
    case = load_resolved_config("configs/experiments/paper1/cases/c1_g2_m2.yaml")
    sys = load_resolved_config("configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml")
    cfg = apply_experiment_presets(_deep_merge(_deep_merge(base, case), sys))
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert normalize_paper1_struct(c.structure) == "fdlc"
    assert c.use_comm_in_fast is True
    assert c.use_energy_in_fast is True


def test_modeling_sweep_comm_aware_overrides_fdlc_fast_flags():
    base = load_resolved_config("configs/base.yaml")
    case = load_resolved_config("configs/experiments/paper1/cases/c1_g2_m0.yaml")
    profile = load_resolved_config("configs/experiments/paper1/sanity/c1_g1_comm_aware_decision.yaml")
    cfg = apply_experiment_presets(_deep_merge(_deep_merge(base, case), profile))
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert normalize_paper1_struct(c.structure) == "fdlc"
    assert c.use_comm_in_fast is True
    assert c.use_energy_in_fast is False


def test_structure_fdlc_operationalization():
    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml")
    )
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert normalize_paper1_struct(c.structure) == "fdlc"
    assert c.slow_policy == "periodic_or_event_replan"
    assert c.enable_event_feedback is True
    assert c.enable_fast_mode_switch is True
    assert c.use_comm_in_fast is True
    assert c.use_energy_in_fast is True
    assert str(c.slow_loop_triggers.get("replan_trigger_policy")) == "hybrid"
    assert str(c.slow_loop_triggers.get("periodic_replan_scope")) == "repeat"
    ev = c.slow_loop_triggers.get("event_triggers") or {}
    assert bool(ev.get("on_link_drop")) is True
    assert bool(ev.get("on_control_link_lost")) is True
    assert c.struct_profile.control_cmd_min_ratio == 0.70
    assert c.struct_profile.candidate_degrade_max_level == 3


def test_struct_axis_shared_slow_modeling():
    """Struct sweep: identical comm+energy slow modeling; only struct profile / coupling / fast differ."""
    paths = (
        "configs/experiments/paper1/system/struct_centralized_single_loop.yaml",
        "configs/experiments/paper1/system/struct_decoupled_dual_loop.yaml",
        "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml",
    )
    contracts = []
    for rel in paths:
        cfg = apply_experiment_presets(load_resolved_config(rel))
        contracts.append(Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0))

    for c in contracts:
        assert c.use_comm_in_slow is True
        assert c.use_energy_in_slow is True
        assert c.comm_objective is True
        assert c.energy_budget_constraint is True
        assert c.energy_hard_constraint is True
        assert bool(c.struct_profile.use_struct_comm_profile) is True

    cdsl, wcdl, fdlc = contracts
    assert cdsl.struct_profile.control_cmd_min_ratio >= wcdl.struct_profile.control_cmd_min_ratio
    assert wcdl.struct_profile.control_cmd_min_ratio >= fdlc.struct_profile.control_cmd_min_ratio
    assert cdsl.coupling_mode == "periodic_goal"
    assert wcdl.coupling_mode == "full_coupling"
    assert fdlc.coupling_mode == "full_coupling"


def test_struct_profile_monotonicity_from_contract():
    for key in ("cdsl", "wcdl", "fdlc"):
        cfg = apply_experiment_presets(
            load_resolved_config(
                f"configs/experiments/paper1/system/struct_{'centralized_single_loop' if key == 'cdsl' else 'decoupled_dual_loop' if key == 'wcdl' else 'full_dual_loop_distributed'}.yaml"
            )
        )
        c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
        assert normalize_paper1_struct(c.structure) == key
    cdsl = Paper1ContractConfig.from_cfg(
        apply_experiment_presets(
            load_resolved_config("configs/experiments/paper1/system/struct_centralized_single_loop.yaml")
        ),
        waypoint_delta_max_m=5.0,
    ).struct_profile
    wcdl = Paper1ContractConfig.from_cfg(
        apply_experiment_presets(
            load_resolved_config("configs/experiments/paper1/system/struct_decoupled_dual_loop.yaml")
        ),
        waypoint_delta_max_m=5.0,
    ).struct_profile
    fdlc = Paper1ContractConfig.from_cfg(
        apply_experiment_presets(
            load_resolved_config("configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml")
        ),
        waypoint_delta_max_m=5.0,
    ).struct_profile
    assert cdsl.control_cmd_min_ratio >= wcdl.control_cmd_min_ratio >= fdlc.control_cmd_min_ratio
    assert cdsl.energy_plan_margin_frac >= wcdl.energy_plan_margin_frac >= fdlc.energy_plan_margin_frac


def test_coupling_periodic_goal_strips_feedback_fields():
    base = load_resolved_config("configs/base.yaml")
    ov = load_resolved_config("configs/experiments/paper1/system/coupling_periodic_goal.yaml")
    cfg = apply_experiment_presets(_deep_merge(base, ov))
    scene_path = str(cfg.get("scene_file", "configs/scenes/g1_uniform.yaml"))
    scene_raw = load_scene_yaml(scene_path)
    scene_geom = load_scene_config(scene_path)
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)
    _ = env.reset(seed=0)
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=float(sim_cfg.waypoint_delta_max_m))
    assert contract.coupling_mode == "periodic_goal"
    assert contract.enable_event_feedback is False
    coup = CouplingPolicy(mode=str(contract.coupling_mode))
    gid = int(env.pois[0].poi_id)
    plan = SlowPlan(goal_id=gid, goal_ne=(float(env.pois[0].pos_ne[0]), float(env.pois[0].pos_ne[1])))
    pkt = coup.build_fast_to_slow_packet(
        contract=contract,
        env=env,
        step=0,
        link_loss_p=0.2,
        plan=plan,
    )
    assert pkt.completion is None
    assert pkt.link is None


def test_coupling_v2_periodic_profile():
    base = load_resolved_config("configs/base.yaml")
    ov = load_resolved_config("configs/experiments/paper1/system/coupling_periodic_goal.yaml")
    cfg = apply_experiment_presets(_deep_merge(base, ov))
    assert float(cfg.get("env", {}).get("fixed_send_ratio")) == 0.2
    assert int(cfg.get("experiment", {}).get("sweep", {}).get("slow_interval_steps")) == 200
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert str(c.slow_loop_triggers.get("replan_trigger_policy")) == "periodic"
    assert c.enable_event_feedback is False


def test_coupling_v2_event_hybrid_p0():
    base = load_resolved_config("configs/base.yaml")
    ov = load_resolved_config("configs/experiments/paper1/system/coupling_event_driven_goal.yaml")
    cfg = apply_experiment_presets(_deep_merge(base, ov))
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert str(c.slow_loop_triggers.get("replan_trigger_policy")) == "hybrid"
    assert float(cfg.get("env", {}).get("fixed_send_ratio")) == 0.2
    assert c.enable_fast_mode_switch is False
    assert float(c.slow_loop_triggers.get("replan_cooldown_s")) == 15.0
    f2s = dict(c.fast_to_slow or {})
    assert bool(f2s.get("send_backlog")) is False
    assert bool(f2s.get("send_mode")) is False


def test_coupling_v2_full_hybrid_fsm():
    base = load_resolved_config("configs/base.yaml")
    ov = load_resolved_config("configs/experiments/paper1/system/coupling_full_coupling.yaml")
    cfg = apply_experiment_presets(_deep_merge(base, ov))
    c = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    assert str(c.slow_loop_triggers.get("replan_trigger_policy")) == "hybrid"
    assert c.enable_fast_mode_switch is True
    assert c.enable_goal_lock is True
    f2s = dict(c.fast_to_slow or {})
    assert bool(f2s.get("send_backlog")) is True
    assert bool(f2s.get("send_mode")) is True


def _run_paper1_smoke(config_rel: str) -> None:
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "uavlab.paper1.runner.run",
            "--config",
            config_rel,
            "--episodes",
            "1",
            "--seed",
            "0",
        ],
        cwd=str(repo),
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr + r.stdout


def test_paper1_runner_smoke_three_modeling_profiles():
    for rel in (
        "configs/experiments/paper1/sanity/c1_g1_comm_aware_decision.yaml",
        "configs/experiments/paper1/sanity/c1_g1_energy_aware_decision.yaml",
        "configs/experiments/paper1/sanity/c1_g1_comm_energy_aware_decision.yaml",
    ):
        _run_paper1_smoke(rel)
