"""§4.2.2 coupling axis: three profiles extend struct FDLC; only two paper dimensions differ."""

from __future__ import annotations

from pathlib import Path

from uavlab.common.config import _deep_merge, load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig, normalize_paper1_struct
from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
from uavlab.paper1.contracts.contract_types import SlowPlan
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config

_STRUCT_FDLC = "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml"
_CASE = "configs/experiments/paper1/cases/c2_g2_m2.yaml"


def _cfg(system_rel: str) -> dict:
    base = load_resolved_config("configs/base.yaml")
    case = load_resolved_config(_CASE)
    sys = load_resolved_config(system_rel)
    return apply_experiment_presets(_deep_merge(_deep_merge(base, case), sys))


def _contract(system_rel: str) -> Paper1ContractConfig:
    return Paper1ContractConfig.from_cfg(_cfg(system_rel), waypoint_delta_max_m=5.0)


def test_coupling_full_matches_struct_fdlc_contract():
    cs = _contract(_STRUCT_FDLC)
    cf = _contract("configs/experiments/paper1/system/coupling_full_coupling.yaml")
    assert normalize_paper1_struct(cf.structure) == "fdlc"
    assert cf.coupling_mode == "full_coupling"
    assert cf.enable_fast_mode_switch is True
    assert cf.allow_mode_switching is True
    assert cf.enable_event_feedback is True
    assert bool(cf.fast_to_slow.get("send_backlog")) is True
    assert bool(cf.fast_to_slow.get("send_mode")) is True
    assert cs.coupling_mode == cf.coupling_mode
    assert cs.enable_fast_mode_switch == cf.enable_fast_mode_switch
    assert cs.return_policy.enable_backlog_gates == cf.return_policy.enable_backlog_gates
    assert str(_cfg(_STRUCT_FDLC)["env"]["fast_upload_mode"]) == "policy"
    assert str(_cfg("configs/experiments/paper1/system/coupling_full_coupling.yaml")["env"]["fast_upload_mode"]) == "policy"


def test_coupling_upload_modes_match_feedback_strength():
    for rel in (
        "configs/experiments/paper1/system/coupling_periodic_goal.yaml",
        "configs/experiments/paper1/system/coupling_event_driven_goal.yaml",
    ):
        cfg = _cfg(rel)
        c = _contract(rel)
        assert normalize_paper1_struct(c.structure) == "fdlc"
        assert str(cfg["env"]["fast_upload_mode"]) == "fixed"
        assert float(cfg["env"]["fixed_send_ratio"]) == 0.2
        assert c.return_policy.enable_backlog_gates is True
        assert c.return_policy.enable_upload_stuck_recovery is True

    for rel in (
        "configs/experiments/paper1/system/coupling_full_coupling.yaml",
    ):
        cfg = _cfg(rel)
        c = _contract(rel)
        assert normalize_paper1_struct(c.structure) == "fdlc"
        assert str(cfg["env"]["fast_upload_mode"]) == "policy"
        assert c.return_policy.enable_backlog_gates is True
        assert c.return_policy.enable_upload_stuck_recovery is True


def test_periodic_goal_dim1_dim2():
    c = _contract("configs/experiments/paper1/system/coupling_periodic_goal.yaml")
    cfg = _cfg("configs/experiments/paper1/system/coupling_periodic_goal.yaml")
    assert c.coupling_mode == "periodic_goal"
    assert c.enable_event_feedback is False
    assert c.enable_fast_mode_switch is False
    assert c.allow_mode_switching is False
    assert c.enable_goal_lock is False
    assert str(cfg["env"]["fast_upload_mode"]) == "fixed"
    assert float(cfg["env"]["fixed_send_ratio"]) == 0.2
    assert str(c.slow_loop_triggers.get("replan_trigger_policy")) == "periodic"
    assert bool(c.fast_to_slow.get("send_completion")) is False
    assert bool(c.fast_to_slow.get("send_backlog")) is False
    assert int(cfg["experiment"]["sweep"]["slow_interval_steps"]) == 200


def test_event_driven_goal_partial_f2s_no_fsm():
    c = _contract("configs/experiments/paper1/system/coupling_event_driven_goal.yaml")
    cfg = _cfg("configs/experiments/paper1/system/coupling_event_driven_goal.yaml")
    assert c.coupling_mode == "event_driven_goal"
    assert c.enable_event_feedback is True
    assert c.enable_fast_mode_switch is False
    assert c.allow_mode_switching is False
    assert c.enable_goal_lock is True
    assert str(cfg["env"]["fast_upload_mode"]) == "fixed"
    assert float(cfg["env"]["fixed_send_ratio"]) == 0.2
    assert str(c.slow_loop_triggers.get("replan_trigger_policy")) == "hybrid"
    assert bool(c.fast_to_slow.get("send_completion")) is True
    assert bool(c.fast_to_slow.get("send_backlog")) is False
    assert bool(c.fast_to_slow.get("send_mode")) is False


def test_periodic_coupling_policy_strips_f2s_packet():
    cfg = _cfg("configs/experiments/paper1/system/coupling_periodic_goal.yaml")
    scene_path = str(cfg.get("scene_file", "configs/scenes/g2_cluster_m2.yaml"))
    env = build_env(from_resolved_config(cfg, load_scene_yaml(scene_path)), load_scene_config(scene_path))
    env.reset(seed=0)
    env.backlog_bits = 4_000_000.0
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    gid = int(env.pois[0].poi_id)
    pkt = CouplingPolicy(mode=contract.coupling_mode).build_fast_to_slow_packet(
        contract=contract,
        env=env,
        step=1,
        link_loss_p=0.1,
        plan=SlowPlan(goal_id=gid, goal_ne=(0.0, 0.0)),
    )
    assert pkt.completion is None
    assert pkt.link is None
    assert pkt.backlog_bits is None
    assert pkt.safety is None


def test_event_coupling_policy_strips_backlog_mode():
    cfg = _cfg("configs/experiments/paper1/system/coupling_event_driven_goal.yaml")
    scene_path = str(cfg.get("scene_file", "configs/scenes/g2_cluster_m2.yaml"))
    env = build_env(from_resolved_config(cfg, load_scene_yaml(scene_path)), load_scene_config(scene_path))
    env.reset(seed=0)
    env.backlog_bits = 4_000_000.0
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    gid = int(env.pois[0].poi_id)
    pkt = CouplingPolicy(mode=contract.coupling_mode).build_fast_to_slow_packet(
        contract=contract,
        env=env,
        step=1,
        link_loss_p=0.1,
        plan=SlowPlan(goal_id=gid, goal_ne=(0.0, 0.0)),
    )
    assert pkt.completion is not None
    assert pkt.backlog_bits is None
    assert pkt.mode is None


def test_sweep_resolve_ts_uses_preset_for_periodic(tmp_path: Path):
    from scripts.sweep import _resolve_slow_interval_steps, _write_axis_combined_config

    root = Path(__file__).resolve().parents[1]
    probe = tmp_path / "c2_g2_m2__coupling_periodic_goal" / ".probe_combined.yaml"
    probe.parent.mkdir(parents=True)
    _write_axis_combined_config(
        out_path=probe,
        case_path=str(root / _CASE),
        system_path=str(root / "configs/experiments/paper1/system/coupling_periodic_goal.yaml"),
        experiment_id="c2_g2_m2__coupling_periodic_goal",
    )
    ts = _resolve_slow_interval_steps(combined_cfg_path=probe, plan_default=40)
    assert ts == 200
