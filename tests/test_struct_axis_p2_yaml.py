"""P2: three struct system YAMLs differ per §6 CDSL / WCDL / FDLC."""

from __future__ import annotations

from uavlab.common.config import load_resolved_config, _deep_merge
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig, normalize_paper1_struct
from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
from uavlab.paper1.contracts.contract_types import SlowPlan
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config


def _contract(rel: str) -> Paper1ContractConfig:
    base = load_resolved_config("configs/base.yaml")
    case = load_resolved_config("configs/experiments/paper1/cases/c1_g2_m2.yaml")
    sys = load_resolved_config(rel)
    cfg = apply_experiment_presets(_deep_merge(_deep_merge(base, case), sys))
    return Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)


def test_cdsl_conservative_fixed_no_backlog_coupling():
    c = _contract("configs/experiments/paper1/system/struct_centralized_single_loop.yaml")
    assert normalize_paper1_struct(c.structure) == "cdsl"
    assert c.coupling_mode == "periodic_goal"
    assert c.enable_event_feedback is False
    assert c.use_comm_in_fast is False
    assert not c.return_policy.enable_backlog_gates
    assert not c.return_policy.enable_upload_stuck_recovery
    assert bool(c.fast_to_slow.get("send_backlog")) is False
    assert float(c.struct_profile.control_cmd_min_ratio) >= 0.9


def test_wcdl_no_backlog_mode_f2s():
    c = _contract("configs/experiments/paper1/system/struct_decoupled_dual_loop.yaml")
    assert normalize_paper1_struct(c.structure) == "wcdl"
    assert c.coupling_mode == "full_coupling"
    assert c.enable_event_feedback is True
    assert c.use_comm_in_fast is False
    assert not c.return_policy.enable_backlog_gates
    assert bool(c.fast_to_slow.get("send_backlog")) is False
    assert bool(c.fast_to_slow.get("send_mode")) is False
    assert bool(c.fast_to_slow.get("send_completion")) is True


def test_wcdl_coupling_strips_backlog_from_packet():
    base = load_resolved_config("configs/base.yaml")
    sys = load_resolved_config("configs/experiments/paper1/system/struct_decoupled_dual_loop.yaml")
    cfg = apply_experiment_presets(_deep_merge(base, sys))
    scene_path = str(cfg.get("scene_file", "configs/scenes/g1_uniform.yaml"))
    scene_geom = load_scene_config(scene_path)
    scene_raw = load_scene_yaml(scene_path)
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)
    env.reset(seed=0)
    env.backlog_bits = 4_000_000.0
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    gid = int(env.pois[0].poi_id)
    pkt = CouplingPolicy().build_fast_to_slow_packet(
        contract=contract,
        env=env,
        step=1,
        link_loss_p=0.1,
        plan=SlowPlan(goal_id=gid, goal_ne=(0.0, 0.0)),
    )
    assert pkt.backlog_bits is None
    assert pkt.mode is None
    assert pkt.completion is not None


def test_yaml_env_overrides_arch_preset_fixed_send_ratio():
    """Experiment YAML ``env`` must win over ARCH_VARIANT_PRESETS (not only paper1_loops)."""
    base = load_resolved_config("configs/base.yaml")
    sys = load_resolved_config("configs/experiments/paper1/system/struct_decoupled_dual_loop.yaml")
    sys = _deep_merge(sys, {"env": {"fast_upload_mode": "fixed", "fixed_send_ratio": 0.42}})
    cfg = apply_experiment_presets(_deep_merge(base, sys))
    assert float(cfg["env"]["fixed_send_ratio"]) == 0.42


def test_fdlc_full_backlog_policy():
    c = _contract("configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml")
    assert normalize_paper1_struct(c.structure) == "fdlc"
    assert c.use_comm_in_fast is True
    assert c.return_policy.enable_backlog_gates
    assert c.return_policy.enable_upload_stuck_recovery
    assert int(c.return_policy.backlog_hard_poi_count) == 5
    assert bool(c.fast_to_slow.get("send_backlog")) is True
    assert bool(c.fast_to_slow.get("send_mode")) is True
    ev = c.slow_loop_triggers.get("event_triggers") or {}
    assert bool(ev.get("on_link_drop")) is True
