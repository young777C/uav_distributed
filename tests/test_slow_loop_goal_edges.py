"""Slow-loop goal completion edges: spatial vs effective decoupling."""

from __future__ import annotations

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig, parse_slow_event_triggers
from uavlab.paper1.contracts.contract_types import CompletionStatus, FastToSlowPacket, SlowObservation
from uavlab.paper1.loops.slow.slow_loop import SlowLoop
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config


def _minimal_contract_cfg(*, event_triggers: dict) -> dict:
    return {
        "paper1_loops": {
            "coupling_mode": "full_coupling",
            "slow_policy": "periodic_or_event_replan",
            "slow_loop": {
                "replan_trigger_policy": "hybrid",
                "event_triggers": event_triggers,
            },
            "semantics": {"enable_event_feedback": True, "enable_goal_lock": True},
        }
    }


def test_parse_slow_event_triggers_legacy_effective_only():
    ev = parse_slow_event_triggers({"on_goal_completed": True})
    assert ev["on_goal_spatial_complete"] is False
    assert ev["on_goal_effective_complete"] is True


def test_parse_slow_event_triggers_spatial_only():
    ev = parse_slow_event_triggers(
        {
            "on_goal_spatial_complete": True,
            "on_goal_effective_complete": False,
        }
    )
    assert ev["on_goal_spatial_complete"] is True
    assert ev["on_goal_effective_complete"] is False


def _slow_loop_for_cfg(cfg: dict) -> SlowLoop:
    scene_path = "configs/scenes/g1_uniform.yaml"
    scene_raw = load_scene_yaml(scene_path)
    scene_geom = load_scene_config(scene_path)
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)
    _ = env.reset(seed=0)
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=5.0)
    return SlowLoop(
        env=env,
        params=SlowLoopParams(),
        contract=contract,
        slow_interval_steps=40,
    )


def _packet(*, gid: int, spatial: bool, effective: bool) -> SlowObservation:
    return SlowObservation(
        step=1,
        fast_to_slow=FastToSlowPacket(
            step=1,
            goal_id=gid,
            goal_ne=(0.0, 0.0),
            completion=CompletionStatus(spatial_complete=spatial, effective=effective),
            link=None,
            backlog_bits=None,
            mode=None,
            safety=None,
        ),
    )


def test_goal_edge_fires_on_spatial_before_effective():
    slow = _slow_loop_for_cfg(
        _minimal_contract_cfg(
            event_triggers={
                "on_goal_spatial_complete": True,
                "on_goal_effective_complete": False,
            }
        )
    )
    assert slow._goal_edge_from_packet(_packet(gid=3, spatial=False, effective=False)) is False
    assert slow._goal_edge_from_packet(_packet(gid=3, spatial=True, effective=False)) is True
    assert slow._goal_edge_from_packet(_packet(gid=3, spatial=True, effective=True)) is False


def test_goal_edge_fires_on_effective_only_when_configured():
    slow = _slow_loop_for_cfg(
        _minimal_contract_cfg(
            event_triggers={
                "on_goal_spatial_complete": False,
                "on_goal_effective_complete": True,
            }
        )
    )
    assert slow._goal_edge_from_packet(_packet(gid=5, spatial=True, effective=False)) is False
    assert slow._goal_edge_from_packet(_packet(gid=5, spatial=True, effective=True)) is True


def test_goal_lock_allows_periodic_switch_after_spatial_complete():
    cfg = _minimal_contract_cfg(
        event_triggers={
            "on_goal_spatial_complete": True,
            "on_goal_effective_complete": True,
        }
    )
    slow = _slow_loop_for_cfg(cfg)
    gid = 0
    slow._goal_id = gid
    slow.env.covered.add(gid)
    obs = _packet(gid=gid, spatial=True, effective=False)
    new_gid, new_seq = slow._apply_goal_lock(
        obs=obs,
        new_gid=1,
        new_seq=[1, 2],
        periodic_only=True,
    )
    assert new_gid == 1
    assert new_seq == [1, 2]


def test_goal_lock_stuck_timeout_releases_on_periodic_replan():
    cfg = _minimal_contract_cfg(
        event_triggers={
            "on_goal_spatial_complete": False,
            "on_goal_effective_complete": False,
        }
    )
    cfg["paper1_loops"]["slow_loop"]["goal_lock_stuck_s"] = 2.0
    slow = _slow_loop_for_cfg(cfg)
    gid = 0
    slow._goal_id = gid
    slow.env.covered.add(gid)
    slow.env.return_queue.enqueue(poi_id=gid, total_bits=1000.0, covered_time_s=0.0)
    stuck_obs = SlowObservation(
        step=10,
        fast_to_slow=FastToSlowPacket(
            step=10,
            goal_id=gid,
            goal_ne=(0.0, 0.0),
            completion=CompletionStatus(spatial_complete=True, effective=False),
            link=None,
            backlog_bits=1000.0,
            mode="Srec",
            safety=None,
        ),
    )
    slow._update_stuck_timer(stuck_obs)
    assert slow._stuck_since_step == 10
    late_obs = SlowObservation(
        step=10 + slow._goal_lock_stuck_threshold_steps(),
        fast_to_slow=stuck_obs.fast_to_slow,
    )
    slow._update_stuck_timer(late_obs)
    new_gid, new_seq = slow._apply_goal_lock(
        obs=late_obs,
        new_gid=2,
        new_seq=[2, 3],
        periodic_only=True,
    )
    assert new_gid == 2
    assert new_seq == [2, 3]
