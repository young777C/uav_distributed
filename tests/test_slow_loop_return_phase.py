"""Slow-loop return phase: GCS goal when task/time/energy conditions are met."""

from __future__ import annotations

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import SlowObservation
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.loops.slow.slow_loop import SlowLoop
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config


def _cfg(*, return_reserve_s: float = 300.0) -> dict:
    return {
        "env": {"return_reserve_s": return_reserve_s, "mission_time_s": 600.0},
        "scene_file": "configs/scenes/g1_uniform.yaml",
        "paper1_loops": {
            "coupling_mode": "full_coupling",
            "slow_policy": "periodic_or_event_replan",
            "semantics": {
                "use_energy_in_slow": True,
                "enable_event_feedback": True,
                "enable_goal_lock": True,
            },
        },
    }


def _slow(cfg: dict) -> SlowLoop:
    scene_path = str(cfg["scene_file"])
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


def test_return_phase_when_all_effective() -> None:
    slow = _slow(_cfg())
    env = slow.env
    for pid in range(len(env.pois)):
        env.covered.add(pid)
        env.returned.add(pid)
    env._sync_effective_from_returned()
    assert slow._should_enter_return_phase()
    plan = slow.step(SlowObservation(step=int(env.t), fast_to_slow=None))
    assert plan.goal_id is None
    assert plan.goal_ne == env.cfg.gcs_ne


def test_return_phase_when_mission_time_low() -> None:
    slow = _slow(_cfg(return_reserve_s=400.0))
    env = slow.env
    hz = int(env.cfg.step_hz)
    env.t = int(env.cfg.episode_steps) - int(400.0 * hz) + 1
    assert slow._should_enter_return_phase()
    plan = slow.step(SlowObservation(step=int(env.t), fast_to_slow=None))
    assert plan.goal_id is None
