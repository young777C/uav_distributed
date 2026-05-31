"""Pursuit speed uses mission-goal distance for approach slowdown, not short lookahead."""

from __future__ import annotations

import math

from uavlab.common.config import load_config
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import FastObservation, SlowPlan
from uavlab.paper1.loops.fast.fast_loop import FastLoop
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import Paper1Env, build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.paper1.types import FastCommState
from uavlab.scene.loader import load_scene_config


def _make_env() -> Paper1Env:
    cfg = load_config("configs/base.yaml")
    cfg["scene_file"] = "configs/scenes/g2_cluster_m2.yaml"
    scene_raw = load_scene_yaml(cfg["scene_file"])
    sim = from_resolved_config(cfg, scene_raw)
    return build_env(sim, load_scene_config(cfg["scene_file"]))


def test_cruise_speed_when_far_from_mission_goal() -> None:
    env = _make_env()
    env.reset(seed=0)
    cfg = load_config("configs/base.yaml")
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=6.0)
    fast = FastLoop(env=env, params=FastLoopParams(), contract=contract)

    # POI far from GCS start (~800 m+)
    goal = env.pois[99].pos_ne
    plan = SlowPlan(goal_id=99, goal_ne=goal)
    dt = 1.0 / float(env.cfg.step_hz)

    max_spd = 0.0
    for t in range(30):
        cmd = fast.step(
            FastObservation(
                step=t,
                pos_ne=env.pos_ne,
                comm_mode=FastCommState.INS,
                backlog_bits=0.0,
                link_loss_p=0.0,
            ),
            plan,
            dt,
        )
        env.comm_mode = cmd.next_comm_mode
        env.step_fast(
            target_ne=cmd.target_ne,
            vel_ne_cmd=cmd.vel_ne_cmd,
            dt=dt,
            approach_goal_ne=cmd.approach_goal_ne,
        )
        max_spd = max(max_spd, math.hypot(env.vel_ne[0], env.vel_ne[1]))

    dist_goal = math.hypot(env.pos_ne[0] - goal[0], env.pos_ne[1] - goal[1])
    assert dist_goal > float(env.cfg.approach_slowdown_radius_m)
    assert max_spd >= 0.95 * float(env.cfg.v_xy_cruise)


def test_approach_slowdown_near_mission_goal() -> None:
    env = _make_env()
    env.reset(seed=0)
    goal = env.pois[99].pos_ne
    r = float(max(env.cfg.approach_slowdown_radius_m, 2.0 * env.cfg.visit_radius_m))
    # Place UAV just outside approach zone toward goal
    env.pos_ne = (goal[0] - 0.5 * r, goal[1])
    dt = 1.0 / float(env.cfg.step_hz)
    env.step_fast(
        target_ne=(goal[0], goal[1]),
        dt=dt,
        approach_goal_ne=goal,
    )
    spd = math.hypot(env.vel_ne[0], env.vel_ne[1])
    expected = float(env.cfg.v_xy_cruise) * (0.5 * r / r)
    assert abs(spd - expected) < 0.5
