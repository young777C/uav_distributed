from __future__ import annotations

import math

from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import FastObservation, SlowPlan
from uavlab.paper1.loops.fast.fast_loop import FastLoop
from uavlab.paper1.loops.fast.guidance import pick_nofly_escape_target
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.paper1.types import FastCommState
from uavlab.scene.loader import load_scene_config


def _env_and_contract():
    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/cases/c2_g2_m0.yaml")
    )
    scene_raw = load_scene_yaml(cfg["scene_file"])
    sim = from_resolved_config(cfg, scene_raw)
    env = build_env(sim, load_scene_config(cfg["scene_file"]))
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=sim.waypoint_delta_max_m)
    return env, contract


def test_nofly_escape_target_moves_outward_not_gcs():
    env, contract = _env_and_contract()
    env.reset(seed=0)
    # Inside nofly disk [320, 1250, r=130] from configs/scenes/base.yaml
    env.pos_ne = (400.0, 1250.0)
    assert env.in_nofly(env.pos_ne)
    gcs = (float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1]))
    mission = (2100.0, 2000.0)
    tgt = pick_nofly_escape_target(env=env, contract=contract, mission_goal_ne=mission)
    assert not env.in_nofly(tgt)
    assert math.hypot(tgt[0] - gcs[0], tgt[1] - gcs[1]) > math.hypot(
        env.pos_ne[0] - gcs[0], env.pos_ne[1] - gcs[1]
    ) * 0.85


def test_fast_loop_s_safe_uses_escape_not_gcs():
    env, contract = _env_and_contract()
    fast = FastLoop(env=env, params=FastLoopParams.from_contract(contract), contract=contract)
    env.reset(seed=0)
    env.pos_ne = (400.0, 1250.0)
    gcs = (float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1]))
    plan = SlowPlan(goal_id=10, goal_ne=(2100.0, 2000.0))
    cmd = fast.step(
        FastObservation(
            step=0,
            pos_ne=env.pos_ne,
            comm_mode=FastCommState.INS,
            backlog_bits=0.0,
            link_loss_p=0.1,
        ),
        plan,
        dt=0.2,
    )
    assert cmd.next_comm_mode == FastCommState.SAFE
    assert math.hypot(cmd.target_ne[0] - gcs[0], cmd.target_ne[1] - gcs[1]) > 50.0


def test_fast_loop_s_back_still_homes_gcs():
    env, contract = _env_and_contract()
    fast = FastLoop(env=env, params=FastLoopParams.from_contract(contract), contract=contract)
    env.reset(seed=0)
    env.pos_ne = (1200.0, 1200.0)
    env.remaining_energy = 0.0
    gcs = (float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1]))
    plan = SlowPlan(goal_id=10, goal_ne=(2100.0, 2000.0))
    cmd = fast.step(
        FastObservation(
            step=0,
            pos_ne=env.pos_ne,
            comm_mode=FastCommState.INS,
            backlog_bits=0.0,
            link_loss_p=0.1,
        ),
        plan,
        dt=0.2,
    )
    assert cmd.next_comm_mode == FastCommState.BACK
    # Macro target is GCS; tracker may apply a short pure-pursuit lookahead.
    assert math.hypot(cmd.target_ne[0] - gcs[0], cmd.target_ne[1] - gcs[1]) < 80.0
