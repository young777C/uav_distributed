from __future__ import annotations

import math

from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.loops.fast.guidance import pick_waypoint
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config


def _contract_and_env():
    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/cases/c2_g2_m0.yaml")
    )
    scene_raw = load_scene_yaml(cfg["scene_file"])
    sim = from_resolved_config(cfg, scene_raw)
    env = build_env(sim, load_scene_config(cfg["scene_file"]))
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=sim.waypoint_delta_max_m)
    return env, contract


def test_terminal_homing_returns_goal_center_when_near():
    env, contract = _contract_and_env()
    env.reset(seed=0)
    goal = (1262.0, 1339.0)
    env.pos_ne = (1258.0, 1333.0)
    wp = pick_waypoint(env=env, goal_ne=goal, contract=contract)
    assert wp == goal


def test_approach_cost_prefers_goal_center_over_ring_near_uav():
    env, contract = _contract_and_env()
    env.reset(seed=0)
    goal = (1262.0, 1339.0)
    env.pos_ne = (1250.0, 1250.0)
    wp = pick_waypoint(env=env, goal_ne=goal, contract=contract)
    d_wp = math.hypot(wp[0] - goal[0], wp[1] - goal[1])
    assert d_wp < 1.0


def test_fdlc_episode_reaches_near_goal_center():
    from uavlab.common.config import _deep_merge
    from uavlab.paper1.contracts.contract_types import FastObservation, SlowObservation, SlowPlan
    from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
    from uavlab.paper1.loops.fast.fast_loop import FastLoop
    from uavlab.paper1.loops.fast.policy import FastLoopParams
    from uavlab.paper1.loops.slow.planner import SlowLoopParams
    from uavlab.paper1.loops.slow.slow_loop import SlowLoop

    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/cases/c2_g2_m0.yaml")
    )
    cfg = _deep_merge(
        cfg,
        load_resolved_config(
            "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml"
        ),
    )
    scene_raw = load_scene_yaml(cfg["scene_file"])
    sim = from_resolved_config(cfg, scene_raw)
    env = build_env(sim, load_scene_config(cfg["scene_file"]))
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=sim.waypoint_delta_max_m)
    slow = SlowLoop(
        env=env,
        params=SlowLoopParams(t_obs_s=sim.poi_dwell_s),
        contract=contract,
        slow_interval_steps=40,
    )
    fast = FastLoop(env=env, params=FastLoopParams.from_contract(contract), contract=contract)
    coupling = CouplingPolicy(mode=str(contract.coupling_mode))
    dt = 1.0 / sim.step_hz
    env.reset(seed=0)
    plan = SlowPlan(goal_id=None, goal_ne=env.cfg.gcs_ne)
    pkt0 = coupling.build_fast_to_slow_packet(
        contract=contract,
        env=env,
        step=0,
        link_loss_p=0.1,
        link_delay_s=0,
        link_bandwidth_bps=1e6,
        plan=plan,
    )
    plan = slow.step(SlowObservation(step=0, fast_to_slow=pkt0))
    min_d = 1e9
    for t in range(9000):
        cmd = fast.step(
            FastObservation(
                step=t,
                pos_ne=env.pos_ne,
                comm_mode=env.comm_mode,
                backlog_bits=float(env.backlog_bits),
                link_loss_p=float(env.observe_link_state().loss_p),
            ),
            plan,
            dt,
        )
        env.comm_mode = cmd.next_comm_mode
        env.step_fast(target_ne=cmd.target_ne, vel_ne_cmd=cmd.vel_ne_cmd, dt=dt)
        if plan.goal_id is not None:
            g = env.pois[plan.goal_id].pos_ne
            min_d = min(min_d, math.hypot(env.pos_ne[0] - g[0], env.pos_ne[1] - g[1]))
        pkt = coupling.build_fast_to_slow_packet(
            contract=contract,
            env=env,
            step=t,
            link_loss_p=float(env.observe_link_state().loss_p),
            link_delay_s=float(env.observe_link_state().delay_s),
            link_bandwidth_bps=float(env.observe_link_state().bandwidth_bps),
            plan=plan,
        )
        plan = slow.step(SlowObservation(step=t, fast_to_slow=pkt))
    assert min_d <= float(env.cfg.visit_radius_m) + 1.0
    assert len(env.covered) >= 1


def test_cdsl_effective_after_coverage_with_frozen_fsm():
    from uavlab.common.config import _deep_merge
    from uavlab.paper1.contracts.contract_types import FastObservation, SlowObservation, SlowPlan
    from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
    from uavlab.paper1.loops.fast.fast_loop import FastLoop
    from uavlab.paper1.loops.fast.policy import FastLoopParams
    from uavlab.paper1.loops.slow.planner import SlowLoopParams
    from uavlab.paper1.loops.slow.slow_loop import SlowLoop
    from uavlab.paper1.comm.dual_link import dual_link_thresholds_from_comm
    from uavlab.paper1.runner.return_scheduling import goal_spatial_complete, should_attempt_key_return
    from uavlab.paper1.types import FastCommState

    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/cases/c2_g2_m0.yaml")
    )
    cfg = _deep_merge(
        cfg,
        load_resolved_config("configs/experiments/paper1/system/struct_centralized_single_loop.yaml"),
    )
    cfg = _deep_merge(cfg, {"comm": {"data_chunk_bits": 8000.0}})
    scene_raw = load_scene_yaml(cfg["scene_file"])
    sim = from_resolved_config(cfg, scene_raw)
    env = build_env(sim, load_scene_config(cfg["scene_file"]))
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=sim.waypoint_delta_max_m)
    assert contract.enable_fast_mode_switch is False
    assert should_attempt_key_return(
        backlog_bits=1.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=False,
        spatial_complete=False,
    )

    slow = SlowLoop(
        env=env,
        params=SlowLoopParams(t_obs_s=sim.poi_dwell_s),
        contract=contract,
        slow_interval_steps=40,
    )
    fast = FastLoop(env=env, params=FastLoopParams.from_contract(contract), contract=contract)
    coupling = CouplingPolicy(mode=str(contract.coupling_mode))
    th = dual_link_thresholds_from_comm(dict(cfg.get("comm") or {}))
    ret_params = th.to_return_params()
    dt = 1.0 / sim.step_hz
    env.reset(seed=0)
    plan = SlowPlan(goal_id=None, goal_ne=env.cfg.gcs_ne)
    pkt0 = coupling.build_fast_to_slow_packet(
        contract=contract,
        env=env,
        step=0,
        link_loss_p=0.1,
        link_delay_s=0,
        link_bandwidth_bps=1e6,
        plan=plan,
    )
    plan = slow.step(SlowObservation(step=0, fast_to_slow=pkt0))
    for t in range(9000):
        link = env.observe_link_state()
        cmd = fast.step(
            FastObservation(
                step=t,
                pos_ne=env.pos_ne,
                comm_mode=env.comm_mode,
                backlog_bits=float(env.backlog_bits),
                link_loss_p=float(link.loss_p),
            ),
            plan,
            dt,
        )
        env.comm_mode = cmd.next_comm_mode
        env.step_fast(target_ne=cmd.target_ne, vel_ne_cmd=cmd.vel_ne_cmd, dt=dt)
        if should_attempt_key_return(
            backlog_bits=float(env.backlog_bits),
            comm_mode=env.comm_mode,
            enable_fast_mode_switch=bool(contract.enable_fast_mode_switch),
            spatial_complete=goal_spatial_complete(env=env, goal_id=plan.goal_id),
        ):
            env.progress_key_return(params=ret_params)
        pkt = coupling.build_fast_to_slow_packet(
            contract=contract,
            env=env,
            step=t,
            link_loss_p=float(link.loss_p),
            link_delay_s=float(link.delay_s),
            link_bandwidth_bps=float(link.bandwidth_bps),
            plan=plan,
        )
        plan = slow.step(SlowObservation(step=t, fast_to_slow=pkt))
    assert 76 in env.covered
    assert len(env.effective) >= 1
    assert 76 in env.effective


def test_uav_can_leave_covered_poi_disk_when_visit_radius_large():
    """After covering one POI, must not hover-lock inside its disk (multi-POI missions)."""
    import math
    from uavlab.common.config import _deep_merge
    from uavlab.paper1.contracts.contract_types import FastObservation, SlowObservation, SlowPlan
    from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
    from uavlab.paper1.loops.fast.fast_loop import FastLoop
    from uavlab.paper1.loops.fast.policy import FastLoopParams
    from uavlab.paper1.loops.slow.planner import SlowLoopParams
    from uavlab.paper1.loops.slow.slow_loop import SlowLoop
    from uavlab.paper1.comm.dual_link import dual_link_thresholds_from_comm
    from uavlab.paper1.runner.return_scheduling import goal_spatial_complete, should_attempt_key_return

    cfg = apply_experiment_presets(
        load_resolved_config("configs/experiments/paper1/cases/c2_g2_m0.yaml")
    )
    cfg = _deep_merge(
        cfg,
        load_resolved_config(
            "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml"
        ),
    )
    cfg = _deep_merge(cfg, {"comm": {"data_chunk_bits": 8000.0}})
    scene_raw = load_scene_yaml(cfg["scene_file"])
    sim = from_resolved_config(cfg, scene_raw)
    env = build_env(sim, load_scene_config(cfg["scene_file"]))
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=sim.waypoint_delta_max_m)
    slow = SlowLoop(
        env=env,
        params=SlowLoopParams(t_obs_s=sim.poi_dwell_s),
        contract=contract,
        slow_interval_steps=40,
    )
    fast = FastLoop(env=env, params=FastLoopParams.from_contract(contract), contract=contract)
    coupling = CouplingPolicy(mode=str(contract.coupling_mode))
    th = dual_link_thresholds_from_comm(dict(cfg.get("comm") or {}))
    ret_params = th.to_return_params()
    dt = 1.0 / sim.step_hz
    env.reset(seed=0)
    plan = SlowPlan(goal_id=None, goal_ne=env.cfg.gcs_ne)
    pkt0 = coupling.build_fast_to_slow_packet(
        contract=contract, env=env, step=0, link_loss_p=0.1, link_delay_s=0, link_bandwidth_bps=1e6, plan=plan
    )
    plan = slow.step(SlowObservation(step=0, fast_to_slow=pkt0))
    min_d106 = 1e9
    for t in range(9000):
        cmd = fast.step(
            FastObservation(
                step=t,
                pos_ne=env.pos_ne,
                comm_mode=env.comm_mode,
                backlog_bits=float(env.backlog_bits),
                link_loss_p=float(env.observe_link_state().loss_p),
            ),
            plan,
            dt,
        )
        env.comm_mode = cmd.next_comm_mode
        env.step_fast(target_ne=cmd.target_ne, vel_ne_cmd=cmd.vel_ne_cmd, dt=dt)
        if should_attempt_key_return(
            backlog_bits=float(env.backlog_bits),
            comm_mode=env.comm_mode,
            enable_fast_mode_switch=bool(contract.enable_fast_mode_switch),
            spatial_complete=goal_spatial_complete(env=env, goal_id=plan.goal_id),
        ):
            env.progress_key_return(params=ret_params)
        pkt = coupling.build_fast_to_slow_packet(
            contract=contract,
            env=env,
            step=t,
            link_loss_p=float(env.observe_link_state().loss_p),
            link_delay_s=0,
            link_bandwidth_bps=1e6,
            plan=plan,
        )
        plan = slow.step(SlowObservation(step=t, fast_to_slow=pkt))
        if plan.goal_id == 106:
            g = env.pois[106].pos_ne
            min_d106 = min(min_d106, math.hypot(env.pos_ne[0] - g[0], env.pos_ne[1] - g[1]))
    assert 76 in env.covered
    assert min_d106 < 30.0
