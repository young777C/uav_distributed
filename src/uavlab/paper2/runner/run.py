"""
Paper 2 experiment runner.

Usage:
    PYTHONPATH=src python -m uavlab.paper2.runner.run --config <yaml> --experiment <baseline_key>
    PYTHONPATH=src python -m uavlab.paper2.runner.run --config <yaml> --experiment d_epa_rhp
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import SlowObservation, SlowPlan, FastObservation, FastCommand
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.runner.run_context import paper1_run_context_label, print_paper1_run_header
from uavlab.paper1.sim.env import Paper1Env

from uavlab.paper2.runner.experiment_registry import ExperimentRecipe, get_recipe


def _slow_loop_params_from_cfg(cfg: Any) -> SlowLoopParams:
    """Extract SlowLoopParams from config."""
    p1 = cfg.get("paper1_loops") if isinstance(cfg.get("paper1_loops"), dict) else {}
    sl = p1.get("slow_loop") if isinstance(p1.get("slow_loop"), dict) else {}
    mp = cfg.get("model_params") if isinstance(cfg.get("model_params"), dict) else {}
    return SlowLoopParams(
        lambda_q=float(sl.get("comm_objective_weight", mp.get("lambda_q", 1.0))),
        mu_dist=float(sl.get("distance_weight", mp.get("mu_dist", 0.01))),
        beta_ret=float(sl.get("return_suitability_weight", mp.get("beta_ret", 0.5))),
        t_obs_s=float(mp.get("t_obs_s", 10.0)),
        t_safe_s=float(mp.get("t_safe_s", 0.0)),
        energy_plan_margin_frac=float(
            p1.get("struct_profile", {}).get("energy_plan_margin_frac", 0.0)
        ),
    )


def _fast_loop_params_from_cfg(cfg: Any) -> FastLoopParams:
    """Extract FastLoopParams from config."""
    fl = {}
    if isinstance(cfg.get("paper1_loops"), dict):
        fl_raw = cfg["paper1_loops"].get("fast_loop")
        if isinstance(fl_raw, dict):
            fl = fl_raw
    fsm = {}
    if isinstance(fl.get("fsm_thresholds"), dict):
        fsm = fl["fsm_thresholds"]
    return FastLoopParams(
        link_loss_safe=float(fsm.get("link_loss_safe", 0.05)),
        link_loss_recover=float(fsm.get("link_loss_recover", 0.20)),
        safe_hover_steps=int(fsm.get("safe_hover_steps", 10)),
        enable_fsm=bool(fl.get("enable_fsm", True)),
        enable_back_mode=bool(fl.get("enable_back_mode", True)),
        enable_safety_mode=bool(fl.get("enable_safety_mode", True)),
        enable_recovery_mode=bool(fl.get("enable_recovery_mode", True)),
    )


def _build_env_from_config(cfg: Any) -> tuple[Paper1Env, Any]:
    """Build Paper1Env + Paper1SimConfig from resolved config using paper1's env builder."""
    from uavlab.paper1.sim.config import from_resolved_config
    from uavlab.paper1.sim.env import build_env
    from uavlab.paper1.sim.scene_loader import load_scene_yaml
    from uavlab.scene.loader import load_scene_config

    scene_path = Path(str(cfg.get("scene_file", "")))
    if scene_path.exists():
        scene_yaml = load_scene_yaml(scene_path)
        scene_geom = load_scene_config(scene_path)
    else:
        scene_yaml = None
        scene_geom = None

    sim_cfg = from_resolved_config(cfg, scene=scene_yaml)
    env = build_env(sim_cfg, scene_geom)
    return env, sim_cfg


def run_single_episode(
    *,
    env: Paper1Env,
    contract: Paper1ContractConfig,
    recipe: ExperimentRecipe,
    slow_interval_steps: int,
    max_steps: int,
    params: Any,
    dt: float = 0.2,
) -> Dict[str, Any]:
    """Run one episode with the given experiment recipe.

    Uses paper1's ``env.step_fast()``, ``env.observe_link_state()``,
    ``env.progress_key_return()``, and ``env.done()`` for the simulation loop.
    """
    from uavlab.paper1.loops.fast.policy import FastLoopParams as P1FastLoopParams
    from uavlab.paper1.loops.slow.planner import SlowLoopParams as P1SlowLoopParams
    from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
    from uavlab.paper1.metrics.link_recovery import LinkRecoveryRecorder
    from uavlab.paper1.runner.return_scheduling import should_attempt_key_return

    # Initialize slow loop
    slow_loop = recipe.slow_loop_cls(
        env=env,
        params=_slow_loop_params_from_cfg(params),
        contract=contract,
        slow_interval_steps=int(slow_interval_steps),
    )

    # Initialize fast loop
    fast_loop = recipe.fast_loop_cls(
        env=env,
        params=_fast_loop_params_from_cfg(params),
        contract=contract,
    )

    # Paper1 helpers
    coupling = CouplingPolicy(mode=str(contract.coupling_mode))
    link_rec = LinkRecoveryRecorder()
    dual_link = contract.dual_link
    ret_params = dual_link.to_return_params()

    # Reset all state
    env.reset()
    slow_loop.reset()
    fast_loop.reset()
    link_rec.reset()

    # ── Bootstrap slow loop at t=0 (matches Paper1 runner pattern) ──
    link0 = env.observe_link_state()
    plan = SlowPlan(goal_id=None, goal_ne=(float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1])))
    pkt0 = coupling.build_fast_to_slow_packet(
        contract=contract,
        env=env,
        step=0,
        link_loss_p=float(link0.loss_p),
        link_delay_s=float(link0.delay_s),
        link_bandwidth_bps=float(link0.bandwidth_bps),
        control_link_lost=fast_loop.control_link_lost,
        control_link_lost_duration_s=fast_loop.control_link_lost_duration_s,
        plan=plan,
    )
    plan = slow_loop.step(SlowObservation(step=0, fast_to_slow=pkt0))

    # ── Adaptive schedule windowed statistics (Bug #3 fix) ──
    _stat_window = 200
    _stat_energy: List[float] = []
    _stat_loss: List[float] = []
    _stat_discrepancy: List[float] = []
    _stat_event_flags: List[int] = []
    _stat_feedback_flags: List[int] = []
    _total_energy_capacity: float = float(getattr(env.cfg, 'battery_wh', 263.2))

    # Episode loop
    step = 0
    while not env.done() and step < int(max_steps):
        link = env.observe_link_state()

        # Fast observation
        obs = FastObservation(
            step=int(env.t),
            pos_ne=env.pos_ne,
            comm_mode=env.comm_mode,
            backlog_bits=float(env.backlog_bits),
            link_loss_p=float(link.loss_p),
        )

        # Fast loop step (paper2 returns (cmd, prob) tuple, paper1 returns cmd)
        fast_result = fast_loop.step(obs=obs, plan=plan, dt=float(dt))
        if isinstance(fast_result, tuple):
            cmd, local_prob = fast_result
        else:
            cmd = fast_result
            local_prob = 0.0

        # ── VoI feedback: gate on policy decision (Bug #5 fix) ──
        feedback_sent_this_step = False
        dpe_this_step = 0.0
        if hasattr(slow_loop, 'evaluate_feedback') and slow_loop.__class__.__name__ == 'Paper2SlowLoop':
            critical_events = []
            if _goal_spatial_complete(env, plan.goal_id):
                from uavlab.paper2.contracts.feedback_policy import CriticalEventType
                critical_events.append(CriticalEventType.COVERAGE_COMPLETE)
            if plan.goal_id is not None and int(plan.goal_id) in env.effective:
                from uavlab.paper2.contracts.feedback_policy import CriticalEventType
                if CriticalEventType.DATA_RETURN_COMPLETE not in critical_events:
                    critical_events.append(CriticalEventType.DATA_RETURN_COMPLETE)
            if float(link.loss_p) > 0.9:
                from uavlab.paper2.contracts.feedback_policy import CriticalEventType
                critical_events.append(CriticalEventType.CONTROL_LINK_DROP)
            if float(env.remaining_energy) < float(env.cfg.energy_safe_margin) * 2:
                from uavlab.paper2.contracts.feedback_policy import CriticalEventType
                critical_events.append(CriticalEventType.LOW_ENERGY)
            decision = slow_loop.evaluate_feedback(
                step=int(env.t),
                uav_probability=float(local_prob),
                critical_events=critical_events,
            )
            dpe_this_step = float(decision.discrepancy)
            # Only count as "feedback sent" when the policy actually approves
            if decision.send_feedback:
                feedback_sent_this_step = True

        # Apply command to env
        env.comm_mode = cmd.next_comm_mode
        env.step_fast(
            target_ne=cmd.target_ne,
            vel_ne_cmd=cmd.vel_ne_cmd,
            dt=float(dt),
            approach_goal_ne=cmd.approach_goal_ne,
        )

        # Track link recovery
        link_rec.observe(step=int(env.t), loss_p=float(link.loss_p), comm_mode=env.comm_mode)

        # Attempt key data return
        if should_attempt_key_return(
            backlog_bits=float(env.backlog_bits),
            comm_mode=env.comm_mode,
            enable_fast_mode_switch=bool(contract.enable_fast_mode_switch),
            spatial_complete=_goal_spatial_complete(env, plan.goal_id),
            return_phase=plan.goal_id is None,
            fast_upload_mode=str(env.cfg.fast_upload_mode),
            fixed_send_ratio=float(env.cfg.fixed_send_ratio),
            step=int(env.t),
        ):
            env.progress_key_return(dt_s=float(dt), params=ret_params)

        # ── Adaptive schedule statistics (Bug #3 fix) ──
        energy_ratio = float(env.remaining_energy) / max(_total_energy_capacity, 1e-12)
        _stat_energy.append(energy_ratio)
        _stat_loss.append(float(link.loss_p))
        _stat_discrepancy.append(dpe_this_step)
        _stat_feedback_flags.append(1 if feedback_sent_this_step else 0)
        has_event = 1 if (float(link.loss_p) > 0.9 or
                          float(env.remaining_energy) < float(env.cfg.energy_safe_margin) * 2) else 0
        _stat_event_flags.append(has_event)

        # Trim windows
        while len(_stat_energy) > _stat_window:
            _stat_energy.pop(0)
            _stat_loss.pop(0)
            _stat_discrepancy.pop(0)
            _stat_feedback_flags.pop(0)
            _stat_event_flags.pop(0)

        # Periodic adaptive schedule update (every 100 steps)
        if hasattr(slow_loop, 'adaptive_schedule') and step > 0 and step % 100 == 0 and len(_stat_energy) >= 10:
            window_steps = len(_stat_energy)
            slow_loop.adaptive_schedule.update_statistics(
                energy_ratio=sum(_stat_energy) / max(1, window_steps),
                avg_loss_p=sum(_stat_loss) / max(1, window_steps),
                event_count=sum(_stat_event_flags),
                total_steps=window_steps,
                feedback_steps=sum(_stat_feedback_flags),
                avg_discrepancy=sum(_stat_discrepancy) / max(1, window_steps),
            )
            slow_loop.adaptive_schedule.step()
            # Apply new schedule state to slow loop
            slow_loop._current_schedule = slow_loop.adaptive_schedule.state
            # Also update feedback threshold
            slow_loop.feedback_policy.set_threshold(float(slow_loop.adaptive_schedule.state.feedback_threshold))

        # ── V3: Communication cost for coupling packet ──
        # D-EPA-RHP: cost only when VoI gate opens (feedback_sent_this_step).
        # Baselines (no VoI): cost applied every slow-loop step (always-send overhead).
        is_depa = hasattr(slow_loop, 'evaluate_feedback') and slow_loop.__class__.__name__ == 'Paper2SlowLoop'
        comm_cost_j = float(params.get('paper2', {}).get('comm_cost_per_feedback_j', 0.0)
                            if isinstance(params, dict) else 0.0)
        apply_comm_cost = (is_depa and feedback_sent_this_step) or (not is_depa)
        if comm_cost_j > 0 and apply_comm_cost:
            e0_j = float(getattr(env.cfg, 'battery_wh', 263.2) if hasattr(env.cfg, 'battery_wh') else 263.2) * 3600.0
            if hasattr(env, 'remaining_energy'):
                env.remaining_energy = max(0.0, float(env.remaining_energy) - comm_cost_j / max(e0_j, 1.0))

        # Build fast→slow packet for coupling feedback
        pkt = coupling.build_fast_to_slow_packet(
            contract=contract,
            env=env,
            step=int(env.t),
            link_loss_p=float(link.loss_p),
            link_delay_s=float(link.delay_s),
            link_bandwidth_bps=float(link.bandwidth_bps),
            control_link_lost=fast_loop.control_link_lost,
            control_link_lost_duration_s=fast_loop.control_link_lost_duration_s,
            plan=plan,
        )

        # ── Single slow loop step (Bug #1 fix: only ONE call, with coupling packet) ──
        plan = slow_loop.step(SlowObservation(step=int(env.t), fast_to_slow=pkt))

        step += 1

    # Collect metrics (include feedback stats if available)
    metrics = _collect_metrics(env=env, step=step)
    if hasattr(slow_loop, 'feedback_policy'):
        fp = slow_loop.feedback_policy
        metrics['n_feedback'] = int(fp.feedback_count)
        metrics['n_replan'] = int(getattr(slow_loop, 'replan_count', 0))
    return metrics


def _goal_spatial_complete(env: Paper1Env, goal_id: Optional[int]) -> bool:
    """Check if the current goal POI has been spatially covered."""
    if goal_id is None:
        return False
    return int(goal_id) in env.covered


def _collect_metrics(env: Paper1Env, step: int) -> Dict[str, Any]:
    """Collect episode metrics matching Paper 2 §6.5 definitions."""
    n_pois = max(1, len(env.pois))
    covered = set(getattr(env, 'covered', []))
    effective = set(getattr(env, 'effective', []))
    returned = set(getattr(env, 'returned', []))

    r_cov = len(covered) / n_pois
    r_eff = len(effective) / n_pois

    # Return latency
    ret_latencies = []
    for pid in effective:
        if hasattr(env, '_return_times') and pid in env._return_times:
            ret_latencies.append(env._return_times[pid])
    t_ret = sum(ret_latencies) / max(1, len(ret_latencies))

    # Energy
    remaining_energy = float(getattr(env, 'remaining_energy', 0.0))
    cfg_obj = getattr(env, 'cfg', None)
    total_energy = float(getattr(cfg_obj, 'battery_wh', 263.2) if cfg_obj is not None else 263.2)

    return {
        "R_cov": float(r_cov),
        "R_task": float(r_eff),
        "R_fail_given_cov": float(max(0.0, 1.0 - r_eff / max(r_cov, 1e-12))),
        "R_home": 1.0 if getattr(env, 'returned_home', False) else 0.0,
        "R_oob": 1.0 if getattr(env, 'oob', False) else 0.0,
        "T_ret_s": float(t_ret),
        "T_nf_s": float(getattr(env, 'nofly_time_s', 0.0)),
        "T_deg_s": float(getattr(env, 'degrade_time_s', 0.0)),
        "energy_remaining_ratio": float(remaining_energy / max(total_energy, 1e-12)),
        "n_replan": 0,  # TODO: track properly
        "episode_steps": int(step + 1),
    }


def run_experiment(
    *,
    config: Dict[str, Any],
    experiment_key: str,
    n_episodes: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Run a paper2 experiment with multiple episodes.

    Returns:
        (aggregate_metrics, episode_metrics_list)
    """
    import random
    import numpy as np
    random.seed(int(seed))
    np.random.seed(int(seed))

    recipe = get_recipe(experiment_key)
    if recipe is None:
        raise ValueError(f"Unknown experiment key: {experiment_key!r}. Available: {list_experiments()}")

    from uavlab.experiments.presets import apply_experiment_presets
    cfg = apply_experiment_presets(dict(config))

    # Create env (also returns sim_cfg for contract creation)
    env, sim_cfg = _build_env_from_config(cfg)

    # Create contract with resolved waypoint_delta_max_m
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=float(sim_cfg.waypoint_delta_max_m))

    slow_interval = int(cfg.get("paper1_loops", {}).get("slow_loop", {}).get("slow_interval_steps", 40))
    max_steps = int(sim_cfg.episode_steps)  # Use the env's actual episode length
    dt = float(1.0 / max(1, int(sim_cfg.step_hz)))

    ctx = print_paper1_run_header(cfg, contract=contract, experiment_id=experiment_key,
                                   slow_interval_steps=slow_interval)

    all_ep_metrics: List[Dict[str, Any]] = []
    for ep in range(int(n_episodes)):
        env.reset()
        metrics = run_single_episode(
            env=env,
            contract=contract,
            recipe=recipe,
            slow_interval_steps=slow_interval,
            max_steps=max_steps,
            params=cfg,
            dt=dt,
        )
        all_ep_metrics.append(metrics)

        if verbose:
            print(
                f"  ep={ep}  R_task={metrics['R_task']:.3f}  "
                f"R_cov={metrics['R_cov']:.3f}  "
                f"R_fail|cov={metrics['R_fail_given_cov']:.3f}  "
                f"T_ret={metrics['T_ret_s']:.1f}s",
                flush=True,
            )

    # Aggregate
    agg = {}
    for key in all_ep_metrics[0]:
        vals = [m[key] for m in all_ep_metrics]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    if verbose:
        print(f"\n[Aggregate] {ctx}  R_task={agg['R_task_mean']:.3f}±{agg['R_task_std']:.3f}  "
              f"R_cov={agg['R_cov_mean']:.3f}±{agg['R_cov_std']:.3f}",
              flush=True)

    return agg, all_ep_metrics


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Paper 2 experiment runner")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment key (baseline name)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="", help="Output JSON path (optional)")
    args = parser.parse_args()

    from uavlab.common.config import load_resolved_config
    config = load_resolved_config(args.config)

    agg, _ep_metrics = run_experiment(
        config=config,
        experiment_key=str(args.experiment),
        n_episodes=int(args.episodes),
        seed=int(args.seed),
    )

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump({"aggregate": agg, "config": str(args.config), "experiment": args.experiment}, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
