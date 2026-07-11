"""
Paper: paper1_v2
Purpose: V5 diagnostic — track action distribution, boundary events, mode switches on C3-High.
Inputs: configs/experiments/paper2/c3_high_m2_d_epa_rhp.yaml
Outputs: results_v2/v5_validation/v5_diagnostic.json
"""
from __future__ import annotations

import json, sys, math, time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uavlab.common.config import load_resolved_config
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import SlowObservation, SlowPlan, FastObservation
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
from uavlab.paper1.metrics.link_recovery import LinkRecoveryRecorder
from uavlab.paper1.runner.return_scheduling import should_attempt_key_return
from uavlab.paper1.sim.env import Paper1Env
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.paper1.sim.env import build_env
from uavlab.scene.loader import load_scene_config

from uavlab.paper2.runner.experiment_registry import get_recipe
from uavlab.paper2.loops.fast_loop_voi import UavAction


def _slow_loop_params_from_cfg(cfg):
    p1 = cfg.get("paper1_loops") if isinstance(cfg.get("paper1_loops"), dict) else {}
    sl = p1.get("slow_loop") if isinstance(p1.get("slow_loop"), dict) else {}
    mp = cfg.get("model_params") if isinstance(cfg.get("model_params"), dict) else {}
    return SlowLoopParams(
        lambda_q=float(sl.get("comm_objective_weight", mp.get("lambda_q", 1.0))),
        mu_dist=float(sl.get("distance_weight", mp.get("mu_dist", 0.01))),
        beta_ret=float(sl.get("return_suitability_weight", mp.get("beta_ret", 0.5))),
        t_obs_s=float(mp.get("t_obs_s", 10.0)),
        t_safe_s=float(mp.get("t_safe_s", 0.0)),
        energy_plan_margin_frac=float(p1.get("struct_profile", {}).get("energy_plan_margin_frac", 0.0)),
    )

def _fast_loop_params_from_cfg(cfg):
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


def run_diagnostic(experiment_key: str, label: str):
    config_path = "configs/experiments/paper2/c3_high_m2_d_epa_rhp.yaml"
    cfg = load_resolved_config(config_path)

    recipe = get_recipe(experiment_key)
    if recipe is None:
        raise ValueError(f"Unknown experiment: {experiment_key}")

    # Build env
    scene_path = Path(str(cfg.get("scene_file", "")))
    scene_yaml = load_scene_yaml(scene_path) if scene_path.exists() else None
    scene_geom = load_scene_config(scene_path) if scene_path.exists() else None
    sim_cfg = from_resolved_config(cfg, scene=scene_yaml)
    env = build_env(sim_cfg, scene_geom)

    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=float(sim_cfg.waypoint_delta_max_m))

    slow_interval = int(cfg.get("paper1_loops", {}).get("slow_loop", {}).get("slow_interval_steps", 40))
    max_steps = int(sim_cfg.episode_steps)
    dt = float(1.0 / max(1, int(sim_cfg.step_hz)))

    slow_loop = recipe.slow_loop_cls(
        env=env, params=_slow_loop_params_from_cfg(cfg),
        contract=contract, slow_interval_steps=slow_interval,
    )
    fast_loop = recipe.fast_loop_cls(
        env=env, params=_fast_loop_params_from_cfg(cfg), contract=contract,
    )

    coupling = CouplingPolicy(mode=str(contract.coupling_mode))
    link_rec = LinkRecoveryRecorder()
    dual_link = contract.dual_link
    ret_params = dual_link.to_return_params()

    env.reset()
    slow_loop.reset()
    fast_loop.reset()
    link_rec.reset()

    # Bootstrap
    link0 = env.observe_link_state()
    plan = SlowPlan(goal_id=None, goal_ne=(float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1])))
    pkt0 = coupling.build_fast_to_slow_packet(
        contract=contract, env=env, step=0,
        link_loss_p=float(link0.loss_p), link_delay_s=float(link0.delay_s),
        link_bandwidth_bps=float(link0.bandwidth_bps),
        control_link_lost=fast_loop.control_link_lost,
        control_link_lost_duration_s=fast_loop.control_link_lost_duration_s,
        plan=plan,
    )
    plan = slow_loop.step(SlowObservation(step=0, fast_to_slow=pkt0))

    # Diagnostics
    action_counts = Counter()
    boundary_near_steps = 0
    nofly_steps = 0
    fsm_fallback_steps = 0
    replan_count = 0
    last_goal_id = None
    goal_switches = 0
    stuck_steps = 0  # steps where goal covered but not returned
    inspect_steps = 0
    safe_steps = 0
    recover_steps = 0
    transmit_steps = 0

    step = 0
    while not env.done() and step < max_steps:
        link = env.observe_link_state()

        obs = FastObservation(
            step=int(env.t), pos_ne=env.pos_ne, comm_mode=env.comm_mode,
            backlog_bits=float(env.backlog_bits), link_loss_p=float(link.loss_p),
        )

        fast_result = fast_loop.step(obs=obs, plan=plan, dt=float(dt))
        if isinstance(fast_result, tuple):
            cmd, local_prob = fast_result
        else:
            cmd = fast_result
            local_prob = 0.0

        # Track actions — map both FastCommState and UavAction values
        mode = cmd.next_comm_mode
        mode_str = mode.value if hasattr(mode, 'value') else str(mode)
        action_counts[mode_str] += 1

        # Track boundary/nofly status
        px, py = float(env.pos_ne[0]), float(env.pos_ne[1])
        n_min, n_max = float(env.cfg.n_min), float(env.cfg.n_max)
        margin_100 = 100.0
        if (px < n_min + margin_100 or px > n_max - margin_100 or
            py < n_min + margin_100 or py > n_max - margin_100):
            boundary_near_steps += 1
        if env.in_nofly((px, py)):
            nofly_steps += 1

        # Track specific actions
        if mode.value == "ins":
            inspect_steps += 1
        elif mode.value == "safe":
            safe_steps += 1
        elif mode.value == "rec":
            recover_steps += 1
        elif mode.value == "tx":
            transmit_steps += 1

        # Track goal switches
        if plan.goal_id != last_goal_id:
            goal_switches += 1
            last_goal_id = plan.goal_id

        # Track stuck at covered POI
        if (plan.goal_id is not None and plan.goal_id in env.covered and
            plan.goal_id not in env.effective):
            stuck_steps += 1

        # Apply command
        env.comm_mode = cmd.next_comm_mode
        env.step_fast(target_ne=cmd.target_ne, vel_ne_cmd=cmd.vel_ne_cmd,
                      dt=float(dt), approach_goal_ne=cmd.approach_goal_ne)

        link_rec.observe(step=int(env.t), loss_p=float(link.loss_p), comm_mode=env.comm_mode)

        if should_attempt_key_return(
            backlog_bits=float(env.backlog_bits), comm_mode=env.comm_mode,
            enable_fast_mode_switch=bool(contract.enable_fast_mode_switch),
            spatial_complete=plan.goal_id is not None and plan.goal_id in env.covered,
            return_phase=plan.goal_id is None,
            fast_upload_mode=str(env.cfg.fast_upload_mode),
            fixed_send_ratio=float(env.cfg.fixed_send_ratio), step=int(env.t),
        ):
            env.progress_key_return(dt_s=float(dt), params=ret_params)

        # Slow loop step
        pkt = coupling.build_fast_to_slow_packet(
            contract=contract, env=env, step=int(env.t),
            link_loss_p=float(link.loss_p), link_delay_s=float(link.delay_s),
            link_bandwidth_bps=float(link.bandwidth_bps),
            control_link_lost=fast_loop.control_link_lost,
            control_link_lost_duration_s=fast_loop.control_link_lost_duration_s,
            plan=plan,
        )
        plan = slow_loop.step(SlowObservation(step=int(env.t), fast_to_slow=pkt))
        step += 1

    # Collect results
    n_pois = max(1, len(env.pois))
    r_cov = len(env.covered) / n_pois
    r_eff = len(env.effective) / n_pois

    covered_ids = set(env.covered)
    effective_ids = set(env.effective)
    total_steps = step
    total_actions = sum(action_counts.values())

    return {
        "experiment": experiment_key,
        "label": label,
        "R_task": r_eff,
        "R_cov": r_cov,
        "R_fail_given_cov": max(0.0, 1.0 - r_eff / max(r_cov, 1e-12)),
        "total_steps": total_steps,
        "action_distribution": dict(action_counts),
        "action_pct": {k: v / max(1, total_actions) for k, v in action_counts.items()},
        "boundary_near_steps": boundary_near_steps,
        "boundary_near_pct": boundary_near_steps / max(1, total_steps),
        "nofly_steps": nofly_steps,
        "goal_switches": goal_switches,
        "stuck_steps": stuck_steps,
        "n_covered": len(covered_ids),
        "n_effective": len(effective_ids),
        "n_pois": n_pois,
        "inspect_steps": inspect_steps,
        "safe_steps": safe_steps,
        "recover_steps": recover_steps,
        "transmit_steps": transmit_steps,
    }


def main():
    print("=" * 72)
    print("  V5 C3-High Diagnostic: Action Distribution & Behavior")
    print("=" * 72)

    experiments = [
        ("epa_rhp_centralized", "EPA-Centralized"),
        ("d_epa_rhp",           "D-EPA-RHP V5"),
    ]

    all_results = []
    for exp_key, label in experiments:
        print(f"\n--- Running {label} ---")
        t0 = time.time()
        result = run_diagnostic(exp_key, label)
        elapsed = time.time() - t0
        result["wall_time_s"] = elapsed
        all_results.append(result)

        print(f"  R_task={result['R_task']:.3f}  R_cov={result['R_cov']:.3f}  "
              f"R_fail|cov={result['R_fail_given_cov']:.3f}")
        act_pct = result.get('action_pct', {})
        print(f"  Actions: {result.get('action_distribution', {})}")
        print(f"  Action %: INS={act_pct.get('ins',act_pct.get('INS',0)):.1%}  "
              f"SAFE={act_pct.get('safe',act_pct.get('SAFE',0)):.1%}  "
              f"REC={act_pct.get('rec',act_pct.get('REC',0)):.1%}  "
              f"TX={act_pct.get('tx',act_pct.get('TX',0)):.1%}  "
              f"BACK={act_pct.get('back',act_pct.get('BACK',0)):.1%}")
        print(f"  100m-boundary steps: {result.get('boundary_near_steps',0)} "
              f"({result.get('boundary_near_pct',0):.1%})")
        print(f"  Nofly steps: {result.get('nofly_steps',0)}")
        print(f"  Goal switches: {result.get('goal_switches',0)}")
        print(f"  Stuck steps: {result.get('stuck_steps',0)}")

    # Comparison
    print("\n" + "=" * 72)
    print("  COMPARISON")
    print("=" * 72)
    for r in all_results:
        print(f"\n{r['label']}:")
        print(f"  R_task={r['R_task']:.3f}  R_cov={r['R_cov']:.3f}  "
              f"R_fail|cov={r['R_fail_given_cov']:.3f}")
        print(f"  Actions: INS={r['action_pct'].get('ins',0):.1%} "
              f"SAFE={r['action_pct'].get('safe',0):.1%} "
              f"REC={r['action_pct'].get('rec',0):.1%} "
              f"TX={r['action_pct'].get('tx',0):.1%} "
              f"BACK={r['action_pct'].get('back',0):.1%}")
        print(f"  100m-boundary: {r['boundary_near_pct']:.1%}  "
              f"Nofly: {r['nofly_steps']}  Switches: {r['goal_switches']}  "
              f"Stuck: {r['stuck_steps']}")

    # Save
    output_dir = Path("results_v2/v5_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "v5_diagnostic.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_dir / 'v5_diagnostic.json'}")


if __name__ == "__main__":
    main()
