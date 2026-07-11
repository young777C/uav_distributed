#!/usr/bin/env python3
"""Comprehensive diagnostic for D-EPA-RHP failure analysis.

Tracks all metrics from the root-cause checklist:
- Candidate filtering statistics (hard constraints, probability penalties)
- Action distribution and mode-switch counts
- GCS/UAV probability divergence (Brier-like tracking)
- Return queue dynamics (peak pending, timeout violations)
- Replan trigger frequency and effectiveness
- Per-POI failure attribution
- Per-step mode and position tracing
"""
from __future__ import annotations

import json, math, sys, os
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import FastObservation, SlowObservation, SlowPlan, FastCommand
from uavlab.paper1.runner.return_scheduling import should_attempt_key_return, goal_spatial_complete
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config
from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
from uavlab.paper2.runner.run import _slow_loop_params_from_cfg, _fast_loop_params_from_cfg
from uavlab.paper2.loops.slow_loop_prob import Paper2SlowLoop, _compute_poi_probability
from uavlab.paper2.loops.fast_loop_voi import Paper2FastLoop, UavAction


@dataclass
class DiagnosticState:
    # ── Candidate filtering ──
    total_candidates_considered: int = 0
    filtered_by_effective: int = 0
    filtered_by_boundary: int = 0  # POIs excluded by boundary penalty > 0.5
    filtered_by_energy_hard: int = 0
    low_prob_candidates: int = 0    # prob < 0.05
    candidate_retention_rate: float = 1.0

    # ── Action distribution ──
    action_steps: Counter = field(default_factory=Counter)
    mode_switch_count: int = 0
    action_durations: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))

    # ── Probability tracking ──
    gcs_prob_history: List[float] = field(default_factory=list)
    uav_prob_history: List[float] = field(default_factory=list)
    discrepancy_history: List[float] = field(default_factory=list)

    # ── Return queue ──
    queue_peak_pending: int = 0
    queue_timeout_violations: int = 0
    queue_history: List[int] = field(default_factory=list)

    # ── Replanning ──
    replan_triggers: int = 0
    replan_effective: int = 0   # replans that changed the goal
    replan_ineffective: int = 0  # replans that kept same goal
    feedback_triggers: int = 0
    feedback_changed_decision: int = 0

    # ── Failure attribution ──
    poi_failures: Dict[int, str] = field(default_factory=dict)  # poi_id -> failure_reason
    failure_reasons: Counter = field(default_factory=Counter)

    # ── Safety ──
    safety_mode_triggers: int = 0
    fsm_fallback_count: int = 0
    boundary_proximity_events: int = 0

    # ── Per-step trace (sampled) ──
    trace: List[Dict] = field(default_factory=list)



def run_diagnostic_episode(config_path: str, seed: int = 42) -> Tuple[Dict, DiagnosticState]:
    """Run one diagnostic episode with full instrumentation."""
    ds = DiagnosticState()

    cfg = apply_experiment_presets(load_resolved_config(config_path))
    scene_path = str(cfg.get('scene_file', ''))
    scene_raw = load_scene_yaml(scene_path) if Path(scene_path).exists() else None
    scene_geom = load_scene_config(scene_path) if Path(scene_path).exists() else None
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=float(sim_cfg.waypoint_delta_max_m))
    slow_interval = int(cfg.get('paper1_loops', {}).get('slow_loop', {}).get('slow_interval_steps', 40))
    dt = 1.0 / max(1, int(sim_cfg.step_hz))
    max_steps = int(sim_cfg.episode_steps)

    env.reset()
    slow_loop = Paper2SlowLoop(env=env, params=_slow_loop_params_from_cfg(cfg),
                                contract=contract, slow_interval_steps=slow_interval)
    fast_loop = Paper2FastLoop(env=env, params=_fast_loop_params_from_cfg(cfg), contract=contract)
    coupling = CouplingPolicy(mode=str(contract.coupling_mode))
    slow_loop.reset()
    fast_loop.reset()

    # Bootstrap
    link0 = env.observe_link_state()
    plan = SlowPlan(goal_id=None, goal_ne=(float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1])))
    pkt0 = coupling.build_fast_to_slow_packet(contract=contract, env=env, step=0,
        link_loss_p=float(link0.loss_p), link_delay_s=float(link0.delay_s),
        link_bandwidth_bps=float(link0.bandwidth_bps),
        control_link_lost=fast_loop.control_link_lost,
        control_link_lost_duration_s=fast_loop.control_link_lost_duration_s, plan=plan)
    plan = slow_loop.step(SlowObservation(step=0, fast_to_slow=pkt0))

    prev_mode = None
    current_mode_duration = 0
    prev_goal = plan.goal_id
    poi_cover_times: Dict[int, int] = {}
    poi_return_times: Dict[int, int] = {}
    last_replan_goal = plan.goal_id
    gcs_probs_at_replan: List[float] = []

    for step in range(1, max_steps + 1):
        if env.done():
            break

        link = env.observe_link_state()

        # ── Track candidate filtering ──
        if step % 40 == 1:  # sample on replan-aligned steps
            done_eff = set(env.effective)
            all_pois = len(env.pois)
            remaining = all_pois - len(done_eff)
            # Count boundary-proximate POIs
            n_min, n_max = float(env.cfg.n_min), float(env.cfg.n_max)
            margin = 100.0
            boundary_pois = 0
            for poi in env.pois:
                if int(poi.poi_id) in done_eff:
                    continue
                px, py = float(poi.pos_ne[0]), float(poi.pos_ne[1])
                if (px < n_min + margin * 2 or px > n_max - margin * 2 or
                    py < n_min + margin * 2 or py > n_max - margin * 2):
                    boundary_pois += 1
            ds.filtered_by_effective = len(done_eff)
            ds.filtered_by_boundary = boundary_pois
            ds.total_candidates_considered = remaining
            ds.candidate_retention_rate = max(0, remaining - boundary_pois) / max(1, remaining)

        # ── Fast loop ──
        obs = FastObservation(step=int(env.t), pos_ne=env.pos_ne, comm_mode=env.comm_mode,
                              backlog_bits=float(env.backlog_bits), link_loss_p=float(link.loss_p))
        fast_result = fast_loop.step(obs=obs, plan=plan, dt=dt)
        cmd, local_prob = fast_result if isinstance(fast_result, tuple) else (fast_result, 0.0)
        mode_str = str(cmd.next_comm_mode.value)

        # Action tracking
        ds.action_steps[mode_str] += 1
        if mode_str != prev_mode:
            if prev_mode is not None:
                ds.action_durations[prev_mode].append(current_mode_duration)
            ds.mode_switch_count += 1
            current_mode_duration = 0
            prev_mode = mode_str
        current_mode_duration += 1

        # Check FSM fallback (SAFE via Paper1 delegation)
        if hasattr(fast_loop, '_p1') and mode_str == 'Ssafe':
            ds.fsm_fallback_count += 1

        # Probability tracking
        if hasattr(slow_loop, '_last_gcs_prob'):
            ds.gcs_prob_history.append(float(slow_loop._last_gcs_prob))
        ds.uav_prob_history.append(float(local_prob))
        if hasattr(slow_loop, '_last_feedback_decision') and slow_loop._last_feedback_decision is not None:
            ds.discrepancy_history.append(slow_loop._last_feedback_decision.discrepancy)

        # ── Apply command ──
        prev_covered = set(env.covered)
        prev_effective = set(env.effective)
        env.comm_mode = cmd.next_comm_mode
        env.step_fast(target_ne=cmd.target_ne, vel_ne_cmd=cmd.vel_ne_cmd, dt=dt,
                      approach_goal_ne=cmd.approach_goal_ne)

        # Track safety events
        if getattr(env, 'in_nofly', lambda _: False)(env.pos_ne):
            ds.safety_mode_triggers += 1
        n_min_m = float(env.cfg.n_min) + 100.0
        n_max_m = float(env.cfg.n_max) - 100.0
        if (env.pos_ne[0] < n_min_m or env.pos_ne[0] > n_max_m or
            env.pos_ne[1] < n_min_m or env.pos_ne[1] > n_max_m):
            ds.boundary_proximity_events += 1

        # Track new coverage/effective
        new_covered = set(env.covered) - prev_covered
        new_effective = set(env.effective) - prev_effective
        for pid in new_covered:
            poi_cover_times[int(pid)] = step
        for pid in new_effective:
            poi_return_times[int(pid)] = step

        # ── Key data return ──
        if should_attempt_key_return(
            backlog_bits=float(env.backlog_bits), comm_mode=env.comm_mode,
            enable_fast_mode_switch=bool(contract.enable_fast_mode_switch),
            spatial_complete=goal_spatial_complete(env=env, goal_id=plan.goal_id),
            return_phase=plan.goal_id is None,
            fast_upload_mode=str(env.cfg.fast_upload_mode),
            fixed_send_ratio=float(env.cfg.fixed_send_ratio), step=int(env.t)):
            env.progress_key_return(dt_s=dt, params=contract.dual_link.to_return_params())

        # ── Return queue tracking ──
        ret_q = getattr(env, 'return_queue', None)
        if ret_q is not None:
            pending = int(ret_q.pending_count)
            ds.queue_history.append(pending)
            ds.queue_peak_pending = max(ds.queue_peak_pending, pending)
            # Check for timeout: POIs older than max_return_time_s
            now_s = float(env.t) * dt
            stall_s = float(contract.dual_link.data_max_return_time_s)
            for buf_id in ret_q.pending_poi_ids():
                if ret_q.is_poi_return_stalled(int(buf_id), now_s=now_s, stall_s=stall_s):
                    ds.queue_timeout_violations += 1

        # ── Slow loop ──
        pkt = coupling.build_fast_to_slow_packet(contract=contract, env=env, step=int(env.t),
            link_loss_p=float(link.loss_p), link_delay_s=float(link.delay_s),
            link_bandwidth_bps=float(link.bandwidth_bps),
            control_link_lost=fast_loop.control_link_lost,
            control_link_lost_duration_s=fast_loop.control_link_lost_duration_s, plan=plan)
        prev_goal_before_step = plan.goal_id
        plan = slow_loop.step(SlowObservation(step=int(env.t), fast_to_slow=pkt))

        # ── Replan tracking ──
        if hasattr(slow_loop, '_last_feedback_decision') and slow_loop._last_feedback_decision is not None:
            ds.feedback_triggers += 1
            if plan.goal_id != prev_goal_before_step:
                ds.feedback_changed_decision += 1

        if plan.goal_id != prev_goal:
            if hasattr(slow_loop, 'last_replan_reason'):
                ds.replan_triggers += 1
            if plan.goal_id != last_replan_goal:
                ds.replan_effective += 1
            else:
                ds.replan_ineffective += 1
            last_replan_goal = plan.goal_id
            prev_goal = plan.goal_id

        # ── Per-step trace (sample every 2000 steps) ──
        if step % 2000 == 0 or step < 20:
            ds.trace.append({
                'step': step, 'goal_id': plan.goal_id,
                'pos': (float(env.pos_ne[0]), float(env.pos_ne[1])),
                'mode': mode_str, 'loss_p': float(link.loss_p),
                'covered': len(env.covered), 'effective': len(env.effective),
                'backlog': float(env.backlog_bits),
                'queue_pending': ret_q.pending_count if ret_q else 0,
                'energy': float(env.remaining_energy),
            })

    # ── Final failure attribution ──
    covered_not_returned = set(env.covered) - set(env.returned)
    for pid in covered_not_returned:
        reason = "unknown"
        if pid in poi_cover_times:
            cover_step = poi_cover_times[pid]
            remaining_steps = max_steps - cover_step
            if remaining_steps < 200:
                reason = "covered_too_late"
            else:
                # Check if data was enqueued
                reason = "return_timeout_or_link_degraded"
        else:
            reason = "covered_but_not_tracked"
        ds.poi_failures[int(pid)] = reason
        ds.failure_reasons[reason] += 1

    # Also attribute late-cover POIs
    all_poi_ids = {int(p.poi_id) for p in env.pois}
    not_covered = all_poi_ids - set(env.covered)
    for pid in not_covered:
        ds.poi_failures[int(pid)] = "never_covered"
        ds.failure_reasons["never_covered"] += 1

    # Metrics
    n_pois = max(1, len(env.pois))
    metrics = {
        'R_task': len(env.effective) / n_pois,
        'R_cov': len(env.covered) / n_pois,
        'R_fail_given_cov': max(0.0, 1.0 - len(env.effective) / max(len(env.covered), 1)),
        'R_oob': 1.0 if getattr(env, 'terminated_by_oob', False) else 0.0,
        'R_home': 1.0 if getattr(env, 'terminated_by_returned_home', False) else 0.0,
        'energy_remaining': float(env.remaining_energy),
        'episode_steps': step,
        'n_replan': getattr(slow_loop, 'replan_count', 0),
        'n_feedback': slow_loop.feedback_policy.feedback_count if hasattr(slow_loop, 'feedback_policy') else 0,
    }

    return metrics, ds


def print_diagnostic_report(metrics: Dict, ds: DiagnosticState):
    """Print comprehensive diagnostic report."""
    print("=" * 70)
    print("D-EPA-RHP DIAGNOSTIC REPORT")
    print("=" * 70)

    print(f"\n── Episode Metrics ──")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n── Candidate Filtering ──")
    print(f"  Total candidates considered: {ds.total_candidates_considered}")
    print(f"  Filtered by effective:       {ds.filtered_by_effective}")
    print(f"  Near-boundary (penalized):   {ds.filtered_by_boundary}")
    print(f"  Candidate retention rate:    {ds.candidate_retention_rate:.3f}")

    print(f"\n── Action Distribution ──")
    total_action_steps = sum(ds.action_steps.values())
    for action, count in ds.action_steps.most_common():
        pct = 100.0 * count / max(1, total_action_steps)
        avg_dur = (sum(ds.action_durations.get(action, [0])) /
                   max(1, len(ds.action_durations.get(action, [1]))))
        print(f"  {action:8s}: {count:6d} steps ({pct:5.1f}%)  avg_duration={avg_dur:.1f} steps")
    print(f"  Mode switches: {ds.mode_switch_count}")

    print(f"\n── Probability Calibration ──")
    if ds.gcs_prob_history and ds.uav_prob_history:
        import numpy as np
        diffs = [abs(g - u) for g, u in zip(ds.gcs_prob_history[-1000:], ds.uav_prob_history[-1000:])]
        print(f"  GCS prob mean (last 1000): {np.mean(ds.gcs_prob_history[-1000:]):.4f}")
        print(f"  UAV prob mean (last 1000): {np.mean(ds.uav_prob_history[-1000:]):.4f}")
        print(f"  Mean |GCS - UAV|:           {np.mean(diffs):.4f}")
        print(f"  Max |GCS - UAV|:            {np.max(diffs) if diffs else 0:.4f}")
        print(f"  Discrepancy samples:        {len(ds.discrepancy_history)}")

    print(f"\n── Return Queue Dynamics ──")
    print(f"  Peak pending POIs:  {ds.queue_peak_pending}")
    print(f"  Timeout violations: {ds.queue_timeout_violations}")
    if ds.queue_history:
        import numpy as np
        print(f"  Mean queue depth:   {np.mean(ds.queue_history):.1f}")
        print(f"  Queue > 0 steps:    {sum(1 for q in ds.queue_history if q > 0)}")

    print(f"\n── Replanning Efficiency ──")
    print(f"  Replan triggers:        {ds.replan_triggers}")
    print(f"  Effective (changed goal): {ds.replan_effective}")
    print(f"  Ineffective (same goal):  {ds.replan_ineffective}")
    effect_pct = 100.0 * ds.replan_effective / max(1, ds.replan_triggers)
    print(f"  Effectiveness: {effect_pct:.1f}%")
    print(f"  Feedback triggers:       {ds.feedback_triggers}")
    print(f"  Feedback → goal change:  {ds.feedback_changed_decision}")

    print(f"\n── Safety Events ──")
    print(f"  SAFE mode triggers:     {ds.safety_mode_triggers}")
    print(f"  FSM fallback count:     {ds.fsm_fallback_count}")
    print(f"  Boundary proximity:     {ds.boundary_proximity_events}")

    print(f"\n── Failure Attribution ──")
    for reason, count in ds.failure_reasons.most_common():
        print(f"  {reason}: {count} POIs")

    print(f"\n── Trace (sampled) ──")
    for t in ds.trace[:5]:
        print(f"  step={t['step']:5d} goal={t['goal_id']} pos=({t['pos'][0]:.0f},{t['pos'][1]:.0f}) mode={t['mode']} loss={t['loss_p']:.3f} cov={t['covered']} eff={t['effective']} queue={t['queue_pending']} energy={t['energy']:.3f}")
    if len(ds.trace) > 5:
        for t in ds.trace[-3:]:
            print(f"  step={t['step']:5d} goal={t['goal_id']} pos=({t['pos'][0]:.0f},{t['pos'][1]:.0f}) mode={t['mode']} loss={t['loss_p']:.3f} cov={t['covered']} eff={t['effective']} queue={t['queue_pending']} energy={t['energy']:.3f}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/experiments/paper2/c3_g2_m2_d_epa_rhp.yaml'
    metrics, ds = run_diagnostic_episode(config_path)
    print_diagnostic_report(metrics, ds)
