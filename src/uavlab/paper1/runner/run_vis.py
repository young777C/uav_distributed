from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, IO, Optional, Set

from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import FastObservation, SlowObservation, SlowPlan
from uavlab.paper1.coupling.coupling_policy import CouplingPolicy
from uavlab.paper1.loops.fast.fast_loop import FastLoop
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.loops.slow.slow_loop import SlowLoop
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config
from uavlab.paper1.metrics.link_recovery import link_recovery_recorder_from_contract
from uavlab.paper1.types import FastCommState
from uavlab.paper1.runner.return_scheduling import goal_spatial_complete, should_attempt_key_return
from uavlab.related_models.key_data_return import ReturnDecisionParams


def _diag_write(f: IO[str], row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def _slow_snapshot(slow: SlowLoop, *, plan: SlowPlan) -> Dict[str, Any]:
    gid = plan.goal_id
    dwell_steps = None
    if gid is not None and 0 <= int(gid) < len(slow.env.poi_dwell_steps):
        dwell_steps = int(slow.env.poi_dwell_steps[int(gid)])
    return {
        "goal_id": gid,
        "goal_ne": [float(plan.goal_ne[0]), float(plan.goal_ne[1])],
        "planned_sequence": [int(x) for x in list(slow._last_sequence)],
        "future_sequence": [int(x) for x in list(slow._future_sequence)],
        "slow_goal_id": slow._goal_id,
        "dwell_steps_on_goal": dwell_steps,
    }


def _log_coverage_edges(
    diag_f: IO[str],
    *,
    episode: int,
    t: int,
    dt: float,
    env: Any,
    prev_covered: Set[int],
    prev_returned: Set[int],
    goal_id: Optional[int],
    goal_assign_step: Dict[int, int],
) -> tuple[Set[int], Set[int]]:
    covered_now = set(int(x) for x in env.covered)
    returned_now = set(int(x) for x in env.returned)
    for pid in sorted(covered_now - prev_covered):
        assign_t = goal_assign_step.get(int(goal_id)) if goal_id is not None else None
        leg_s = None
        if assign_t is not None:
            leg_s = float(max(0, t - assign_t)) * float(dt)
        _diag_write(
            diag_f,
            {
                "event": "spatial_complete",
                "episode": int(episode),
                "t": int(t),
                "poi_id": int(pid),
                "active_goal_id": goal_id,
                "goal_assign_t": assign_t,
                "leg_wall_s": leg_s,
                "dwell_steps": int(env.poi_dwell_steps[int(pid)])
                if 0 <= int(pid) < len(env.poi_dwell_steps)
                else None,
                "poi_dwell_s": float(env.cfg.poi_dwell_s),
            },
        )
    for pid in sorted(returned_now - prev_returned):
        _diag_write(
            diag_f,
            {
                "event": "effective_complete",
                "episode": int(episode),
                "t": int(t),
                "poi_id": int(pid),
                "active_goal_id": goal_id,
            },
        )
    return covered_now, returned_now


def main() -> None:
    p = argparse.ArgumentParser(description="Paper1 lightweight runner (task-level, with POI status logging).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slow_interval_steps", type=int, default=40)
    p.add_argument("--comm_mode", type=str, default="")
    p.add_argument("--run_dir", type=str, default="")
    p.add_argument("--metrics_jsonl", type=str, default="")
    p.add_argument("--traj_jsonl", type=str, default="", help="Optional per-step trajectory log (JSONL).")
    p.add_argument(
        "--diag_jsonl",
        type=str,
        default="",
        help="Slow-loop / goal / coverage diagnostic log (JSONL). Default: <run_dir>/diag.jsonl when --run_dir is set.",
    )
    args = p.parse_args()

    cfg = apply_experiment_presets(load_resolved_config(args.config))
    scene_path = str(cfg.get("scene_file", "configs/scenes/g1_uniform.yaml"))
    scene_raw = load_scene_yaml(scene_path)
    scene_geom = load_scene_config(scene_path)
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)

    dt = 1.0 / float(max(1, sim_cfg.step_hz))
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=float(sim_cfg.waypoint_delta_max_m))
    fast_params = FastLoopParams.from_contract(contract)
    dual_link = contract.dual_link
    ret_params = dual_link.to_return_params()
    slow_params = SlowLoopParams(
        t_obs_s=float(sim_cfg.poi_dwell_s),
        t_safe_s=0.0,
        dual_link=dual_link,
    )
    slow = SlowLoop(env=env, params=slow_params, contract=contract, slow_interval_steps=int(args.slow_interval_steps))
    fast = FastLoop(env=env, params=fast_params, contract=contract)
    coupling = CouplingPolicy(mode=str(contract.coupling_mode))
    link_rec = link_recovery_recorder_from_contract(contract, dt=float(dt))

    run_dir: Optional[Path] = None
    if args.run_dir.strip():
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    metrics_f = None
    if args.metrics_jsonl.strip():
        mp = Path(args.metrics_jsonl)
        mp.parent.mkdir(parents=True, exist_ok=True)
        metrics_f = mp.open("w", encoding="utf-8")

    traj_f = None
    if args.traj_jsonl.strip():
        tp = Path(args.traj_jsonl)
        tp.parent.mkdir(parents=True, exist_ok=True)
        traj_f = tp.open("w", encoding="utf-8")
    elif run_dir is not None:
        traj_f = (run_dir / "traj.jsonl").open("w", encoding="utf-8")

    diag_f = None
    if args.diag_jsonl.strip():
        dp = Path(args.diag_jsonl)
        dp.parent.mkdir(parents=True, exist_ok=True)
        diag_f = dp.open("w", encoding="utf-8")
    elif run_dir is not None:
        diag_f = (run_dir / "diag.jsonl").open("w", encoding="utf-8")

    exp = cfg.get("experiment") if isinstance(cfg.get("experiment"), dict) else {}
    exp_id = exp.get("id")
    p1 = exp.get("paper1") if isinstance(exp.get("paper1"), dict) else {}
    env_case = exp.get("env_case") if isinstance(exp.get("env_case"), dict) else {}

    for ep in range(int(args.episodes)):
        _ = env.reset(seed=int(args.seed) + ep)
        slow.reset()
        fast.reset()
        link_rec.reset()
        if args.comm_mode.strip():
            try:
                env.comm_mode = FastCommState(str(args.comm_mode))
            except Exception:
                env.comm_mode = str(args.comm_mode)

        prev_covered: Set[int] = set()
        prev_returned: Set[int] = set()
        goal_assign_step: Dict[int, int] = {}
        prev_goal_id: Optional[int] = None
        prev_sequence: list[int] = []
        goal_switch_count = 0
        mode_step_counts: Dict[str, int] = {}
        pending_return_max_bits = 0.0

        exec_plan = SlowPlan(goal_id=None, goal_ne=(float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1])))
        link0 = env.observe_link_state()
        pkt0 = coupling.build_fast_to_slow_packet(
            contract=contract,
            env=env,
            step=int(env.t),
            link_loss_p=float(link0.loss_p),
            link_delay_s=float(link0.delay_s),
            link_bandwidth_bps=float(link0.bandwidth_bps),
            control_link_lost=fast.control_link_lost,
            control_link_lost_duration_s=fast.control_link_lost_duration_s,
            plan=exec_plan,
        )
        exec_plan = slow.step(SlowObservation(step=int(env.t), fast_to_slow=pkt0))

        if diag_f is not None:
            snap = _slow_snapshot(slow, plan=exec_plan)
            _diag_write(
                diag_f,
                {
                    "event": "slow_bootstrap",
                    "episode": int(ep),
                    "t": int(env.t),
                    **snap,
                    "covered_n": len(env.covered),
                    "effective_n": len(env.returned),
                },
            )
            if exec_plan.goal_id is not None:
                goal_assign_step[int(exec_plan.goal_id)] = int(env.t)
            prev_goal_id = exec_plan.goal_id
            prev_sequence = list(snap["planned_sequence"])

        while not env.done():
            link = env.observe_link_state()
            cmd = fast.step(
                FastObservation(
                    step=int(env.t),
                    pos_ne=env.pos_ne,
                    comm_mode=env.comm_mode,
                    backlog_bits=float(env.backlog_bits),
                    link_loss_p=float(link.loss_p),
                ),
                exec_plan,
                dt=float(dt),
            )
            env.comm_mode = cmd.next_comm_mode
            mode_key = str(getattr(env.comm_mode, "value", env.comm_mode)).strip().lower()
            mode_step_counts[mode_key] = int(mode_step_counts.get(mode_key, 0)) + 1
            pending_return_max_bits = max(pending_return_max_bits, float(env.backlog_bits))
            env.step_fast(
                target_ne=cmd.target_ne,
                vel_ne_cmd=cmd.vel_ne_cmd,
                dt=float(dt),
                approach_goal_ne=cmd.approach_goal_ne,
            )
            link_rec.observe(step=int(env.t), loss_p=float(link.loss_p), comm_mode=env.comm_mode)

            if should_attempt_key_return(
                backlog_bits=float(env.backlog_bits),
                comm_mode=env.comm_mode,
                enable_fast_mode_switch=bool(contract.enable_fast_mode_switch),
                spatial_complete=goal_spatial_complete(env=env, goal_id=exec_plan.goal_id),
                return_phase=exec_plan.goal_id is None,
            ):
                env.try_return_key(params=ret_params)

            seq_before = list(slow._last_sequence)
            goal_before = exec_plan.goal_id
            pkt = coupling.build_fast_to_slow_packet(
                contract=contract,
                env=env,
                step=int(env.t),
                link_loss_p=float(link.loss_p),
                link_delay_s=float(link.delay_s),
                link_bandwidth_bps=float(link.bandwidth_bps),
                control_link_lost=fast.control_link_lost,
                control_link_lost_duration_s=fast.control_link_lost_duration_s,
                plan=exec_plan,
            )
            exec_plan = slow.step(SlowObservation(step=int(env.t), fast_to_slow=pkt))

            if diag_f is not None:
                snap = _slow_snapshot(slow, plan=exec_plan)
                seq_after = list(snap["planned_sequence"])
                if seq_after != seq_before or exec_plan.goal_id != goal_before:
                    _diag_write(
                        diag_f,
                        {
                            "event": "slow_replan",
                            "episode": int(ep),
                            "t": int(env.t),
                            "goal_before": goal_before,
                            "goal_after": exec_plan.goal_id,
                            "sequence_before": [int(x) for x in seq_before],
                            "replan_reason": str(slow.last_replan_reason),
                            "event_level": str(slow.last_event_level),
                            **snap,
                            "covered_n": len(env.covered),
                            "effective_n": len(env.returned),
                            "backlog_bits": float(env.backlog_bits),
                            "comm_mode": str(env.comm_mode),
                        },
                    )
                if exec_plan.goal_id != prev_goal_id:
                    if exec_plan.goal_id is not None:
                        goal_assign_step[int(exec_plan.goal_id)] = int(env.t)
                    _diag_write(
                        diag_f,
                        {
                            "event": "goal_switch",
                            "episode": int(ep),
                            "t": int(env.t),
                            "prev_goal_id": prev_goal_id,
                            "goal_id": exec_plan.goal_id,
                            "planned_sequence_head": seq_after[:8],
                        },
                    )
                    prev_goal_id = exec_plan.goal_id
                    goal_switch_count += 1
                    prev_sequence = seq_after

                if int(env.t) % int(max(1, args.slow_interval_steps)) == 0:
                    _diag_write(
                        diag_f,
                        {
                            "event": "slow_periodic_tick",
                            "episode": int(ep),
                            "t": int(env.t),
                            **snap,
                        },
                    )

                prev_covered, prev_returned = _log_coverage_edges(
                    diag_f,
                    episode=ep,
                    t=int(env.t),
                    dt=float(dt),
                    env=env,
                    prev_covered=prev_covered,
                    prev_returned=prev_returned,
                    goal_id=exec_plan.goal_id,
                    goal_assign_step=goal_assign_step,
                )

            if traj_f is not None:
                link2 = env.observe_link_state()
                traj_f.write(
                    json.dumps(
                        {
                            "episode": ep,
                            "t": int(env.t),
                            "pos_ne": [float(env.pos_ne[0]), float(env.pos_ne[1])],
                            "goal_id": exec_plan.goal_id,
                            "backlog_bits": float(env.backlog_bits),
                            "in_nofly": bool(env.in_nofly(env.pos_ne)),
                            "in_obstacle": bool(env.in_nofly(env.pos_ne)),
                            "in_blackhole": bool(env.in_blackhole(env.pos_ne)),
                            "link_loss_p": float(link2.loss_p),
                            "comm_mode": str(env.comm_mode),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        row: Dict[str, Any] = {
            "experiment_id": exp_id,
            "paper1": p1,
            "env_case": env_case,
            "episode": ep,
            "covered_ids": sorted(list(env.covered)),
            "returned_ids": sorted(list(env.returned)),
            "effective_ids": sorted(list(env.returned)),
            **env.metrics(link_recovery_latencies_s=link_rec.latencies_s()),
        }
        total_steps = int(max(1, env.t))
        row["mode_time_ratio"] = {k: float(v) / float(total_steps) for k, v in mode_step_counts.items()}
        row["goal_switch_count"] = int(goal_switch_count)
        row["replan_count"] = int(slow.replan_count)
        row["pending_return_max_bits"] = float(pending_return_max_bits)
        print(
            f"[Paper1Lite] ep={ep} R_cov={row['R_cov']:.3f} R_task={row['R_task']:.3f} "
            f"R_fail|cov={row['R_fail_given_cov']:.3f} T_nf={row['T_nf_s']:.2f}s"
        )
        if metrics_f is not None:
            metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_f.flush()

    if metrics_f is not None:
        metrics_f.close()
    if traj_f is not None:
        traj_f.close()
    if diag_f is not None:
        diag_f.close()


if __name__ == "__main__":
    main()
