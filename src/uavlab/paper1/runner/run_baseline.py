"""
Paper1 external baseline runner (RHC-Inspection / CBCP).

Usage:
    python -m uavlab.paper1.runner.run_baseline --config <config.yaml> [options]

The config must set ``experiment.paper1.external_baseline`` to ``rhc_inspection``
or ``cbcp``.  No FDLC-specific contract (SlowLoop / FastLoop / CouplingPolicy)
is created — the runner implements a minimal main loop with simple tracking
and periodic replanning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Set, Tuple

from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper1.baselines.common import BaselinePlannerInput, BaselinePlannerOutput
from uavlab.paper1.baselines.rhc_inspection import RHCInspectionPlanner
from uavlab.paper1.baselines.cbcp import CBCPPlanner
from uavlab.paper1.metrics.link_recovery import link_recovery_recorder_from_contract
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.sim.config import from_resolved_config
from uavlab.paper1.sim.env import build_env
from uavlab.paper1.sim.scene_loader import load_scene_yaml
from uavlab.scene.loader import load_scene_config
from uavlab.paper1.types import FastCommState
from uavlab.paper1.runner.run_context import format_paper1_episode_line, print_paper1_run_header


# =========================================================================
# Shared helpers (aligned with run_vis.py)
# =========================================================================

def _diag_write(f: IO[str], row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def _compute_delivery_latency_distribution(
    covered: Set[int],
    poi_cov_time_s: Dict[int, float],
    poi_return_time_s: Dict[int, float],
) -> Tuple[float, float, float]:
    latencies: List[float] = []
    for pid in covered:
        cov_t = poi_cov_time_s.get(pid)
        ret_t = poi_return_time_s.get(pid)
        if cov_t is not None and ret_t is not None and ret_t >= cov_t:
            latencies.append(float(ret_t - cov_t))
    if not latencies:
        return float("nan"), float("nan"), float("nan")
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    mean_val = float(statistics.mean(sorted_lat))
    median_val = float(sorted_lat[n // 2] if n % 2 == 1 else (sorted_lat[n // 2 - 1] + sorted_lat[n // 2]) / 2.0)
    p90_idx = max(0, int(math.ceil(0.90 * n)) - 1)
    p90_val = float(sorted_lat[p90_idx])
    return mean_val, median_val, p90_val


def _git_commit(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(project_root), stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _config_hash(cfg: Dict[str, Any]) -> str:
    try:
        raw = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]
    except Exception:
        return ""


def _scene_hash(scene_raw: Dict[str, Any]) -> str:
    try:
        raw = json.dumps(scene_raw, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]
    except Exception:
        return ""


def _env_info() -> Dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": str(os.cpu_count() or 0),
    }


# =========================================================================
# Helper: extract uncovered POI IDs from the environment
# =========================================================================

def _uncovered_ids(env) -> list[int]:
    covered = set(int(x) for x in env.covered)
    effective = set(int(x) for x in env.effective)
    return [int(p.poi_id) for p in env.pois if int(p.poi_id) not in effective]


def _all_covered(env) -> bool:
    return len(_uncovered_ids(env)) == 0


# =========================================================================
# Helper: energy-return threshold (same logic as SlowLoop._should_enter_return_phase)
# =========================================================================

def _return_energy_threshold(env) -> float:
    """Return-home energy + safe margin."""
    from uavlab.paper1.sim.energy_model import energy_return_need_from_env

    need = energy_return_need_from_env(env)
    margin = float(env.cfg.energy_safe_margin)
    return float(need + margin)


def _should_return_to_gcs(env) -> bool:
    """Check conditions that trigger the return-to-GCS phase."""
    # All POIs covered → return
    n_pois = len(env.pois)
    if n_pois > 0 and len(env.effective) >= n_pois:
        return True
    # Time running out
    hz = float(max(1, int(env.cfg.step_hz)))
    total_s = float(env.cfg.episode_steps) / hz
    elapsed_s = float(env.t) / hz
    remain_s = float(max(0.0, total_s - elapsed_s))
    reserve_s = float(env.cfg.return_reserve_s)
    if reserve_s > 0.0 and remain_s < reserve_s:
        return True
    # Energy too low
    if float(env.remaining_energy) < _return_energy_threshold(env):
        return True
    return False


# =========================================================================
# Planner factory
# =========================================================================

_PLANNER_CLASSES = {
    "rhc_inspection": RHCInspectionPlanner,
    "cbcp": CBCPPlanner,
}


def _create_planner(baseline_type: str):
    cls = _PLANNER_CLASSES.get(str(baseline_type).strip().lower())
    if cls is None:
        raise ValueError(
            f"Unknown external_baseline {baseline_type!r}. "
            f"Known: {sorted(_PLANNER_CLASSES.keys())}"
        )
    return cls()


# =========================================================================
# Main runner
# =========================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Paper1 external baseline runner (RHC-Inspection / CBCP).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--system", type=str, default="", help="Optional path to system YAML (merged with --config).")
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
        help="Coverage / delivery diagnostic log (JSONL). Default: <run_dir>/diag.jsonl when --run_dir is set.",
    )
    p.add_argument(
        "--events_jsonl",
        type=str,
        default="",
        help="High-level event log (JSONL). Default: <run_dir>/events.jsonl when --run_dir is set.",
    )
    args = p.parse_args()

    # -- Load config --------------------------------------------------------
    cfg = apply_experiment_presets(load_resolved_config(args.config))
    if args.system.strip():
        sys_cfg = load_resolved_config(args.system)
        cfg.update({k: v for k, v in sys_cfg.items() if k not in cfg or cfg[k] != v})
        cfg = apply_experiment_presets(cfg)
    p1 = cfg.get("experiment", {}).get("paper1", {}) if isinstance(cfg.get("experiment"), dict) else {}
    baseline_type = str(p1.get("external_baseline", "") or "").strip().lower()
    if not baseline_type:
        print("[ERROR] config must set experiment.paper1.external_baseline")
        return

    # -- Build environment --------------------------------------------------
    scene_path = str(cfg.get("scene_file", "configs/scenes/g1_uniform.yaml"))
    scene_raw = load_scene_yaml(scene_path)
    scene_geom = load_scene_config(scene_path)
    sim_cfg = from_resolved_config(cfg, scene_raw)
    env = build_env(sim_cfg, scene_geom)

    dt = 1.0 / float(max(1, sim_cfg.step_hz))
    dual_link = sim_cfg.dual_link

    # -- Link recovery recorder ---------------------------------------------
    # Build a minimal contract for link_recovery_recorder.
    contract = Paper1ContractConfig.from_cfg(cfg, waypoint_delta_max_m=float(sim_cfg.waypoint_delta_max_m))
    link_rec = link_recovery_recorder_from_contract(contract, dt=float(dt))

    # -- Create planner -----------------------------------------------------
    planner = _create_planner(baseline_type)

    # -- Output setup -------------------------------------------------------
    run_dir: Optional[Path] = None
    if args.run_dir.strip():
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    metrics_f: Optional[IO[str]] = None
    if args.metrics_jsonl.strip():
        mp = Path(args.metrics_jsonl)
        mp.parent.mkdir(parents=True, exist_ok=True)
        metrics_f = mp.open("w", encoding="utf-8")

    traj_f: Optional[IO[str]] = None
    if args.traj_jsonl.strip():
        tp = Path(args.traj_jsonl)
        tp.parent.mkdir(parents=True, exist_ok=True)
        traj_f = tp.open("w", encoding="utf-8")
    elif run_dir is not None:
        traj_f = (run_dir / "traj.jsonl").open("w", encoding="utf-8")

    diag_f: Optional[IO[str]] = None
    if args.diag_jsonl.strip():
        dp = Path(args.diag_jsonl)
        dp.parent.mkdir(parents=True, exist_ok=True)
        diag_f = dp.open("w", encoding="utf-8")
    elif run_dir is not None:
        diag_f = (run_dir / "diag.jsonl").open("w", encoding="utf-8")

    events_f: Optional[IO[str]] = None
    if args.events_jsonl.strip():
        ep_path = Path(args.events_jsonl)
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        events_f = ep_path.open("w", encoding="utf-8")
    elif run_dir is not None:
        events_f = (run_dir / "events.jsonl").open("w", encoding="utf-8")

    exp = cfg.get("experiment") if isinstance(cfg.get("experiment"), dict) else {}
    exp_id = exp.get("id")
    env_case = exp.get("env_case") if isinstance(exp.get("env_case"), dict) else {}

    # -- System info --------------------------------------------------------
    project_root = Path(__file__).resolve().parents[4]
    git_commit = _git_commit(project_root)
    cfg_hash = _config_hash(cfg)
    scene_hash_val = _scene_hash(scene_raw)
    env_info = _env_info()
    baseline_label = str(baseline_type).strip().lower()
    run_id = f"{exp_id or 'run'}__baseline_{baseline_label}__Ts{int(args.slow_interval_steps)}__seed{int(args.seed)}"

    if run_dir is not None:
        (run_dir / "run_id.txt").write_text(run_id + "\n", encoding="utf-8")
        (run_dir / "git_commit.txt").write_text(git_commit + "\n", encoding="utf-8")
        (run_dir / "env_info.json").write_text(
            json.dumps(env_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    run_ctx = print_paper1_run_header(
        cfg,
        experiment_id=exp_id,
        slow_interval_steps=int(args.slow_interval_steps),
    )

    # -- Episode loop -------------------------------------------------------
    for ep in range(int(args.episodes)):
        t_ep_start = time.perf_counter()
        _ = env.reset(seed=int(args.seed) + ep)
        link_rec.reset()
        if args.comm_mode.strip():
            try:
                env.comm_mode = FastCommState(str(args.comm_mode))
            except Exception:
                env.comm_mode = str(args.comm_mode)

        # Bootstrap: initial plan
        goal_id: Optional[int] = None
        goal_pos = (float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1]))

        inp = _build_planner_input(env, baseline_type, sim_cfg, dual_link, args)
        out = planner.plan(inp)
        if out.target_poi_id is not None:
            goal_id = int(out.target_poi_id)
            goal_pos = _poi_position(env, goal_id)

        if events_f is not None:
            _diag_write(events_f, {
                "event": "episode_start",
                "run_id": run_id,
                "episode": int(ep),
                "t": 0,
                "seed": int(args.seed),
                "goal_id": goal_id,
            })

        # Track state deltas and stats per episode
        prev_covered: Set[int] = set()
        prev_returned: Set[int] = set()
        goal_switch_count = 0
        prev_goal_id: Optional[int] = None
        pending_return_max_bits = 0.0
        pending_time_integral = 0.0

        # Communication stats
        loss_vals: List[float] = []
        latency_vals: List[float] = []

        # Energy margin
        from uavlab.paper1.sim.energy_model import energy_return_need_from_env
        min_energy_margin = float("inf")

        # Planner stats
        planner_stats: Dict[str, Any] = {
            "planning_count": 0,
            "total_solve_time_s": 0.0,
            "solve_time_max_s": 0.0,
            "solve_times": [],
            "infeasible_count": 0,
            "fallback_count": 0,
        }

        while not env.done():
            # Accumulate comm stats
            link_state = env.observe_link_state()
            loss_vals.append(float(link_state.loss_p))
            latency_vals.append(float(link_state.delay_s))
            link_rec.observe(step=int(env.t), loss_p=float(link_state.loss_p), comm_mode=env.comm_mode)

            # --- Simple tracking (no FSM, no waypoint delta) ---------------
            target = goal_pos
            env.step_fast(target_ne=target, dt=dt, approach_goal_ne=goal_pos)

            # --- Key return progress (env uses real link state) ------------
            pending_return_max_bits = max(pending_return_max_bits, float(env.backlog_bits))
            pending_time_integral += float(env.backlog_bits) * float(dt)
            if float(env.backlog_bits) > 0:
                env.progress_key_return(dt_s=dt, params=dual_link.to_return_params())

            if traj_f is not None:
                traj_f.write(
                    json.dumps(
                        {
                            "episode": ep,
                            "t": int(env.t),
                            "pos_ne": [float(env.pos_ne[0]), float(env.pos_ne[1])],
                            "goal_id": goal_id,
                            "backlog_bits": float(env.backlog_bits),
                            "in_nofly": bool(env.in_nofly(env.pos_ne)),
                            "in_obstacle": bool(env.in_nofly(env.pos_ne)),
                            "in_blackhole": bool(env.in_blackhole(env.pos_ne)),
                            "link_loss_p": float(link_state.loss_p),
                            "comm_mode": str(env.comm_mode),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # --- Check if current goal is reached --------------------------
            if goal_id is not None and goal_id in env.covered:
                # POI covered; keep it as goal until replan decides next.
                pass

            # --- Return-to-GCS phase ---------------------------------------
            return_phase = _should_return_to_gcs(env)
            if return_phase:
                goal_id = None
                goal_pos = (float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1]))

            # --- Periodic replanning ---------------------------------------
            is_replan_tick = (
                int(env.t) == 0
                or (int(args.slow_interval_steps) > 0 and int(env.t) % int(args.slow_interval_steps) == 0)
            )

            if is_replan_tick and not return_phase:
                inp = _build_planner_input(env, baseline_type, sim_cfg, dual_link, args)
                out = planner.plan(inp)
                planner_stats["planning_count"] += 1
                st = float(out.solve_time_s)
                planner_stats["total_solve_time_s"] += st
                planner_stats["solve_time_max_s"] = max(planner_stats["solve_time_max_s"], st)
                planner_stats["solve_times"].append(st)
                if out.status in ("infeasible",):
                    planner_stats["infeasible_count"] += 1
                if out.status in ("fallback",):
                    planner_stats["fallback_count"] += 1

                if out.target_poi_id is not None:
                    new_id = int(out.target_poi_id)
                    if new_id != goal_id:
                        if events_f is not None:
                            _diag_write(events_f, {
                                "event": "goal_switch",
                                "run_id": run_id,
                                "episode": int(ep),
                                "t": int(env.t),
                                "seed": int(args.seed),
                                "prev_goal_id": goal_id,
                                "goal_id": new_id,
                            })
                        goal_switch_count += 1
                    goal_id = new_id
                    goal_pos = _poi_position(env, goal_id)
                else:
                    if goal_id is not None and events_f is not None:
                        _diag_write(events_f, {
                            "event": "goal_switch",
                            "run_id": run_id,
                            "episode": int(ep),
                            "t": int(env.t),
                            "seed": int(args.seed),
                            "prev_goal_id": goal_id,
                            "goal_id": None,
                        })
                    goal_id = None
                    goal_pos = (float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1]))

            # --- Energy margin tracking -------------------------------------
            try:
                need = energy_return_need_from_env(env)
                margin = float(env.remaining_energy) - float(need)
                min_energy_margin = min(min_energy_margin, margin)
            except Exception:
                pass

            # --- Track coverage/delivery edges for diag log -----------------
            covered_now = set(int(x) for x in env.covered)
            returned_now = set(int(x) for x in env.returned)
            if diag_f is not None:
                for pid in sorted(covered_now - prev_covered):
                    _diag_write(diag_f, {
                        "event": "spatial_complete",
                        "episode": int(ep),
                        "t": int(env.t),
                        "poi_id": int(pid),
                        "active_goal_id": goal_id,
                    })
                for pid in sorted(returned_now - prev_returned):
                    _diag_write(diag_f, {
                        "event": "effective_complete",
                        "episode": int(ep),
                        "t": int(env.t),
                        "poi_id": int(pid),
                        "active_goal_id": goal_id,
                    })
            prev_covered = covered_now
            prev_returned = returned_now

        # -- Episode termination event + termination.json -----------------
        base_metrics_pre = env.metrics(link_recovery_latencies_s=link_rec.latencies_s())
        term_reason = str(base_metrics_pre.get("termination_reason", ""))

        if events_f is not None:
            _diag_write(events_f, {
                "event": "episode_end",
                "run_id": run_id,
                "episode": int(ep),
                "seed": int(args.seed),
                "t": int(env.t),
                "termination_reason": term_reason,
                "R_task": float(base_metrics_pre.get("R_task", float("nan"))),
                "R_cov": float(base_metrics_pre.get("R_cov", float("nan"))),
                "covered_count": int(len(env.covered)),
                "returned_count": int(len(env.returned)),
                "goal_switch_count": int(goal_switch_count),
            })

        if run_dir is not None:
            term_path = run_dir / f"episode_{int(ep)}" / "termination.json"
            term_path.parent.mkdir(parents=True, exist_ok=True)
            term_path.write_text(json.dumps({
                "run_id": run_id,
                "episode": int(ep),
                "seed": int(args.seed),
                "termination_reason": term_reason,
                "steps": int(env.t),
                "R_task": float(base_metrics_pre.get("R_task", float("nan"))),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # -- Episode metrics ------------------------------------------------
        base_metrics = base_metrics_pre
        termination_reason = term_reason

        # Delivery latency distribution
        cov_t = dict(env.poi_cov_time_s)
        ret_t = dict(env.poi_return_time_s)
        del_mean, del_median, del_p90 = _compute_delivery_latency_distribution(
            set(env.covered), cov_t, ret_t,
        )

        # Solver timing distribution
        solve_times = planner_stats.pop("solve_times", [])
        solve_mean = float(statistics.mean(solve_times) * 1000.0) if solve_times else float("nan")
        solve_p95 = float(sorted(solve_times)[max(0, int(math.ceil(0.95 * len(solve_times))) - 1)] * 1000.0) if solve_times else float("nan")
        solve_max = float(max(solve_times) * 1000.0) if solve_times else float("nan")
        t_ep_wall = float(time.perf_counter() - t_ep_start)

        row: Dict[str, Any] = {
            # Identity
            "run_id": run_id,
            "stage": "overall",
            "method": baseline_label,
            "strategy": baseline_label,
            "scene_id": Path(scene_path).stem,
            "rho": int(args.slow_interval_steps),
            "seed": int(args.seed),
            "episode_id": int(ep),
            "experiment_id": exp_id,
            "paper1": p1,
            "env_case": env_case,
            "git_commit": git_commit,
            "config_hash": cfg_hash,
            "scene_hash": scene_hash_val,
            # Task metrics
            "episode": ep,
            "covered_ids": sorted(int(x) for x in env.covered),
            "returned_ids": sorted(int(x) for x in env.returned),
            "effective_ids": sorted(int(x) for x in env.returned),
            **base_metrics,
            "delivery_latency_mean_s": float(del_mean),
            "delivery_latency_median_s": float(del_median),
            "delivery_latency_p90_s": float(del_p90),
            "pending_peak_bits": float(pending_return_max_bits),
            "pending_time_integral_bit_s": float(pending_time_integral),
            # Communication
            "loss_mean": float(statistics.mean(loss_vals)) if loss_vals else float("nan"),
            "loss_p95": float(sorted(loss_vals)[max(0, int(math.ceil(0.95 * len(loss_vals))) - 1)]) if loss_vals else float("nan"),
            "latency_mean_ms": float(statistics.mean(latency_vals) * 1000.0) if latency_vals else float("nan"),
            "latency_p95_ms": float(sorted(latency_vals)[max(0, int(math.ceil(0.95 * len(latency_vals))) - 1)] * 1000.0) if latency_vals else float("nan"),
            "link_quality_mean": float(1.0 - statistics.mean(loss_vals)) if loss_vals else float("nan"),
            # Safety
            "oob_triggered": bool(base_metrics.get("terminated_by_oob", False)),
            "energy_triggered": bool(base_metrics.get("terminated_by_energy_depleted", False)),
            "timeout_triggered": bool(termination_reason == "time_limit"),
            "final_energy": float(base_metrics.get("remaining_energy", float("nan"))),
            "min_return_energy_margin": float(min_energy_margin) if min_energy_margin != float("inf") else float("nan"),
            # Solver / planner stats
            "planning_count": int(planner_stats["planning_count"]),
            "total_solve_time_s": float(planner_stats["total_solve_time_s"]),
            "solve_time_max_s": float(planner_stats["solve_time_max_s"]),
            "slow_solve_time_mean_ms": float(solve_mean),
            "slow_solve_time_p95_ms": float(solve_p95),
            "slow_solve_time_max_ms": float(solve_max),
            "solver_infeasible_count": int(planner_stats["infeasible_count"]),
            "solver_fallback_count": int(planner_stats["fallback_count"]),
            "solver_timeout_count": 0,
            "solver_failure_count": 0,
            "mip_gap_mean": float("nan"),
            "mip_gap_max": float("nan"),
            # Coordination
            "replan_total": int(planner_stats["planning_count"]),
            "replan_periodic": int(planner_stats["planning_count"]),
            "replan_event": 0,
            "event_replan_ratio": 0.0,
            "feedback_total": int(planner_stats["planning_count"]),
            "feedback_periodic": int(planner_stats["planning_count"]),
            "feedback_event": 0,
            "target_switch_count": int(goal_switch_count),
            "mode_switch_count": 0,
            # Computation
            "fast_time_mean_ms": float("nan"),
            "fast_time_p95_ms": float("nan"),
            "fast_time_p99_ms": float("nan"),
            "fast_time_max_ms": float("nan"),
            "slow_time_mean_ms": float(solve_mean),
            "slow_time_p95_ms": float(solve_p95),
            "slow_time_p99_ms": float("nan"),
            "slow_time_max_ms": float(solve_max),
            "episode_wall_time_s": float(t_ep_wall),
            # Backward compat
            "goal_switch_count": int(goal_switch_count),
        }
        print(format_paper1_episode_line(ctx=run_ctx, episode=ep, metrics=row), flush=True)
        if metrics_f is not None:
            metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_f.flush()

    if metrics_f is not None:
        metrics_f.close()
    if traj_f is not None:
        traj_f.close()
    if diag_f is not None:
        diag_f.close()
    if events_f is not None:
        events_f.close()


# =========================================================================
# Helpers
# =========================================================================

def _build_planner_input(
    env, baseline_type: str, sim_cfg, dual_link, args
) -> BaselinePlannerInput:
    return BaselinePlannerInput(
        current_position=(float(env.pos_ne[0]), float(env.pos_ne[1])),
        gcs_position=(float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1])),
        uncovered_poi_ids=_uncovered_ids(env),
        remaining_energy=float(env.remaining_energy),
        simulation_step=int(env.t),
        step_hz=int(max(1, env.cfg.step_hz)),
        poi_dwell_s=float(env.cfg.poi_dwell_s),
        energy_per_meter=float(env.cfg.energy_per_meter),
        energy_hover_per_s=float(env.cfg.energy_hover_per_s),
        energy_safe_margin=float(env.cfg.energy_safe_margin),
        control_max_loss_p=float(dual_link.control_max_loss_p),
        data_max_loss_p=float(dual_link.data_max_loss_p),
        data_max_return_time_s=float(dual_link.data_max_return_time_s),
        window_k=16,  # document recommendation (same as FDLC base window K)
        prefetch_k_multiplier=4,
        horizon_h=6,
        path_samples=9,
        _env=env,
        solver_time_limit_s=2.0,
    )


def _poi_position(env, poi_id: int):
    for poi in env.pois:
        if int(poi.poi_id) == int(poi_id):
            return (float(poi.pos_ne[0]), float(poi.pos_ne[1]))
    return (float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1]))


if __name__ == "__main__":
    main()
