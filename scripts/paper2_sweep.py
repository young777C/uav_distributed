#!/usr/bin/env python3
"""
Paper 2: Batch experiment sweep runner.

Reads a sweep YAML config and runs all experiment × scene combinations.

Usage:
    python scripts/paper2_sweep.py --config configs/experiments/paper2/sweep_c3_main.yaml
    python scripts/paper2_sweep.py --config configs/experiments/paper2/sweep_c3_main.yaml --dry-run
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uavlab.paper2.runner.run import run_experiment


def _build_base_config(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Build base config dict for one scene.

    Supports two scene formats:
    1. New C3 format: ``{scene_cfg, label, comm_case, task_case, conflict}``
       where scene_cfg like "c3_g2_m2" → scene_file = "configs/scenes/g2_m2.yaml"
    2. Old format: ``{comm_case, task_case, conflict}``
       scene_file = "scenes/{comm_case}_{task_case}_{conflict}.yaml"
    """
    scene_cfg_id = scene.get("scene_cfg", "")
    if scene_cfg_id:
        # C3 format: strip comm prefix to get task scene file
        # e.g. "c3_g2_m2" → scene_file = "configs/scenes/g2_cluster_m2.yaml"
        parts = scene_cfg_id.split("_", 1)
        task_part = parts[1] if len(parts) > 1 else "g2_m0"
        if task_part == "g1":
            raise ValueError("G1 uniform scenes are deprecated; use G2 cluster scenes (c3_g2_m0–m3).")
        elif "_" in task_part:
            # g2_m2 → g2_cluster_m2
            subtask, conflict = task_part.split("_", 1)
            scene_file = f"configs/scenes/{subtask}_cluster_{conflict}.yaml"
        else:
            scene_file = f"configs/scenes/{task_part}.yaml"
    else:
        # Old format
        cl = scene.get("conflict", "")
        conflict_suffix = f"_{cl}" if cl and str(cl).strip().lower() not in ("", "none") else ""
        task = scene.get("task_case", "g2").lower()
        if task == "g1":
            raise ValueError("G1 uniform scenes are deprecated; use G2 cluster scenes.")
        else:
            scene_file = f"configs/scenes/{task}{conflict_suffix}.yaml"

    # Determine comm profile from scene_cfg prefix
    comm_case = str(scene.get("comm_case", "C3")).upper().strip()
    is_severe = "C3S" in comm_case or "SEVERE" in comm_case

    comm_cfg = {
        "enable_distance_decay": True,
        "distance_d0_m": 0.0,
        "distance_d1_m": 2500.0,
        "distance_loss_min": 0.05,
        "distance_loss_max": 0.40,
        "use_scene_blackholes": True,
    }
    if is_severe:
        comm_cfg["blackhole_extra_loss"] = 0.30
        comm_cfg["loss_jitter_sigma"] = 0.08
        comm_cfg["data_chunk_bits"] = 4_000_000.0
        comm_cfg["data_link"] = {"max_loss_p": 0.15, "max_return_time_s": 8.0}
    else:
        comm_cfg["blackhole_extra_loss"] = 0.15
        comm_cfg["loss_jitter_sigma"] = 0.08
        comm_cfg["data_chunk_bits"] = 4_000_000.0
        comm_cfg["data_link"] = {"max_loss_p": 0.20, "max_return_time_s": 15.0}

    return {
        "scene_file": scene_file,
        "env": {
            "fast_upload_mode": "policy",
            "v_xy_cruise": 11.0,
            "v_xy_max": 20.0,
            "approach_slowdown_radius_m": 50.0,
            "return_reserve_s": 30.0,
            "step_hz": 5,
            "mission_time_s": 2000.0,
        },
        "comm": comm_cfg,
        "paper1_loops": {
            "return_policy": {"enable_backlog_gates": True, "enable_upload_stuck_recovery": True},
            "slow_policy": "periodic_or_event_replan",
            "allow_waypoint_delta": True,
            "allow_mode_switching": True,
            "semantics": {
                "enable_event_feedback": True,
                "enable_goal_lock": True,
                "enable_fast_mode_switch": True,
                "use_comm_in_fast": True,
                "use_energy_in_fast": True,
                "use_comm_in_slow": True,
                "use_energy_in_slow": True,
            },
            "slow_loop": {
                "slow_interval_steps": 40,
                "replan_trigger_policy": "hybrid",
                "periodic_replan_scope": "repeat",
                "window_k": 12,
                "prefetch_k_multiplier": 3,
                "horizon_h": 8,
                "event_triggers": {
                    "on_goal_spatial_complete": True,
                    "on_goal_effective_complete": True,
                    "on_safety_event": True,
                    "on_energy_low": True,
                    "on_link_drop": True,
                    "on_control_link_lost": True,
                },
            },
        },
        "paper2": {"feedback_threshold": 0.15, "adaptive_schedule": True},
    }


def _build_jobs(sweep_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build list of job dicts from sweep config."""
    jobs = []
    for exp in sweep_cfg.get("experiments", []):
        for scene in sweep_cfg.get("scenes", []):
            # New format with explicit scene_cfg
            if scene.get("scene_cfg"):
                tags = {
                    "experiment": str(exp["key"]),
                    "label": str(exp["label"]),
                    "scene_cfg": str(scene["scene_cfg"]),
                    "comm_case": str(scene.get("comm_case", "C3")),
                    "task_case": str(scene.get("task_case", "G2")),
                    "conflict": str(scene.get("conflict_level", "") or "none"),
                }
                jobs.append(tags)
            else:
                # Old format with conflict_levels list
                for cl in scene.get("conflict_levels", []):
                    tags = {
                        "experiment": str(exp["key"]),
                        "label": str(exp["label"]),
                        "scene_cfg": "",
                        "comm_case": str(scene["comm_case"]),
                        "task_case": str(scene["task_case"]),
                        "conflict": str(cl),
                    }
                    jobs.append(tags)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 2 batch sweep runner")
    parser.add_argument("--config", type=str, required=True, help="Sweep YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without running")
    parser.add_argument("--resume", type=str, default="", help="Results file to resume from")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        sweep_cfg = yaml.safe_load(f)
    sweep_cfg = sweep_cfg.get("sweep", sweep_cfg)

    jobs = _build_jobs(sweep_cfg)
    common = dict(sweep_cfg.get("common", {}))
    n_episodes = int(common.get("n_episodes", 10))
    base_seed = int(common.get("seed", 42))
    output_dir = Path(sweep_cfg.get("output_dir", "results/paper2/sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Paper2Sweep] {len(jobs)} jobs, {n_episodes} episodes each, output: {output_dir}\n")

    if args.dry_run:
        for j in jobs:
            label = j["label"]
            scene = f"{j['comm_case']}-{j['task_case']}-{j['conflict']}"
            print(f"  {label:30s}  {scene}")
        print(f"\nTotal: {len(jobs)} jobs")
        return

    # Load resume state
    completed: List[str] = []
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path, "r") as f:
                completed = [line.strip() for line in f if line.strip()]
            print(f"[Resume] {len(completed)} completed jobs loaded from {resume_path}")

    results: List[Dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log = output_dir / f"run_{timestamp}.json"

    for idx, job in enumerate(jobs):
        job_id = f"{job['experiment']}_{job['comm_case']}_{job['task_case']}_{job['conflict']}"
        label = job["label"]
        scene_tag = f"{job['comm_case']}-{job['task_case']}-{job['conflict']}"

        if job_id in completed:
            print(f"[{idx+1}/{len(jobs)}] SKIP (resumed) {label:30s}  {scene_tag}")
            continue

        print(f"[{idx+1}/{len(jobs)}] {label:30s}  {scene_tag}  ...", end=" ", flush=True)
        t0 = time.time()

        try:
            base = _build_base_config(job)
            agg, _ = run_experiment(
                config=base,
                experiment_key=str(job["experiment"]),
                n_episodes=n_episodes,
                seed=base_seed + idx,
                verbose=False,
            )

            elapsed = time.time() - t0
            r_task = agg.get("R_task_mean", 0.0)
            r_cov = agg.get("R_cov_mean", 0.0)
            print(f"R_task={r_task:.3f}  R_cov={r_cov:.3f}  ({elapsed:.0f}s)")

            record = {
                "job_id": job_id,
                "experiment": job["experiment"],
                "label": label,
                "comm_case": job["comm_case"],
                "task_case": job["task_case"],
                "conflict": job["conflict"],
                "aggregate": agg,
            }
            results.append(record)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.0f}s): {e}")
            import traceback; traceback.print_exc()
            record = {
                "job_id": job_id,
                "experiment": job["experiment"],
                "label": label,
                "comm_case": job["comm_case"],
                "task_case": job["task_case"],
                "conflict": job["conflict"],
                "error": str(e),
            }
            results.append(record)

        # Save incremental results
        with open(run_log, "w") as f:
            json.dump({"timestamp": timestamp, "config": str(args.config), "results": results}, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"Done: {len(results)}/{len(jobs)} jobs")
    success = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    print(f"Success: {len(success)}, Failed: {len(failed)}")
    print(f"Results: {run_log}")

    if success:
        print(f"\nSummary (R_task):")
        for r in success:
            a = r["aggregate"]
            scene = f"{r['comm_case']}-{r['task_case']}-{r['conflict']}"
            print(f"  {r['label']:30s}  {scene:15s}  "
                  f"R_task={a.get('R_task_mean',0):.3f}±{a.get('R_task_std',0):.3f}  "
                  f"R_cov={a.get('R_cov_mean',0):.3f}")


if __name__ == "__main__":
    main()
