#!/usr/bin/env python3
"""Parallel dual time-scale sensitivity experiment runner.

Launches individual runner_vis.py instances in parallel to maximize throughput
on a 4-core machine. Reuses existing Ts=40 data from the struct axis sweep.

Usage:
    PYTHONPATH=src python3 scripts/run_ts_parallel.py [--max-workers 3]

Output:
    results/ts_sensitivity/parallel/
      {case}__{system}/Ts{ts}/seed{seed}/{metrics.jsonl, traj.jsonl, resolved_config.json, ...}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from uavlab.common.config import load_resolved_config, save_yaml
from uavlab.experiments.presets import apply_experiment_presets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = "uavlab.paper1.runner.run_vis"
OUT_ROOT = PROJECT_ROOT / "results/ts_sensitivity/parallel"
EPISODES = 3
SEEDS = [0, 1, 2]

# Existing Ts=40 data (reuse instead of re-running)
EXISTING_TS40 = {
    ("c1_g2_m2", "struct_full_dual_loop_distributed"): [
        PROJECT_ROOT / "artifacts/baseline_before_paper_sweep_20260615_103151/runs_snapshot/sweeps/struct_axis_full_20260606/c1_g2_m2__struct_full_dual_loop_distributed/Ts40",
    ],
    ("c2_g2_m2", "struct_full_dual_loop_distributed"): [
        PROJECT_ROOT / "artifacts/baseline_before_paper_sweep_20260615_103151/runs_snapshot/sweeps/struct_axis_full_20260606/c2_g2_m2__struct_full_dual_loop_distributed/Ts40",
    ],
}

# Command matrix: (case_config, system_config, ts_values_to_run)
COMMANDS: List[Dict[str, Any]] = [
    # Layer 1a: FDLC on C1–G2–M2
    {
        "case": "c1_g2_m2",
        "case_cfg": "configs/experiments/paper1/cases/c1_g2_m2.yaml",
        "system": "struct_full_dual_loop_distributed",
        "system_cfg": "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml",
        "ts_values": [20, 80],   # 40 exists in EXISTING_TS40
    },
    # Layer 1b: FDLC on C2–G2–M2
    {
        "case": "c2_g2_m2",
        "case_cfg": "configs/experiments/paper1/cases/c2_g2_m2.yaml",
        "system": "struct_full_dual_loop_distributed",
        "system_cfg": "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml",
        "ts_values": [20, 80],   # 40 exists
    },
    # Layer 2: Periodic Goal on C2–G2–M2
    {
        "case": "c2_g2_m2",
        "case_cfg": "configs/experiments/paper1/cases/c2_g2_m2.yaml",
        "system": "ts_sweep_periodic_goal",
        "system_cfg": "configs/experiments/paper1/system/ts_sweep_periodic_goal.yaml",
        "ts_values": [20, 40, 80],  # all need running
    },
]


def link_existing_data() -> int:
    """Symlink existing Ts=40 data into the parallel output directory."""
    count = 0
    for (case, system), ts_dirs in EXISTING_TS40.items():
        for ts_dir in ts_dirs:
            ts_val = int(ts_dir.name[2:]) if ts_dir.name.startswith("Ts") else 0
            for seed_dir in ts_dir.glob("seed*"):
                seed_val = int(seed_dir.name[4:]) if seed_dir.name.startswith("seed") else -1
                if seed_val not in SEEDS:
                    continue
                out_dir = OUT_ROOT / f"{case}__{system}" / f"Ts{ts_val}" / f"seed{seed_val}"
                if not out_dir.exists():
                    out_dir.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(str(seed_dir.resolve()), str(out_dir), target_is_directory=True)
                    count += 1
    return count


def run_single(case_cfg: str, system_cfg: str, ts: int, seed: int) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    case_name = Path(case_cfg).stem
    system_name = Path(system_cfg).stem
    exp_name = f"{case_name}__{system_name}"
    run_dir = OUT_ROOT / exp_name / f"Ts{ts}" / f"seed{seed}"
    metrics_path = run_dir / "metrics.jsonl"
    traj_path = run_dir / "traj.jsonl"
    diag_path = run_dir / "diag.jsonl"

    # Skip if already completed
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        n_ep = sum(1 for _ in metrics_path.open("r") if _.strip())
        if n_ep >= EPISODES:
            return {"case": case_name, "system": system_name, "ts": ts, "seed": seed,
                    "status": "SKIP (exists)", "episodes": n_ep}

    run_dir.mkdir(parents=True, exist_ok=True)

    # Write combined config
    combined_cfg = run_dir / "combined_config.yaml"
    from uavlab.common.config import save_yaml
    payload = {
        "extends": [
            str(Path(case_cfg).resolve()),
            str(Path(system_cfg).resolve()),
        ],
        "experiment": {"id": exp_name},
    }
    save_yaml(payload, combined_cfg)

    # Write resolved_config.json
    from uavlab.common.config import load_resolved_config
    from uavlab.experiments.presets import apply_experiment_presets
    cfg = apply_experiment_presets(load_resolved_config(str(combined_cfg)))
    (run_dir / "resolved_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Launch runner
    cmd = [
        "python3", "-m", RUNNER,
        "--config", str(combined_cfg),
        "--episodes", str(EPISODES),
        "--seed", str(seed),
        "--slow_interval_steps", str(ts),
        "--run_dir", str(run_dir),
        "--metrics_jsonl", str(metrics_path),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    t0 = time.time()
    try:
        subprocess.check_call(cmd, cwd=str(PROJECT_ROOT), env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elapsed = time.time() - t0
        return {"case": case_name, "system": system_name, "ts": ts, "seed": seed,
                "status": "OK", "elapsed_s": f"{elapsed:.0f}"}
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        return {"case": case_name, "system": system_name, "ts": ts, "seed": seed,
                "status": f"FAIL (rc={e.returncode})", "elapsed_s": f"{elapsed:.0f}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel dual time-scale sensitivity sweeps.")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="Max parallel runner processes (default 2 for 4-core machine).")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Link existing Ts=40 data
    n_linked = link_existing_data()
    print(f"[INFO] Linked {n_linked} existing Ts=40 run directories.")

    # Build task list
    tasks: List[Dict[str, Any]] = []
    for entry in COMMANDS:
        for ts in entry["ts_values"]:
            for seed in SEEDS:
                tasks.append({
                    "case_cfg": entry["case_cfg"],
                    "system_cfg": entry["system_cfg"],
                    "case": entry["case"],
                    "system": entry["system"],
                    "ts": ts,
                    "seed": seed,
                })

    # Check how many are already done (skip existing)
    pending = []
    for t in tasks:
        exp_name = f"{t['case']}__{t['system']}"
        run_dir = OUT_ROOT / exp_name / f"Ts{t['ts']}" / f"seed{t['seed']}"
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists() and metrics_path.stat().st_size > 0:
            n_ep = sum(1 for _ in metrics_path.open("r") if _.strip())
            if n_ep >= EPISODES:
                continue
        pending.append(t)

    total = len(tasks)
    todo = len(pending)
    print(f"[INFO] Total tasks: {total}, Pending: {todo}, Completed/Skipped: {total - todo}")

    if not pending:
        print("[INFO] All tasks already completed!")
        return

    # Run pending tasks in parallel
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        for t in pending:
            fut = pool.submit(run_single, case_cfg=t["case_cfg"], system_cfg=t["system_cfg"],
                              ts=t["ts"], seed=t["seed"])
            futures[fut] = t

        for fut in as_completed(futures):
            task = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                lbl = f"{task['case']}__{task['system']} Ts={task['ts']} seed={task['seed']}"
                print(f"  [{result['status']}] {lbl}  ({result.get('elapsed_s', '?')}s)")
            except Exception as e:
                lbl = f"{task['case']}__{task['system']} Ts={task['ts']} seed={task['seed']}"
                print(f"  [ERROR] {lbl}: {e}")

    # Summary
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] != "OK" and r["status"] != "SKIP (exists)")
    print(f"\n[SUMMARY] OK: {ok}, Failed: {fail}, Linked existing: {n_linked}")


if __name__ == "__main__":
    main()
