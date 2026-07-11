#!/usr/bin/env python3
"""Optimized parallel runner for essential missing TS data.

Prioritizes fast-running configurations (Ts=80) to fill the paper's key comparisons.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "results/ts_sensitivity/parallel"
EPISODES = 3
SEEDS = [0, 1, 2]

# What we still need to run: (case, system_cfg_path, ts_values)
NEEDED = [
    # FDLC C1+C2 at Ts=80 (fast ~3 min each)
    ("c1_g2_m2", "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml", [80]),
    ("c2_g2_m2", "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml", [80]),
    # Periodic Goal at Ts=40 and Ts=80
    ("c2_g2_m2", "configs/experiments/paper1/system/ts_sweep_periodic_goal.yaml", [40, 80]),
    # FDLC C2 at Ts=20 (slower, lower priority, run last)
    ("c2_g2_m2", "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml", [20]),
    # FDLC C1 Ts=20 seed=1 (resume the partial run)
    # Will handle separately since it's a resume
]


def is_done(case: str, system_cfg: str, ts: int, seed: int) -> bool:
    system_name = Path(system_cfg).stem
    exp_name = f"{case}__{system_name}"
    metrics = OUT_ROOT / exp_name / f"Ts{ts}" / f"seed{seed}" / "metrics.jsonl"
    if metrics.exists() and metrics.stat().st_size > 0:
        n = sum(1 for _ in metrics.open("r") if _.strip())
        return n >= EPISODES
    return False


def run_one(case: str, system_cfg: str, ts: int, seed: int) -> Dict[str, Any]:
    system_name = Path(system_cfg).stem
    exp_name = f"{case}__{system_name}"
    run_dir = OUT_ROOT / exp_name / f"Ts{ts}" / f"seed{seed}"
    metrics_path = run_dir / "metrics.jsonl"

    if is_done(case, system_cfg, ts, seed):
        return {"case": case, "system": system_name, "ts": ts, "seed": seed, "status": "SKIP"}

    run_dir.mkdir(parents=True, exist_ok=True)

    # Write combined config
    combined_cfg = run_dir / "combined_config.yaml"
    from uavlab.common.config import save_yaml
    payload = {
        "extends": [
            str(PROJECT_ROOT / f"configs/experiments/paper1/cases/{case}.yaml"),
            str(PROJECT_ROOT / system_cfg),
        ],
        "experiment": {"id": exp_name},
    }
    save_yaml(payload, combined_cfg)

    cmd = [
        "python3", "-m", "uavlab.paper1.runner.run_vis",
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
        return {"case": case, "system": system_name, "ts": ts, "seed": seed,
                "status": "OK", "time": f"{elapsed:.0f}s"}
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        return {"case": case, "system": system_name, "ts": ts, "seed": seed,
                "status": f"FAIL({e.returncode})", "time": f"{elapsed:.0f}s"}


def build_tasks() -> List[Dict[str, Any]]:
    tasks = []
    for case, system_cfg, ts_list in NEEDED:
        for ts in ts_list:
            for seed in SEEDS:
                tasks.append({"case": case, "system_cfg": system_cfg, "ts": ts, "seed": seed})
    # Print pending tasks
    pending = [t for t in tasks if not is_done(t["case"], t["system_cfg"], t["ts"], t["seed"])]
    total = len(tasks)
    print(f"Total: {total}, Pending: {len(pending)}, Done: {total - len(pending)}")
    if pending:
        print("Pending tasks:")
        for t in pending:
            name = Path(t["system_cfg"]).stem
            print(f"  {t['case']} {name} Ts={t['ts']} seed={t['seed']}")
    return tasks


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks()

    pending = [t for t in tasks if not is_done(t["case"], t["system_cfg"], t["ts"], t["seed"])]
    if not pending:
        print("All tasks already complete!")
        return

    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        fut_map = {}
        for t in pending:
            fut = pool.submit(run_one, case=t["case"], system_cfg=t["system_cfg"],
                              ts=t["ts"], seed=t["seed"])
            fut_map[fut] = t

        for fut in as_completed(fut_map):
            t = fut_map[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  [{r['status']}] {t['case']}/{Path(t['system_cfg']).stem} Ts={t['ts']} seed={t['seed']} {r.get('time', '')}")
            except Exception as e:
                print(f"  [ERROR] {t['case']}/{Path(t['system_cfg']).stem} Ts={t['ts']} seed={t['seed']}: {e}")

    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] != "OK" and r["status"] != "SKIP")
    print(f"\nDone: {ok} OK, {fail} Failed")


if __name__ == "__main__":
    main()
