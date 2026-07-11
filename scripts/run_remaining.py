#!/usr/bin/env python3
"""Sequential runner for remaining TS experiments. More reliable than multiprocessing."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "results/ts_sensitivity/parallel"
EPISODES = 3

REMAINING = [
    # (case, system_cfg_rel, ts, seeds)
    ("c2_g2_m2", "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml", 80, [1, 2]),
    ("c2_g2_m2", "configs/experiments/paper1/system/ts_sweep_periodic_goal.yaml", 80, [0, 1, 2]),
    ("c2_g2_m2", "configs/experiments/paper1/system/ts_sweep_periodic_goal.yaml", 40, [0, 1, 2]),
    ("c2_g2_m2", "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml", 20, [0, 1, 2]),
]


def is_done(case: str, system_cfg: str, ts: int, seed: int) -> bool:
    system = Path(system_cfg).stem
    metrics = OUT_ROOT / f"{case}__{system}" / f"Ts{ts}" / f"seed{seed}" / "metrics.jsonl"
    if metrics.exists() and metrics.stat().st_size > 0:
        n = sum(1 for _ in metrics.open("r") if _.strip())
        return n >= EPISODES
    return False


def run_one(case: str, system_cfg: str, ts: int, seed: int) -> bool:
    if is_done(case, system_cfg, ts, seed):
        print(f"  SKIP {case} {Path(system_cfg).stem} Ts={ts} seed={seed} (already done)")
        return True

    system = Path(system_cfg).stem
    exp = f"{case}__{system}"
    run_dir = OUT_ROOT / exp / f"Ts{ts}" / f"seed{seed}"
    metrics_path = run_dir / "metrics.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write combined config
    combined_cfg = run_dir / "combined_config.yaml"
    from uavlab.common.config import save_yaml
    payload = {
        "extends": [
            str((PROJECT_ROOT / f"configs/experiments/paper1/cases/{case}.yaml").resolve()),
            str((PROJECT_ROOT / system_cfg).resolve()),
        ],
        "experiment": {"id": exp},
    }
    save_yaml(payload, combined_cfg)

    cmd = [
        sys.executable, "-m", "uavlab.paper1.runner.run_vis",
        "--config", str(combined_cfg),
        "--episodes", str(EPISODES),
        "--seed", str(seed),
        "--slow_interval_steps", str(ts),
        "--run_dir", str(run_dir),
        "--metrics_jsonl", str(metrics_path),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    print(f"  RUN {exp} Ts={ts} seed={seed}...", end=" ", flush=True)
    t0 = time.time()
    try:
        subprocess.check_call(cmd, cwd=str(PROJECT_ROOT), env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elapsed = time.time() - t0
        print(f"OK ({elapsed:.0f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"FAIL (rc={e.returncode}, {elapsed:.0f}s)")
        return False


def main() -> None:
    total = 0
    ok = 0
    fail = 0
    skip = 0

    for case, system_cfg, ts, seeds in REMAINING:
        for seed in seeds:
            total += 1
            if is_done(case, system_cfg, ts, seed):
                skip += 1
                continue
            if run_one(case, system_cfg, ts, seed):
                ok += 1
            else:
                fail += 1

    print(f"\nTotal: {total}, OK: {ok}, Skip: {skip}, Fail: {fail}")


if __name__ == "__main__":
    main()
