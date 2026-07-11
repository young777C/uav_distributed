#!/usr/bin/env python3
"""Quick C3-G2-M2 baseline comparison for Paper 2."""
from __future__ import annotations
import json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper2.runner.run import run_experiment

EXPERIMENTS = [
    "greedy_distance",
    "centralized_rhp",
    "periodic_dual_rhp",
    "event_dual_rhp",
    "epa_rhp_centralized",
    "d_epa_rhp",
]

SEEDS = [42, 123, 456]
CONFIG = "configs/experiments/paper2/c3_g2_m2_d_epa_rhp.yaml"
OUTPUT = "results/paper2/c3_g2_m2_full_comparison.json"

def main():
    cfg = load_resolved_config(CONFIG)
    # Reduce episodes for CP-SAT baselines (expensive)
    n_ep = {"centralized_rhp": 5, "periodic_dual_rhp": 5}.get
    all_results = {}

    for exp_key in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Running: {exp_key}")
        print(f"{'='*60}")
        ne = n_ep(exp_key) if n_ep else 10
        exp_results = []
        for seed in SEEDS:
            print(f"  Seed {seed} ({ne} episodes)...")
            agg, eps = run_experiment(
                config=apply_experiment_presets(cfg),
                experiment_key=exp_key,
                n_episodes=ne,
                seed=seed,
            )
            exp_results.append({"seed": seed, "aggregate": agg, "episodes": eps})
            print(f"    R_task={agg['R_task_mean']:.3f} R_cov={agg['R_cov_mean']:.3f}")
        all_results[exp_key] = exp_results

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT}")

if __name__ == "__main__":
    main()
