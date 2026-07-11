"""
Paper: paper1_v2
Purpose: Fill missing C3-High + C3-Severe data for full baseline comparison.
Inputs: configs/experiments/paper2/c3_{high,severe}_m2_d_epa_rhp.yaml
Outputs: stdout table
"""
from __future__ import annotations
import json, sys, statistics, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uavlab.common.config import load_resolved_config
from uavlab.paper2.runner.run import run_experiment

SEEDS = [42, 43, 44]
N_EPS = 3

# Methods to run for each missing level
# High: need centralized_rhp, periodic_dual_rhp, event_dual_rhp
# Severe: need all 6
HIGH_NEED = ["centralized_rhp", "periodic_dual_rhp", "event_dual_rhp"]
SEVERE_NEED = ["greedy_distance", "centralized_rhp", "periodic_dual_rhp", "event_dual_rhp", "epa_rhp_centralized", "d_epa_rhp"]

LABELS = {
    "greedy_distance": "Greedy-Distance",
    "centralized_rhp": "Centralized-RHP (CDSL)",
    "periodic_dual_rhp": "Periodic-Dual-RHP",
    "event_dual_rhp": "Event-Dual-RHP (FDLC)",
    "epa_rhp_centralized": "EPA-Centralized",
    "d_epa_rhp": "D-EPA-RHP V5",
}

def run_level(level, methods):
    config_path = f"configs/experiments/paper2/c3_{level}_m2_d_epa_rhp.yaml"
    config = load_resolved_config(config_path)
    print(f"\nC3-{level.upper()}:")
    results = {}
    for exp_key in methods:
        seed_metrics = []
        for seed in SEEDS:
            agg, _ = run_experiment(config=config, experiment_key=exp_key, n_episodes=N_EPS, seed=seed, verbose=False)
            seed_metrics.append({"R_task": float(agg["R_task_mean"]), "R_cov": float(agg["R_cov_mean"]), "R_fail_given_cov": float(agg.get("R_fail_given_cov_mean", 0))})
        rtasks = [m["R_task"] for m in seed_metrics]
        rcovs = [m["R_cov"] for m in seed_metrics]
        rfails = [m["R_fail_given_cov"] for m in seed_metrics]
        results[exp_key] = {
            "R_task_mean": statistics.mean(rtasks), "R_task_std": statistics.stdev(rtasks) if len(rtasks)>1 else 0.0,
            "R_cov_mean": statistics.mean(rcovs), "R_fail_given_cov_mean": statistics.mean(rfails),
        }
        print(f"  {LABELS[exp_key]:<25s} R_task={results[exp_key]['R_task_mean']:.3f}±{results[exp_key]['R_task_std']:.3f}  R_cov={results[exp_key]['R_cov_mean']:.3f}  R_fail|cov={results[exp_key]['R_fail_given_cov_mean']:.3f}")
    return results

results_high_extra = run_level("high", HIGH_NEED)
results_severe = run_level("severe", SEVERE_NEED)

# Save
out = {"c3_high_extra": {k: v for k,v in results_high_extra.items()}, "c3_severe": {k: v for k,v in results_severe.items()}}
Path("results_v2/v5_validation").mkdir(parents=True, exist_ok=True)
with open("results_v2/v5_validation/v5_sweep_remainder.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved.")
