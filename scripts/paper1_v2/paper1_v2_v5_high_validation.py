"""
Paper: paper1_v2
Purpose: V5 FSM-path-planning validation — compare D-EPA-RHP V5 vs baselines on C3-High G2-M2.
Inputs: configs/experiments/paper2/c3_high_m2_d_epa_rhp.yaml
Outputs: results_v2/v5_validation/
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uavlab.common.config import load_resolved_config
from uavlab.paper2.runner.run import run_experiment

def main():
    config_path = "configs/experiments/paper2/c3_high_m2_d_epa_rhp.yaml"
    output_dir = Path("results_v2/v5_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("greedy_distance",     "Greedy-Distance"),
        ("epa_rhp_centralized", "EPA-Centralized"),
        ("d_epa_rhp",           "D-EPA-RHP V5"),
    ]

    n_episodes = 10
    seed = 42

    print("=" * 72)
    print(f"  V5 FSM Integration Validation — C3-High G2-M2")
    print(f"  {n_episodes} episodes × seed={seed}")
    print("=" * 72)

    # Load config once
    config = load_resolved_config(config_path)

    all_results = {}
    for exp_key, label in experiments:
        print(f"\n--- {label} ({exp_key}) ---")
        t0 = time.time()
        agg, ep_metrics = run_experiment(
            config=config,
            experiment_key=exp_key,
            n_episodes=n_episodes,
            seed=seed,
            verbose=True,
        )
        elapsed = time.time() - t0

        all_results[exp_key] = {
            "label": label,
            "aggregate": {k: float(v) for k, v in agg.items()},
            "wall_time_s": elapsed,
        }

    # ── Comparison table ──
    print("\n" + "=" * 72)
    print("  RESULTS: C3-High G2-M2 Comparison")
    print("=" * 72)
    header = f"{'Method':<25s} {'R_task':>8s} {'R_cov':>8s} {'R_fail|cov':>10s} {'R_home':>8s} {'R_oob':>6s}"
    print(header)
    print("-" * 72)
    for exp_key, label in experiments:
        a = all_results[exp_key]["aggregate"]
        print(f"{label:<25s} {a['R_task_mean']:8.3f} {a['R_cov_mean']:8.3f} "
              f"{a['R_fail_given_cov_mean']:10.4f} {a['R_home_mean']:8.3f} {a['R_oob_mean']:6.3f}")
    print("=" * 72)

    # ── V5 vs EPA-Centralized gap ──
    v5_r_task = all_results["d_epa_rhp"]["aggregate"]["R_task_mean"]
    epa_r_task = all_results["epa_rhp_centralized"]["aggregate"]["R_task_mean"]
    gap = v5_r_task - epa_r_task
    print(f"\nV5 R_task: {v5_r_task:.3f}  |  EPA-Centralized R_task: {epa_r_task:.3f}  |  Gap: {gap:+.3f}")
    if v5_r_task >= 0.30:
        print("✓ V5 target achieved (R_task ≥ 0.30)")
    else:
        print(f"✗ V5 target not met (need R_task ≥ 0.30, got {v5_r_task:.3f})")

    # ── Save results ──
    out_path = output_dir / "v5_high_validation.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": config_path,
            "n_episodes": n_episodes,
            "seed": seed,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
