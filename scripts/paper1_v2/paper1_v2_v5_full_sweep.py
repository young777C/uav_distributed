"""
Paper: paper1_v2
Purpose: Full C3 degradation sweep — V5 vs all baselines across Low/Medium/High/Severe.
Inputs: configs/experiments/paper2/c3_{low,medium,high,severe}_m2_d_epa_rhp.yaml
Outputs: results_v2/v5_validation/v5_full_sweep.json
"""
from __future__ import annotations

import json, sys, statistics, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uavlab.common.config import load_resolved_config
from uavlab.paper2.runner.run import run_experiment

DEGRADATION_LEVELS = ["low", "medium", "high", "severe"]
BASELINES = [
    ("greedy_distance",     "Greedy-Distance"),
    ("centralized_rhp",     "Centralized-RHP (CDSL)"),
    ("periodic_dual_rhp",   "Periodic-Dual-RHP"),
    ("event_dual_rhp",      "Event-Dual-RHP (FDLC)"),
    ("epa_rhp_centralized", "EPA-Centralized"),
    ("d_epa_rhp",           "D-EPA-RHP V5"),
]

def main():
    seeds = [42, 43, 44]
    n_episodes = 3
    all_results = {}

    for level in DEGRADATION_LEVELS:
        config_path = f"configs/experiments/paper2/c3_{level}_m2_d_epa_rhp.yaml"
        config = load_resolved_config(config_path)
        print(f"\n{'='*72}")
        print(f"  C3-{level.upper()} G2-M2  ({len(seeds)} seeds × {n_episodes} eps)")
        print(f"{'='*72}")

        level_results = {}
        for exp_key, label in BASELINES:
            seed_metrics = []
            for seed in seeds:
                t0 = time.time()
                agg, _ = run_experiment(
                    config=config, experiment_key=exp_key,
                    n_episodes=n_episodes, seed=seed, verbose=False,
                )
                elapsed = time.time() - t0
                seed_metrics.append({
                    "R_task": float(agg["R_task_mean"]),
                    "R_cov": float(agg["R_cov_mean"]),
                    "R_fail_given_cov": float(agg.get("R_fail_given_cov_mean", 0)),
                    "R_home": float(agg.get("R_home_mean", 0)),
                    "R_oob": float(agg.get("R_oob_mean", 0)),
                    "T_ret_s": float(agg.get("T_ret_s_mean", 0)),
                    "energy_rem": float(agg.get("energy_remaining_ratio_mean", 0)),
                })

            rtasks = [m["R_task"] for m in seed_metrics]
            rcovs = [m["R_cov"] for m in seed_metrics]
            rfails = [m["R_fail_given_cov"] for m in seed_metrics]

            level_results[exp_key] = {
                "label": label,
                "R_task_mean": statistics.mean(rtasks),
                "R_task_std": statistics.stdev(rtasks) if len(rtasks) > 1 else 0.0,
                "R_cov_mean": statistics.mean(rcovs),
                "R_fail_given_cov_mean": statistics.mean(rfails),
                "R_oob_mean": statistics.mean([m["R_oob"] for m in seed_metrics]),
                "T_ret_s_mean": statistics.mean([m["T_ret_s"] for m in seed_metrics]),
                "energy_remaining_ratio_mean": statistics.mean([m["energy_rem"] for m in seed_metrics]),
                "seeds": seed_metrics,
            }
            print(f"  {label:<25s}  R_task={level_results[exp_key]['R_task_mean']:.3f}±"
                  f"{level_results[exp_key]['R_task_std']:.3f}  "
                  f"R_cov={level_results[exp_key]['R_cov_mean']:.3f}  "
                  f"R_fail|cov={level_results[exp_key]['R_fail_given_cov_mean']:.3f}")

        all_results[level] = level_results

    # ── Print comprehensive table ──
    print("\n\n" + "=" * 100)
    print("  C3 DEGRADATION SWEEP — V5 vs ALL BASELINES (G2-M2, 3 seeds × 3 eps)")
    print("=" * 100)

    # Table 1: R_task
    print(f"\n  {'R_task':─^100}")
    header = f"  {'Method':<25s}"
    for level in DEGRADATION_LEVELS:
        header += f" {'C3-'+level.capitalize():>12s}"
    header += f" {'Avg':>8s}"
    print(header)
    print(f"  {'-'*85}")
    for exp_key, label in BASELINES:
        row = f"  {label:<25s}"
        vals = []
        for level in DEGRADATION_LEVELS:
            v = all_results[level][exp_key]["R_task_mean"]
            s = all_results[level][exp_key]["R_task_std"]
            row += f" {v:6.3f}±{s:.3f}  "
            vals.append(v)
        row += f" {statistics.mean(vals):6.3f}"
        print(row)

    # Table 2: R_cov
    print(f"\n  {'R_cov':─^100}")
    header = f"  {'Method':<25s}"
    for level in DEGRADATION_LEVELS:
        header += f" {'C3-'+level.capitalize():>12s}"
    print(header)
    print(f"  {'-'*85}")
    for exp_key, label in BASELINES:
        row = f"  {label:<25s}"
        for level in DEGRADATION_LEVELS:
            v = all_results[level][exp_key]["R_cov_mean"]
            row += f" {v:11.3f}  "
        print(row)

    # Table 3: R_fail|cov
    print(f"\n  {'R_fail|cov':─^100}")
    header = f"  {'Method':<25s}"
    for level in DEGRADATION_LEVELS:
        header += f" {'C3-'+level.capitalize():>12s}"
    print(header)
    print(f"  {'-'*85}")
    for exp_key, label in BASELINES:
        row = f"  {label:<25s}"
        for level in DEGRADATION_LEVELS:
            v = all_results[level][exp_key]["R_fail_given_cov_mean"]
            row += f" {v:11.4f}  "
        print(row)

    # Rankings per level
    print(f"\n  {'Rankings (by R_task)':─^100}")
    header = f"  {'Rank':<6s}"
    for level in DEGRADATION_LEVELS:
        header += f" {'C3-'+level.capitalize():>22s}"
    print(header)
    print(f"  {'-'*100}")
    for rank in range(1, 7):
        row = f"  #{rank:<5d}"
        for level in DEGRADATION_LEVELS:
            sorted_methods = sorted(
                BASELINES, key=lambda x: all_results[level][x[0]]["R_task_mean"], reverse=True
            )
            if rank <= len(sorted_methods):
                exp_key, label = sorted_methods[rank - 1]
                v = all_results[level][exp_key]["R_task_mean"]
                row += f" {label:<16s} ({v:.3f})  "
            else:
                row += f" {'':>22s}"
        print(row)

    # Save
    output_dir = Path("results_v2/v5_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for level in DEGRADATION_LEVELS:
        out[level] = {}
        for exp_key, label in BASELINES:
            r = all_results[level][exp_key]
            out[level][exp_key] = {
                "label": label,
                "R_task_mean": r["R_task_mean"], "R_task_std": r["R_task_std"],
                "R_cov_mean": r["R_cov_mean"],
                "R_fail_given_cov_mean": r["R_fail_given_cov_mean"],
                "R_oob_mean": r["R_oob_mean"],
                "T_ret_s_mean": r["T_ret_s_mean"],
                "energy_remaining_ratio_mean": r["energy_remaining_ratio_mean"],
            }
    with open(output_dir / "v5_full_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {output_dir / 'v5_full_sweep.json'}")

if __name__ == "__main__":
    main()
