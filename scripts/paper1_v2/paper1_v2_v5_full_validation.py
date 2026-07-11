"""
Paper: paper1_v2
Purpose: V5 full validation — C3-High + C3-Medium vs baselines, multi-seed.
Inputs: configs/experiments/paper2/
Outputs: results_v2/v5_validation/
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uavlab.common.config import load_resolved_config
from uavlab.paper2.runner.run import run_experiment


def run_comparison(config_path: str, seeds: list, n_episodes: int, label: str):
    config = load_resolved_config(config_path)
    experiments = [
        ("greedy_distance",     "Greedy-Distance"),
        ("epa_rhp_centralized", "EPA-Centralized"),
        ("d_epa_rhp",           "D-EPA-RHP V5"),
    ]

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  {len(seeds)} seeds × {n_episodes} episodes per seed")
    print(f"{'='*72}")

    all_results = {}
    for exp_key, exp_label in experiments:
        seed_means = []
        for seed in seeds:
            t0 = time.time()
            agg, _ = run_experiment(
                config=config, experiment_key=exp_key,
                n_episodes=n_episodes, seed=seed, verbose=False,
            )
            elapsed = time.time() - t0
            seed_means.append({
                "seed": seed,
                "R_task": agg["R_task_mean"],
                "R_cov": agg["R_cov_mean"],
                "R_fail_given_cov": agg.get("R_fail_given_cov_mean", 0),
                "wall_s": elapsed,
            })
            print(f"  {exp_label:<22s} seed={seed}  R_task={agg['R_task_mean']:.3f}  "
                  f"R_cov={agg['R_cov_mean']:.3f}  R_fail|cov={agg.get('R_fail_given_cov_mean', 0):.3f}  "
                  f"({elapsed:.0f}s)")

        rtasks = [s["R_task"] for s in seed_means]
        rcovs = [s["R_cov"] for s in seed_means]
        rfails = [s["R_fail_given_cov"] for s in seed_means]
        import statistics
        all_results[exp_key] = {
            "label": exp_label,
            "R_task_mean": statistics.mean(rtasks),
            "R_task_std": statistics.stdev(rtasks) if len(rtasks) > 1 else 0.0,
            "R_cov_mean": statistics.mean(rcovs),
            "R_cov_std": statistics.stdev(rcovs) if len(rcovs) > 1 else 0.0,
            "R_fail_given_cov_mean": statistics.mean(rfails),
            "R_fail_given_cov_std": statistics.stdev(rfails) if len(rfails) > 1 else 0.0,
            "seeds": seed_means,
        }

    # Summary
    print(f"\n  --- {label} Summary ---")
    print(f"  {'Method':<25s} {'R_task':>12s} {'R_cov':>12s} {'R_fail|cov':>12s}")
    print(f"  {'-'*60}")
    for exp_key, exp_label in experiments:
        r = all_results[exp_key]
        print(f"  {exp_label:<25s} {r['R_task_mean']:>6.3f}±{r['R_task_std']:.3f}  "
              f"{r['R_cov_mean']:>6.3f}±{r['R_cov_std']:.3f}  "
              f"{r['R_fail_given_cov_mean']:>6.3f}±{r['R_fail_given_cov_std']:.3f}")

    v5_r = all_results["d_epa_rhp"]["R_task_mean"]
    epa_r = all_results["epa_rhp_centralized"]["R_task_mean"]
    gap = v5_r - epa_r
    print(f"  V5 vs EPA-Centralized gap: {gap:+.3f}")
    if gap >= -0.03:
        print(f"  ✓ V5 matches or exceeds EPA-Centralized")
    elif v5_r >= 0.30:
        print(f"  ✓ V5 target met (R_task={v5_r:.3f} ≥ 0.30)")
    else:
        print(f"  ✗ V5 target not met")

    return all_results


def main():
    seeds_high = [42, 43, 44]
    seeds_medium = [42, 43]

    results = {}

    # C3-High
    results["c3_high"] = run_comparison(
        "configs/experiments/paper2/c3_high_m2_d_epa_rhp.yaml",
        seeds=seeds_high, n_episodes=3,
        label="C3-High G2-M2 (3 seeds × 3 eps)"
    )

    # C3-Medium
    results["c3_medium"] = run_comparison(
        "configs/experiments/paper2/c3_medium_m2_d_epa_rhp.yaml",
        seeds=seeds_medium, n_episodes=3,
        label="C3-Medium G2-M2 (2 seeds × 3 eps)"
    )

    # Save
    output_dir = Path("results_v2/v5_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, v in results.items():
        out[k] = {kk: {kkk: vvv for kkk, vvv in vv.items() if kkk != "seeds"} for kk, vv in v.items()}
    with open(output_dir / "v5_full_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {output_dir / 'v5_full_validation.json'}")


if __name__ == "__main__":
    main()
