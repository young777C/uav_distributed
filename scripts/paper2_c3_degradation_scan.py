#!/usr/bin/env python3
"""C3 degradation level scan: D-EPA-RHP V2 vs baselines across low/medium/high/severe."""
from __future__ import annotations
import json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper2.runner.run import run_experiment
import numpy as np

LEVELS = ['low', 'medium', 'high', 'severe']
METHODS = ['greedy_distance', 'event_dual_rhp', 'epa_rhp_centralized', 'd_epa_rhp']

def run_scan():
    all_results = {}

    for level in LEVELS:
        config_path = f'configs/experiments/paper2/c3_{level}_m2_d_epa_rhp.yaml'
        print(f"\n{'='*70}")
        print(f"C3-{level.upper()} G2-M2  |  seed=42  episodes=3")
        print(f"{'='*70}")

        cfg = apply_experiment_presets(load_resolved_config(config_path))
        all_results[level] = {}

        for method in METHODS:
            ne = 5 if method in ('centralized_rhp',) else 3
            agg, eps = run_experiment(config=cfg, experiment_key=method,
                                      n_episodes=ne, seed=42)
            all_results[level][method] = {
                'R_task': round(agg['R_task_mean'], 4),
                'R_task_std': round(agg['R_task_std'], 4),
                'R_cov': round(agg['R_cov_mean'], 4),
                'R_fail': round(agg['R_fail_given_cov_mean'], 4),
                'R_oob': round(agg['R_oob_mean'], 4),
                'n_replan': round(agg.get('n_replan_mean', 0), 1),
                'n_feedback': round(agg.get('n_feedback_mean', 0), 1),
            }
            print(f"  {method:<25s} R_task={agg['R_task_mean']:.3f}±{agg['R_task_std']:.3f}  "
                  f"R_cov={agg['R_cov_mean']:.3f}  R_fail={agg['R_fail_given_cov_mean']:.3f}  "
                  f"R_oob={agg['R_oob_mean']:.3f}")

    # ── Summary table ──
    print(f"\n\n{'='*90}")
    print("C3 DEGRADATION SCAN — SUMMARY")
    print(f"{'='*90}")

    for level in LEVELS:
        print(f"\n── C3-{level.upper()} ──")
        print(f"{'Method':<25s} {'R_task':>8s} {'R_cov':>8s} {'R_fail':>8s} {'R_oob':>6s}")
        print('-' * 60)
        r = all_results[level]
        sorted_m = sorted(r.items(), key=lambda x: x[1]['R_task'], reverse=True)
        for method, m in sorted_m:
            marker = ' <--' if method == 'd_epa_rhp' else ''
            print(f"{method:<25s} {m['R_task']:8.3f} {m['R_cov']:8.3f} {m['R_fail']:8.3f} {m['R_oob']:6.3f}{marker}")

    # ── Trend analysis ──
    print(f"\n\n── PERFORMANCE vs DEGRADATION LEVEL ──")
    print(f"{'Level':<8s}", end='')
    for method in METHODS:
        print(f" {method:<22s}", end='')
    print()
    print('-' * (8 + 23 * len(METHODS)))
    for level in LEVELS:
        print(f"{level:<8s}", end='')
        for method in METHODS:
            rt = all_results[level][method]['R_task']
            print(f" {rt:<22.3f}", end='')
        print()

    # ── Relative to best baseline ──
    print(f"\n\n── D-EPA-RHP V2 vs BEST BASELINE ──")
    print(f"{'Level':<8s} {'D-EPA':>8s} {'Best':>8s} {'Gap':>8s} {'D-EPA/Best':>10s}")
    print('-' * 50)
    for level in LEVELS:
        r = all_results[level]
        depa = r['d_epa_rhp']['R_task']
        baselines = {k: v['R_task'] for k, v in r.items() if k != 'd_epa_rhp'}
        best_val = max(baselines.values())
        gap = best_val - depa
        ratio = depa / max(best_val, 1e-12)
        print(f"{level:<8s} {depa:8.3f} {best_val:8.3f} {gap:8.3f} {ratio:10.3f}")

    # Save results
    os.makedirs('results/paper2/c3_degradation_scan', exist_ok=True)
    with open('results/paper2/c3_degradation_scan/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results/paper2/c3_degradation_scan/results.json")

    return all_results

if __name__ == "__main__":
    run_scan()
