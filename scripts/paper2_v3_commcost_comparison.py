#!/usr/bin/env python3
"""V3: Communication-cost-aware comparison.

Tests D-EPA-RHP's VoI feedback advantage: when communication has an energy cost,
methods that always send feedback waste energy while D-EPA-RHP selectively gates.
"""
from __future__ import annotations
import json, math, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper2.runner.run import run_experiment
import numpy as np

def compare_with_comm_cost():
    """Run D-EPA-RHP vs baselines with communication cost."""
    cfg = apply_experiment_presets(load_resolved_config(
        'configs/experiments/paper2/c3_g2_m2_d_epa_rhp.yaml'))

    # Communication cost levels (Joules per feedback transmission)
    # A typical feedback message might be ~500 bytes. At 1 Mbps and 400W flight power,
    # that's about 500*8/1e6 * 400 = 1.6 J. We use 0 (baseline), 1000, 5000, 20000 J.
    cost_levels = [0, 1000, 5000, 20000]

    methods = ['d_epa_rhp', 'event_dual_rhp', 'greedy_distance', 'epa_rhp_centralized']
    results = {}

    for cost_j in cost_levels:
        print(f"\n{'='*60}")
        print(f"Communication cost: {cost_j} J/feedback")
        print(f"{'='*60}")
        results[cost_j] = {}

        for method in methods:
            # Inject comm cost into config
            if 'paper2' not in cfg:
                cfg['paper2'] = {}
            cfg['paper2']['comm_cost_per_feedback_j'] = float(cost_j)
            cfg['paper2']['enable_comm_cost'] = bool(cost_j > 0)

            ne = 5 if method in ('centralized_rhp',) else 10
            print(f"  {method}...")
            agg, eps = run_experiment(config=cfg, experiment_key=method,
                                      n_episodes=ne, seed=42)
            results[cost_j][method] = {
                'R_task': agg['R_task_mean'],
                'R_cov': agg['R_cov_mean'],
                'R_fail': agg['R_fail_given_cov_mean'],
                'n_feedback': agg.get('n_feedback_mean', 0),
                'energy_remaining': agg.get('energy_remaining_ratio_mean', 0),
            }
            print(f"    R_task={agg['R_task_mean']:.3f}  n_feedback={agg.get('n_feedback_mean',0):.0f}  energy={agg.get('energy_remaining_ratio_mean',0):.3f}")

    # Print summary
    print(f"\n\n{'='*80}")
    print("COMMUNICATION COST COMPARISON")
    print(f"{'='*80}")
    for cost_j in cost_levels:
        print(f"\n--- Comm cost = {cost_j} J/feedback ---")
        print(f"{'Method':<25s} {'R_task':>8s} {'R_cov':>8s} {'n_fb':>8s} {'energy':>8s}")
        print('-' * 60)
        r = results[cost_j]
        sorted_m = sorted(r.items(), key=lambda x: x[1]['R_task'], reverse=True)
        for method, m in sorted_m:
            print(f"{method:<25s} {m['R_task']:8.3f} {m['R_cov']:8.3f} {m['n_feedback']:8.0f} {m['energy']:8.3f}")

    # D-EPA-RHP advantage analysis
    print(f"\n\n--- D-EPA-RHP Relative Performance vs Comm Cost ---")
    print(f"{'Cost':>8s} {'D-EPA':>8s} {'Best Baseline':>14s} {'Gap':>8s} {'Feedback Saved':>15s}")
    print('-' * 60)
    for cost_j in cost_levels:
        r = results[cost_j]
        depa = r['d_epa_rhp']['R_task']
        baselines = {k: v['R_task'] for k, v in r.items() if k != 'd_epa_rhp'}
        best_name = max(baselines, key=baselines.get)
        best_val = baselines[best_name]
        gap = best_val - depa
        # Feedback savings: how many fewer feedbacks D-EPA sends vs best baseline
        depa_fb = r['d_epa_rhp']['n_feedback']
        best_fb = max(r[m]['n_feedback'] for m in baselines)
        fb_saved = best_fb - depa_fb
        print(f"{cost_j:8.0f} {depa:8.3f} {best_name:<14s} {best_val:8.3f} {gap:8.3f} {fb_saved:15.0f}")

    return results

if __name__ == "__main__":
    compare_with_comm_cost()
