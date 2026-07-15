#!/usr/bin/env python3
"""Analyze Paper 2 C3 experiment results and generate report."""
from __future__ import annotations
import json, sys, os
from pathlib import Path
import numpy as np

def analyze_results(results_file: str):
    with open(results_file) as f:
        results = json.load(f)

    print("=" * 70)
    print("Paper 2: D-EPA-RHP C3-G2-M2 Evaluation")
    print("=" * 70)

    # Sort by R_task descending
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['R_task_mean'], reverse=True)

    print(f"\n{'Method':<25s} {'R_task':>8s} {'R_cov':>8s} {'R_fail|cov':>10s} {'R_home':>8s} {'n_replan':>8s}")
    print("-" * 73)
    for k, v in sorted_methods:
        print(f"{k:<25s} {v['R_task_mean']:8.3f} {v['R_cov_mean']:8.3f} "
              f"{v['R_fail_given_cov_mean']:10.3f} {v['R_home_mean']:8.3f} "
              f"{v['n_replan_mean']:8.0f}")

    # Best baseline (excluding d_epa_rhp)
    baselines = {k: v for k, v in results.items() if k != 'd_epa_rhp'}
    best_baseline = max(baselines.items(), key=lambda x: x[1]['R_task_mean'])
    d_epa = results.get('d_epa_rhp', {})

    print(f"\n--- Key Findings ---")
    print(f"Best baseline: {best_baseline[0]} (R_task={best_baseline[1]['R_task_mean']:.3f})")
    if d_epa:
        print(f"D-EPA-RHP: R_task={d_epa['R_task_mean']:.3f}")
        gap = best_baseline[1]['R_task_mean'] - d_epa['R_task_mean']
        print(f"Performance gap: {gap:.3f} ({gap/best_baseline[1]['R_task_mean']*100:.1f}% below best baseline)")

    # Ranking
    print(f"\n--- Performance Ranking ---")
    for i, (k, v) in enumerate(sorted_methods):
        marker = " <-- PROPOSED" if k == 'd_epa_rhp' else ""
        print(f"  {i+1}. {k}: R_task={v['R_task_mean']:.3f}{marker}")

    # Component analysis
    print(f"\n--- Component Contribution Analysis ---")
    # EPA-Centralized has prob model but no UAV autonomy or VoI
    # Event-Dual-RHP has CP-SAT + UAV autonomy + event feedback, no prob model
    # D-EPA-RHP has prob model + UAV autonomy + VoI feedback
    epa = results.get('epa_rhp_centralized', {})
    event = results.get('event_dual_rhp', {})
    greedy = results.get('greedy_distance', {})

    if all([epa, event, greedy, d_epa]):
        print(f"Greedy-Distance (no intelligence):          R_task={greedy['R_task_mean']:.3f}")
        print(f"EPA-Centralized (prob model only):           R_task={epa['R_task_mean']:.3f}")
        print(f"Event-Dual-RHP (CP-SAT + event feedback):    R_task={event['R_task_mean']:.3f}")
        print(f"D-EPA-RHP (prob model + VoI + UAV autonomy): R_task={d_epa['R_task_mean']:.3f}")
        print()
        print(f"Prob model contribution:  {epa['R_task_mean'] - greedy['R_task_mean']:+.3f} (vs Greedy)")
        print(f"Event feedback contribution: {event['R_task_mean'] - greedy['R_task_mean']:+.3f} (vs Greedy)")
        print(f"D-EPA-RHP net effect:      {d_epa['R_task_mean'] - greedy['R_task_mean']:+.3f} (vs Greedy)")

    # Root causes
    print(f"\n--- Root Cause Analysis ---")
    print("""
1. Effective-only advancement stalls UAV at each POI waiting for data return.
   Under C3 (loss_p fluctuates around data threshold 0.20), data return is
   intermittent. The UAV hovers for extended periods waiting for brief link
   quality windows, wasting flight time.

2. Action-conditional probability evaluation (Paper2FastLoop) produces overly
   conservative behavior. The utility function's link improvement bonus and
   backlog reduction rewards create mode oscillations that the simpler FSM
   (Paper1FastLoop) avoids.

3. VoI feedback provides marginal benefit under C3 because GCS-UAV probability
   discrepancy is driven by random link jitter rather than systematic model
   mismatch. The adaptive schedule cannot distinguish jitter noise from
   structural discrepancy.

4. The simplified periodic-replan approach (EPA-Centralized) outperforms all
   complex mechanisms because it makes aggressive decisions (always move to
   next goal) and lets data return happen in background during transit.
""")

if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else 'results/paper2/c3_g2_m2_comparison/results.json'
    analyze_results(results_file)
