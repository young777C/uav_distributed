"""
Paper: paper1_v2
Purpose: Quick P0 fix validation — V5 on 3 M3 collapse scenes only.
Inputs: configs/experiments/paper1/cases/c3_{medium,high,severe}_m3.yaml
Outputs: results_v2/v5_validation/p0_quick_validate.json
"""
from __future__ import annotations
import json, sys, statistics, time, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper2.runner.run import run_experiment

SCENES = [
    ("medium", "m3"),
    ("high", "m3"),
    ("severe", "m3"),
]
SEEDS = [42, 43, 44]
N_EPS = 3

PAPER2_PRESET = {
    "experiment": {
        "paper1": {"struct": "fdlc", "modeling": "comm_energy_aware_decision", "coupling": "full_coupling"},
        "paper2": {"experiment": "d_epa_rhp"},
        "n_episodes": N_EPS,
    },
    "paper2": {"feedback_threshold": 0.15, "adaptive_schedule": True, "lambda_d": 1.0, "lambda_m": 0.5},
}

# Baseline reference values from V5 16-scene sweep (pre-fix)
BASELINE_REF = {
    "c3_medium_m3": 0.200,
    "c3_high_m3":   0.120,
    "c3_severe_m3": 0.110,
}

def build_config(level, conflict):
    case_path = f"configs/experiments/paper1/cases/c3_{level}_{conflict}.yaml"
    raw = load_resolved_config(case_path)
    merged = {**raw, **copy.deepcopy(PAPER2_PRESET)}
    if "experiment" in raw and "experiment" in merged:
        merged["experiment"] = {**raw.get("experiment", {}), **merged["experiment"]}
    return apply_experiment_presets(merged)

def main():
    print("=" * 90)
    print("  P0 Fix Validation: V5 on 3 M3 Collapse Scenes")
    print(f"  {len(SEEDS)} seeds × {N_EPS} eps  |  Phase A RECOVER + M3 Candidate Screening")
    print("=" * 90)

    all_data = {}
    for level, conflict in SCENES:
        scene = f"c3_{level}_{conflict}"
        baseline = BASELINE_REF.get(scene, 0.0)
        print(f"\n--- {scene} (baseline R_task ≈ {baseline:.3f}) ---")
        try:
            config = build_config(level, conflict)
        except Exception as e:
            print(f"  SKIP: config build failed: {e}")
            continue

        seed_rts, seed_rcs, seed_rfs = [], [], []
        for seed in SEEDS:
            t0 = time.time()
            agg, _ = run_experiment(
                config=config, experiment_key="d_epa_rhp",
                n_episodes=N_EPS, seed=seed, verbose=False,
            )
            elapsed = time.time() - t0
            rt = agg["R_task_mean"]
            rc = agg["R_cov_mean"]
            rf = agg.get("R_fail_given_cov_mean", 0)
            seed_rts.append(rt)
            seed_rcs.append(rc)
            seed_rfs.append(rf)
            delta = rt - baseline
            print(f"  seed={seed}  R_task={rt:.3f}  R_cov={rc:.3f}  "
                  f"R_fail|cov={rf:.3f}  Δ={delta:+.3f}  {elapsed:.1f}s")

        avg_rt = statistics.mean(seed_rts)
        avg_rc = statistics.mean(seed_rcs)
        avg_rf = statistics.mean(seed_rfs)
        std_rt = statistics.stdev(seed_rts) if len(seed_rts) > 1 else 0.0
        delta_avg = avg_rt - baseline

        all_data[scene] = {
            "R_task_mean": avg_rt, "R_task_std": std_rt,
            "R_cov_mean": avg_rc, "R_fail_given_cov_mean": avg_rf,
            "baseline": baseline, "delta": delta_avg,
        }
        print(f"  ── AVG: R_task={avg_rt:.3f}±{std_rt:.3f}  "
              f"R_cov={avg_rc:.3f}  R_fail|cov={avg_rf:.3f}  "
              f"Δ={delta_avg:+.3f} ({delta_avg/baseline*100:+.1f}%)" if baseline > 0 else "")

    # Summary
    print("\n" + "=" * 90)
    print("  SUMMARY: P0 Fix vs Baseline (V5 pre-fix)")
    print("=" * 90)
    print(f"  {'Scene':<20s} {'Baseline':>8s} {'P0 R_task':>10s} {'Δ':>8s} {'Δ%':>8s}")
    print(f"  {'-'*60}")
    all_rt, all_bl = [], []
    for level, conflict in SCENES:
        scene = f"c3_{level}_{conflict}"
        if scene in all_data:
            d = all_data[scene]
            bl = d["baseline"]
            rt = d["R_task_mean"]
            delta = d["delta"]
            pct = delta / bl * 100 if bl > 0 else 0
            print(f"  {scene:<20s} {bl:8.3f} {rt:10.3f} {delta:+8.3f} {pct:+7.1f}%")
            all_rt.append(rt)
            all_bl.append(bl)

    if all_rt:
        avg_rt_all = statistics.mean(all_rt)
        avg_bl_all = statistics.mean(all_bl)
        delta_all = avg_rt_all - avg_bl_all
        print(f"  {'─'*60}")
        print(f"  {'M3 AVG':<20s} {avg_bl_all:8.3f} {avg_rt_all:10.3f} {delta_all:+8.3f} {delta_all/avg_bl_all*100:+7.1f}%")

    # Save
    out_dir = Path("results_v2/v5_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "p0_quick_validate.json"
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
