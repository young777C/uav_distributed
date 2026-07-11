"""
Paper: paper1_v2
Purpose: Quick 16-scene validation sweep — calibration effect (1 seed × 2 eps).
Inputs: configs/experiments/paper1/cases/c3_{level}_{conflict}.yaml
Outputs: results_v2/v5_validation/calibration_16scene_quick.json
"""
from __future__ import annotations
import json, sys, statistics, time, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uavlab.common.config import load_resolved_config
from uavlab.experiments.presets import apply_experiment_presets
from uavlab.paper2.runner.run import run_experiment

DEGRADATION = ["low", "medium", "high", "severe"]
CONFLICTS   = ["m0", "m1", "m2", "m3"]
BASELINES   = [
    ("greedy_distance",     "Greedy-Distance"),
    ("centralized_rhp",     "Centralized-RHP"),
    ("periodic_dual_rhp",   "Periodic-Dual-RHP"),
    ("event_dual_rhp",      "Event-Dual-RHP"),
    ("epa_rhp_centralized", "EPA-Centralized"),
    ("d_epa_rhp",           "D-EPA-RHP V5"),
]
SEEDS, N_EPS = [42], 2

PAPER2_PRESET = {
    "experiment": {
        "paper1": {"struct": "fdlc", "modeling": "comm_energy_aware_decision", "coupling": "full_coupling"},
        "paper2": {"experiment": "d_epa_rhp"},
        "n_episodes": N_EPS,
    },
    "paper2": {"feedback_threshold": 0.15, "adaptive_schedule": True, "lambda_d": 1.0, "lambda_m": 0.5},
}

def build_config(level, conflict):
    case_path = f"configs/experiments/paper1/cases/c3_{level}_{conflict}.yaml"
    raw = load_resolved_config(case_path)
    merged = {**raw, **copy.deepcopy(PAPER2_PRESET)}
    if "experiment" in raw and "experiment" in merged:
        merged["experiment"] = {**raw.get("experiment", {}), **merged["experiment"]}
    return apply_experiment_presets(merged)

def main():
    print("=" * 110)
    print("  Calibration Validation — 16 C3 Scenes (1 seed × 2 eps)")
    print("=" * 110)

    all_data = {}
    t_start = time.time()
    total_exps = len(DEGRADATION) * len(CONFLICTS) * len(BASELINES)
    n_done = 0

    for level in DEGRADATION:
        for conflict in CONFLICTS:
            scene = f"c3_{level}_{conflict}"
            print(f"\n--- {scene} ---")
            try:
                config = build_config(level, conflict)
            except Exception as e:
                print(f"  SKIP: config build failed: {e}")
                continue

            scene_results = {}
            for exp_key, label in BASELINES:
                t0 = time.time()
                seed = SEEDS[0]
                agg, _ = run_experiment(
                    config=config, experiment_key=exp_key,
                    n_episodes=N_EPS, seed=seed, verbose=False,
                )
                elapsed = time.time() - t0
                rt = agg["R_task_mean"]
                rc = agg["R_cov_mean"]
                rf = agg.get("R_fail_given_cov_mean", 0)
                ro = agg.get("R_oob_mean", 0)
                scene_results[exp_key] = {
                    "label": label, "R_task": rt, "R_cov": rc,
                    "R_fail_given_cov": rf, "R_oob": ro,
                }
                n_done += 1
                pct = n_done / total_exps * 100
                print(f"  [{n_done:3d}/{total_exps} {pct:5.1f}%] {label:<22s} "
                      f"R_task={rt:.3f}  R_cov={rc:.3f}  "
                      f"R_fail|cov={rf:.3f}  {elapsed:.1f}s")
            all_data[scene] = scene_results

    elapsed_total = time.time() - t_start
    print(f"\n\nCompleted {n_done} experiments in {elapsed_total:.0f}s "
          f"({elapsed_total/n_done:.1f}s/exp)")

    # ── R_task matrix ──
    print("\n" + "=" * 130)
    print("  R_task MATRIX")
    print("=" * 130)
    for level in DEGRADATION:
        print(f"\n  ── C3-{level.upper()} ──")
        header = f"  {'Method':<22s}"
        for c in CONFLICTS:
            header += f" M{c[-1]:>1s}      "
        header += f" {'Avg':>7s}"
        print(header)
        print(f"  {'-'*70}")
        for exp_key, label in BASELINES:
            row = f"  {label:<22s}"
            vals = []
            for conflict in CONFLICTS:
                s = f"c3_{level}_{conflict}"
                if s in all_data:
                    v = all_data[s][exp_key]["R_task"]
                    row += f" {v:.3f}   "
                    vals.append(v)
                else:
                    row += f"  N/A    "
            row += f" {statistics.mean(vals):7.3f}" if vals else "     N/A"
            print(row)

    # ── Degradation × Conflict averages ──
    print(f"\n\n  {'DEGRADATION × CONFLICT AVERAGES (R_task)':─^100}")
    header = f"  {'Method':<22s}"
    for level in DEGRADATION:
        header += f" C3-{level[:4]:>4s}  "
    header += "  Avg"
    print(header)
    print(f"  {'-'*70}")
    method_avgs = {}
    for exp_key, label in BASELINES:
        row = f"  {label:<22s}"
        method_vals = []
        for level in DEGRADATION:
            level_vals = []
            for conflict in CONFLICTS:
                s = f"c3_{level}_{conflict}"
                if s in all_data:
                    level_vals.append(all_data[s][exp_key]["R_task"])
            avg = statistics.mean(level_vals) if level_vals else 0
            method_vals.append(avg)
            row += f" {avg:7.3f}"
        overall = statistics.mean(method_vals)
        method_avgs[exp_key] = overall
        row += f" {overall:7.3f}"
        print(row)

    # ── Ranking ──
    print(f"\n  {'OVERALL RANKING':─^60}")
    for rank, (ek, avg) in enumerate(sorted(method_avgs.items(), key=lambda x: x[1], reverse=True), 1):
        label = dict(BASELINES)[ek]
        medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, f'#{rank}')
        print(f"  {medal} {label:<22s} R_task={avg:.3f}")

    # ── OOB check ──
    any_oob = False
    for level in DEGRADATION:
        for conflict in CONFLICTS:
            s = f"c3_{level}_{conflict}"
            if s in all_data:
                for exp_key, label in BASELINES:
                    if all_data[s][exp_key]["R_oob"] > 0:
                        print(f"  ⚠ {s} {label}: OOB={all_data[s][exp_key]['R_oob']:.3f}")
                        any_oob = True
    if not any_oob:
        print(f"\n  ✓ All methods OOB=0 across all 16 scenes")

    # Save
    out_dir = Path("results_v2/v5_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for scene, methods in all_data.items():
        out[scene] = {
            ek: {"label": v["label"], "R_task": v["R_task"],
                 "R_cov": v["R_cov"], "R_fail_given_cov": v["R_fail_given_cov"],
                 "R_oob": v["R_oob"]}
            for ek, v in methods.items()
        }
    out_path = out_dir / "calibration_16scene_quick.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
