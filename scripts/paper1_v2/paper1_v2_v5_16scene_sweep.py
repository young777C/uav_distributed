"""
Paper: paper1_v2
Purpose: Full 16-scene C3 sweep — V5 vs all baselines across all degradation × conflict.
Inputs: configs/experiments/paper1/cases/c3_{level}_{conflict}.yaml
Outputs: results_v2/v5_validation/v5_16scene_sweep.json
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
SEEDS, N_EPS = [42, 43, 44], 3

# Paper2 experiment preset (applied on top of case config)
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
    print("  D-EPA-RHP V5 vs ALL BASELINES — 16 C3 Scenes (4 degradation × 4 conflict)")
    print(f"  {len(SEEDS)} seeds × {N_EPS} eps  |  G2 clustered, 100 POI, 2500×2500m")
    print("=" * 110)

    all_data = {}
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
                seed_rts = []
                for seed in SEEDS:
                    agg, _ = run_experiment(config=config, experiment_key=exp_key,
                                            n_episodes=N_EPS, seed=seed, verbose=False)
                    seed_rts.append((agg["R_task_mean"], agg["R_cov_mean"],
                                     agg.get("R_fail_given_cov_mean", 0), agg.get("R_oob_mean", 0)))
                rt = statistics.mean([s[0] for s in seed_rts])
                rts = statistics.stdev([s[0] for s in seed_rts]) if len(seed_rts) > 1 else 0.0
                rc = statistics.mean([s[1] for s in seed_rts])
                rf = statistics.mean([s[2] for s in seed_rts])
                ro = statistics.mean([s[3] for s in seed_rts])
                scene_results[exp_key] = {"label": label, "R_task": rt, "R_task_std": rts,
                                           "R_cov": rc, "R_fail_given_cov": rf, "R_oob": ro}
                print(f"  {label:<22s} R_task={rt:.3f}±{rts:.3f}  R_cov={rc:.3f}  R_fail|cov={rf:.3f}  R_oob={ro:.3f}")
            all_data[scene] = scene_results

    # ── Summary tables ──
    print("\n\n" + "=" * 130)
    print("  R_task MATRIX: D-EPA-RHP V5 vs ALL BASELINES")
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
                scene = f"c3_{level}_{conflict}"
                if scene in all_data:
                    v = all_data[scene][exp_key]["R_task"]
                    row += f" {v:.3f}   "
                    vals.append(v)
                else:
                    row += f"  N/A    "
            row += f" {statistics.mean(vals):7.3f}" if vals else "     N/A"
            print(row)
        # Ranking row
        rank_row = f"  {'V5 rank':<22s}"
        for conflict in CONFLICTS:
            scene = f"c3_{level}_{conflict}"
            if scene in all_data:
                sorted_methods = sorted(BASELINES, key=lambda x: all_data[scene][x[0]]["R_task"], reverse=True)
                v5_idx = next(i for i, (ek, _) in enumerate(sorted_methods) if ek == "d_epa_rhp")
                v5_val = all_data[scene]["d_epa_rhp"]["R_task"]
                top_val = all_data[scene][sorted_methods[0][0]]["R_task"]
                marker = "🥇" if v5_idx == 0 else f"#{v5_idx+1}"
                rank_row += f" {marker:<7s}"
            else:
                rank_row += f"  N/A    "
        print(rank_row)

    # ── R_cov matrix ──
    print(f"\n\n  {'R_cov MATRIX':─^100}")
    for level in DEGRADATION:
        print(f"\n  ── C3-{level.upper()} ──")
        header = f"  {'Method':<22s}"
        for c in CONFLICTS:
            header += f" M{c[-1]:>1s}      "
        print(header)
        print(f"  {'-'*70}")
        for exp_key, label in BASELINES:
            row = f"  {label:<22s}"
            for conflict in CONFLICTS:
                scene = f"c3_{level}_{conflict}"
                if scene in all_data:
                    row += f" {all_data[scene][exp_key]['R_cov']:.3f}   "
            print(row)

    # ── OOB check ──
    print(f"\n\n  {'OOB RATE':─^100}")
    any_oob = False
    for level in DEGRADATION:
        for conflict in CONFLICTS:
            scene = f"c3_{level}_{conflict}"
            if scene in all_data:
                for exp_key, label in BASELINES:
                    if all_data[scene][exp_key]["R_oob"] > 0:
                        print(f"  ⚠ {scene} {label}: OOB={all_data[scene][exp_key]['R_oob']:.3f}")
                        any_oob = True
    if not any_oob:
        print("  ✓ All methods OOB=0 across all 16 scenes")

    # ── Degradation × Conflict averages ──
    print(f"\n\n  {'DEGRADATION × CONFLICT AVERAGES (R_task)':─^100}")
    header = f"  {'Method':<22s}"
    for level in DEGRADATION:
        header += f" C3-{level[:4]:>4s}  "
    header += "  Avg"
    print(header)
    print(f"  {'-'*70}")
    for exp_key, label in BASELINES:
        row = f"  {label:<22s}"
        method_vals = []
        for level in DEGRADATION:
            level_vals = []
            for conflict in CONFLICTS:
                scene = f"c3_{level}_{conflict}"
                if scene in all_data:
                    level_vals.append(all_data[scene][exp_key]["R_task"])
            avg = statistics.mean(level_vals) if level_vals else 0
            method_vals.append(avg)
            row += f" {avg:7.3f}"
        row += f" {statistics.mean(method_vals):7.3f}"
        print(row)

    # Save
    out_dir = Path("results_v2/v5_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for scene, methods in all_data.items():
        out[scene] = {ek: {"label": v["label"], "R_task": v["R_task"], "R_task_std": v["R_task_std"],
                            "R_cov": v["R_cov"], "R_fail_given_cov": v["R_fail_given_cov"], "R_oob": v["R_oob"]}
                       for ek, v in methods.items()}
    with open(out_dir / "v5_16scene_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_dir / 'v5_16scene_sweep.json'}")

if __name__ == "__main__":
    main()
