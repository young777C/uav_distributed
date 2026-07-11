#!/usr/bin/env python3
"""Analyze dual time-scale sensitivity sweep results.

Usage:
    python3 scripts/analyze_ts_sensitivity.py

Outputs:
    - Prints aggregate tables to stdout
    - Saves summary CSVs under results/ts_sensitivity/analysis/
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SWEEP_ROOTS: Dict[str, Path] = {
    "FDLC C1-G2-M2": PROJECT_ROOT / "results/ts_sensitivity/layer1_c1/c1_fdlc",
    "FDLC C2-G2-M2": PROJECT_ROOT / "results/ts_sensitivity/layer1_c2/c2_fdlc",
    "PeriodicGoal C2-G2-M2": PROJECT_ROOT / "results/ts_sensitivity/layer2_periodic/c2_periodic",
}

OUT_DIR = PROJECT_ROOT / "results/ts_sensitivity/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_mean(xs: List[float]) -> float:
    return float(statistics.fmean(xs)) if xs else float("nan")


def _safe_stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(statistics.stdev(xs))


def _metric_val(obj: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k not in obj or obj[k] is None:
            continue
        try:
            v = float(obj[k])
        except (TypeError, ValueError):
            continue
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return None


def collect_run(metrics_path: Path) -> Dict[str, Any]:
    """Collect per-episode metrics from a single seed run."""
    episodes: List[Dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            episodes.append(obj)

    if not episodes:
        return {}

    covs = [_metric_val(e, "R_cov", "coverage_ratio") for e in episodes]
    tasks = [_metric_val(e, "R_task", "effective_ratio") for e in episodes]
    rets = [_metric_val(e, "T_ret_s", "mean_key_return_delay_s") for e in episodes]
    rets_cov = [_metric_val(e, "P_ret_given_cov", "R_fail_given_cov") for e in episodes]
    nf = [_metric_val(e, "T_nf_s", "nofly_dwell_s") for e in episodes]
    replans = [_metric_val(e, "replan_count") for e in episodes]
    goal_sw = [_metric_val(e, "goal_switch_count") for e in episodes]
    dhome = [_metric_val(e, "dist_to_home_m") for e in episodes]
    steps = [_metric_val(e, "steps") for e in episodes]
    energy = [_metric_val(e, "remaining_energy") for e in episodes]

    # Termination reasons
    term_energy = sum(1 for e in episodes if str(e.get("termination_reason", "")).lower() == "energy_depleted")
    term_return = sum(1 for e in episodes if str(e.get("termination_reason", "")).lower() == "returned_home")
    term_oob = sum(1 for e in episodes if str(e.get("termination_reason", "")).lower() == "oob")
    term_time = sum(1 for e in episodes if str(e.get("termination_reason", "")).lower() == "time_limit")
    n_ep = len(episodes)

    # Diag file for detailed replan breakdown
    diag_path = metrics_path.parent / "diag.jsonl"
    n_periodic = 0
    n_event = 0
    solve_times: List[float] = []
    if diag_path.exists():
        with diag_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("event_type") == "replan":
                    if d.get("trigger") == "periodic":
                        n_periodic += 1
                    elif d.get("trigger") in ("event", "safety", "link_drop", "energy_low"):
                        n_event += 1
                    st = _metric_val(d, "solve_time_s")
                    if st is not None:
                        solve_times.append(st)

    return {
        "episodes": n_ep,
        "R_cov_mean": _safe_mean([v for v in covs if v is not None]),
        "R_task_mean": _safe_mean([v for v in tasks if v is not None]),
        "R_task_std": _safe_stdev([v for v in tasks if v is not None]),
        "T_ret_mean": _safe_mean([v for v in rets if v is not None]),
        "P_ret_given_cov": _safe_mean([1 - (v or 0) for v in rets_cov if v is not None]),
        "T_nf_mean": _safe_mean([v for v in nf if v is not None]),
        "replan_count_mean": _safe_mean([v for v in replans if v is not None]),
        "goal_switch_mean": _safe_mean([v for v in goal_sw if v is not None]),
        "dist_home_mean": _safe_mean([v for v in dhome if v is not None]),
        "remaining_energy_mean": _safe_mean([v for v in energy if v is not None]),
        "returned_home_rate": term_return / max(1, n_ep),
        "energy_rate": term_energy / max(1, n_ep),
        "oob_rate": term_oob / max(1, n_ep),
        "time_limit_rate": term_time / max(1, n_ep),
        "n_periodic": n_periodic,
        "n_event": n_event,
        "n_replan_total": n_periodic + n_event,
        "event_ratio": n_event / max(1, n_periodic + n_event),
        "solve_time_mean": _safe_mean(solve_times),
        "solve_time_p95": sorted(solve_times)[max(0, int(len(solve_times)*0.95)-1)] if len(solve_times) > 5 else float("nan"),
        "total_solve_time": sum(solve_times),
    }


def parse_exp_name(exp_name: str) -> Tuple[str, str, int]:
    """Extract case, system, Ts from experiment directory structure."""
    # exp_name format: "{case}__{system}" or similar
    if "__" in exp_name:
        parts = exp_name.split("__", 1)
        case = parts[0]
        system = parts[1]
    else:
        case = exp_name
        system = exp_name
    return case, system, 0  # Ts extracted from directory name


def collect_all() -> List[Dict[str, Any]]:
    """Collect all sweep results."""
    all_rows: List[Dict[str, Any]] = []

    for label, sweep_root in SWEEP_ROOTS.items():
        if not sweep_root.is_dir():
            print(f"  [SKIP] {label}: directory not found: {sweep_root}")
            continue

        # Discover all metrics.jsonl files
        for metrics_path in sorted(sweep_root.rglob("metrics.jsonl")):
            if not metrics_path.is_file():
                continue
            rel = metrics_path.relative_to(sweep_root)
            parts = rel.parts
            if len(parts) < 3:
                continue

            exp_name = parts[0]      # e.g. "c2_g2_m2__struct_full_dual_loop_distributed"
            ts_str = parts[1]         # e.g. "Ts5" or "Ts40"
            seed_str = parts[2]       # e.g. "seed0"

            ts = int(ts_str[2:]) if ts_str.startswith("Ts") else 0
            seed = int(seed_str[4:]) if seed_str.startswith("seed") else -1

            data = collect_run(metrics_path)
            if not data:
                continue

            row = {
                "label": label,
                "exp_name": exp_name,
                "Ts": ts,
                "rho": ts,  # rho = Ts/Tf where Tf=1 step
                "seed": seed,
            }
            row.update(data)
            all_rows.append(row)

    return all_rows


def print_aggregate_table(rows: List[Dict[str, Any]]) -> None:
    """Print aggregate results by (label, Ts)."""
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["Ts"])].append(r)

    print("=" * 140)
    print(f"{'Experiment':<32} {'ρ':>4} {'N_seed':>6} {'R_task':>8} {'R_task_σ':>8} {'T_ret':>8} {'P_ret|cov':>10} {'Replan':>8} {'Event%':>8} {'Solve(s)':>9} {'Home%':>8} {'OOB%':>8} {'N_ep':>5}")
    print("-" * 140)

    for (label, ts) in sorted(groups.keys()):
        members = groups[(label, ts)]
        n_seed = len(members)
        n_ep_total = sum(m.get("episodes", 0) for m in members)

        r_tasks = [m["R_task_mean"] for m in members if not math.isnan(m["R_task_mean"])]
        r_task_mm = _safe_mean(r_tasks)
        r_task_sd = _safe_stdev(r_tasks)

        t_rets = [m["T_ret_mean"] for m in members if not math.isnan(m.get("T_ret_mean", float("nan")))]
        t_ret_m = _safe_mean(t_rets)

        p_rets = [m["P_ret_given_cov"] for m in members if not math.isnan(m.get("P_ret_given_cov", float("nan")))]
        p_ret_m = _safe_mean(p_rets)

        replan_m = _safe_mean([m["replan_count_mean"] for m in members])
        event_r = _safe_mean([m["event_ratio"] for m in members])
        solve_m = _safe_mean([m["solve_time_mean"] for m in members])
        home_r = _safe_mean([m["returned_home_rate"] for m in members])
        oob_r = _safe_mean([m["oob_rate"] for m in members])

        print(
            f"{label:<32} {ts:>4} {n_seed:>6} "
            f"{r_task_mm:>8.3f} {r_task_sd:>8.3f} "
            f"{t_ret_m:>8.2f} {p_ret_m:>10.3f} "
            f"{replan_m:>8.1f} {event_r:>8.3f} "
            f"{solve_m:>9.3f} {home_r:>8.3f} {oob_r:>8.3f} {n_ep_total:>5}"
        )

    print("=" * 140)


def write_csv(rows: List[Dict[str, Any]]) -> None:
    """Write aggregate per-(label, Ts) table to CSV."""
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["Ts"])].append(r)

    out_path = OUT_DIR / "ts_sensitivity_aggregate.csv"
    fieldnames = [
        "label", "rho", "n_seed", "n_episodes",
        "R_task_mean", "R_task_std",
        "T_ret_mean",
        "P_ret_given_cov",
        "replan_mean",
        "event_ratio",
        "solve_time_mean", "solve_time_p95", "total_solve_time",
        "returned_home_rate", "oob_rate",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (label, ts) in sorted(groups.keys()):
            members = groups[(label, ts)]
            r_tasks = [m["R_task_mean"] for m in members if not math.isnan(m["R_task_mean"])]
            w.writerow({
                "label": label,
                "rho": ts,
                "n_seed": len(members),
                "n_episodes": sum(m.get("episodes", 0) for m in members),
                "R_task_mean": f"{_safe_mean(r_tasks):.4f}",
                "R_task_std": f"{_safe_stdev(r_tasks):.4f}",
                "T_ret_mean": f"{_safe_mean([m['T_ret_mean'] for m in members if not math.isnan(m.get('T_ret_mean', float('nan')))]):.3f}",
                "P_ret_given_cov": f"{_safe_mean([m['P_ret_given_cov'] for m in members if not math.isnan(m.get('P_ret_given_cov', float('nan')))]):.4f}",
                "replan_mean": f"{_safe_mean([m['replan_count_mean'] for m in members]):.1f}",
                "event_ratio": f"{_safe_mean([m['event_ratio'] for m in members]):.4f}",
                "solve_time_mean": f"{_safe_mean([m['solve_time_mean'] for m in members if not math.isnan(m.get('solve_time_mean', float('nan')))]):.4f}",
                "solve_time_p95": f"{_safe_mean([m['solve_time_p95'] for m in members if not math.isnan(m.get('solve_time_p95', float('nan')))]):.4f}",
                "total_solve_time": f"{_safe_mean([m['total_solve_time'] for m in members]):.1f}",
                "returned_home_rate": f"{_safe_mean([m['returned_home_rate'] for m in members]):.4f}",
                "oob_rate": f"{_safe_mean([m['oob_rate'] for m in members]):.4f}",
            })
    print(f"\n[CSV] Wrote: {out_path}")


def write_per_seed_csv(rows: List[Dict[str, Any]]) -> None:
    """Write per-(label, Ts, seed) table for detailed analysis."""
    out_path = OUT_DIR / "ts_sensitivity_per_seed.csv"
    fieldnames = [
        "label", "rho", "seed", "episodes",
        "R_task_mean", "R_task_std",
        "T_ret_mean", "P_ret_given_cov",
        "replan_count_mean", "event_ratio",
        "solve_time_mean", "solve_time_p95",
        "returned_home_rate", "oob_rate",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["label"], x["Ts"], x["seed"])):
            w.writerow({
                "label": r["label"],
                "rho": r["Ts"],
                "seed": r["seed"],
                "episodes": r.get("episodes", 0),
                "R_task_mean": f"{r.get('R_task_mean', float('nan')):.4f}",
                "R_task_std": f"{r.get('R_task_std', float('nan')):.4f}",
                "T_ret_mean": f"{r.get('T_ret_mean', float('nan')):.3f}",
                "P_ret_given_cov": f"{r.get('P_ret_given_cov', float('nan')):.4f}",
                "replan_count_mean": f"{r.get('replan_count_mean', float('nan')):.1f}",
                "event_ratio": f"{r.get('event_ratio', float('nan')):.4f}",
                "solve_time_mean": f"{r.get('solve_time_mean', float('nan')):.4f}",
                "solve_time_p95": f"{r.get('solve_time_p95', float('nan')):.4f}",
                "returned_home_rate": f"{r.get('returned_home_rate', float('nan')):.4f}",
                "oob_rate": f"{r.get('oob_rate', float('nan')):.4f}",
            })
    print(f"[CSV] Wrote: {out_path}")


def main() -> None:
    rows = collect_all()
    if not rows:
        print("No data collected. Sweeps may still be running.")
        print(f"Check sweep roots: { {k: str(v) for k, v in SWEEP_ROOTS.items()} }")
        return

    print(f"\nCollected {len(rows)} (label, Ts, seed) combinations.\n")
    print_aggregate_table(rows)
    write_csv(rows)
    write_per_seed_csv(rows)

    # Print summary by label
    for label in sorted(set(r["label"] for r in rows)):
        label_rows = [r for r in rows if r["label"] == label]
        print(f"\n=== {label} ({len(label_rows)} seed-runs) ===")
        for r in sorted(label_rows, key=lambda x: x["Ts"]):
            print(
                f"  ρ={r['Ts']:>2}  "
                f"R_task={r.get('R_task_mean', float('nan')):.3f}±{r.get('R_task_std', float('nan')):.3f}  "
                f"T_ret={r.get('T_ret_mean', float('nan')):.2f}s  "
                f"Replan={r.get('replan_count_mean', float('nan')):.0f}  "
                f"Event%={r.get('event_ratio', float('nan')):.2%}  "
                f"Home%={r.get('returned_home_rate', float('nan')):.0%}  "
                f"OOB%={r.get('oob_rate', float('nan')):.0%}"
            )


if __name__ == "__main__":
    main()
