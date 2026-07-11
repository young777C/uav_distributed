#!/usr/bin/env python3
"""Analyze completed time-scale sensitivity results from results/ts_sensitivity/parallel/.

Usage:
    PYTHONPATH=src python3 scripts/analyze_ts_results.py
    PYTHONPATH=src python3 scripts/analyze_ts_results.py --latex-table  # print LaTeX table
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARALLEL_DIR = PROJECT_ROOT / "results/ts_sensitivity/parallel"
EPISODES_EXPECTED = 3


def _safe_mean(xs: List[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _safe_stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return statistics.stdev(xs)


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


def read_metrics(metrics_path: Path) -> List[Dict[str, Any]]:
    episodes = []
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return episodes
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except Exception:
                pass
    return episodes


def read_diag(diag_path: Path) -> Dict[str, Any]:
    """Read diagnostics for replan breakdown."""
    n_periodic = 0
    n_event = 0
    solve_times = []
    if not diag_path.exists():
        return {"n_periodic": 0, "n_event": 0, "solve_times": []}
    with diag_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            ev = d.get("event", "")
            reason = d.get("replan_reason", "")
            if ev == "slow_replan":
                if reason == "periodic":
                    n_periodic += 1
                else:
                    n_event += 1
                st = _metric_val(d, "solve_time_s")
                if st is not None:
                    solve_times.append(st)
    return {"n_periodic": n_periodic, "n_event": n_event, "solve_times": solve_times}


def compute_episode_stats(episodes: List[Dict[str, Any]], diag: Dict[str, Any]) -> Dict[str, float]:
    if not episodes:
        return {}

    r_tasks = [_metric_val(e, "R_task", "effective_ratio") for e in episodes]
    r_covs = [_metric_val(e, "R_cov", "coverage_ratio") for e in episodes]
    t_rets = [_metric_val(e, "T_ret_s", "mean_key_return_delay_s") for e in episodes]
    p_ret_cov = [1 - (_metric_val(e, "R_fail_given_cov", "R_fail_cov") or 0) for e in episodes]
    replans = [_metric_val(e, "replan_count") for e in episodes]
    goal_sw = [_metric_val(e, "goal_switch_count") for e in episodes]

    term_return = sum(1 for e in episodes if str(e.get("termination_reason", "")).lower() == "returned_home")
    term_energy = sum(1 for e in episodes if str(e.get("termination_reason", "")).lower() == "energy_depleted")
    term_oob = sum(1 for e in episodes if str(e.get("termination_reason", "")).lower() == "oob")
    n_ep = len(episodes)

    st_mean = _safe_mean(diag["solve_times"])
    st_p95 = sorted(diag["solve_times"])[max(0, int(len(diag["solve_times"]) * 0.95) - 1)] if len(diag["solve_times"]) > 5 else float("nan")

    n_periodic_diag = diag["n_periodic"]
    n_event_diag = diag["n_event"]
    # Use diag event ratio applied to metrics replan total for consistency
    diag_total = n_periodic_diag + n_event_diag
    event_ratio = n_event_diag / max(1, diag_total) if diag_total > 0 else 0.0

    return {
        "n_ep": n_ep,
        "R_cov_mean": _safe_mean([v for v in r_covs if v is not None]),
        "R_task_mean": _safe_mean([v for v in r_tasks if v is not None]),
        "R_task_std": _safe_stdev([v for v in r_tasks if v is not None]),
        "T_ret_mean": _safe_mean([v for v in t_rets if v is not None]),
        "P_ret_given_cov": _safe_mean([v for v in p_ret_cov if v is not None]),
        "replan_mean": _safe_mean([v for v in replans if v is not None]),
        "goal_switch_mean": _safe_mean([v for v in goal_sw if v is not None]),
        "n_periodic": n_periodic_diag,
        "n_event": n_event_diag,
        "event_ratio": event_ratio,
        "solve_time_mean": st_mean,
        "solve_time_p95": st_p95,
        "total_solve_time": sum(diag["solve_times"]),
        "returned_home_rate": term_return / max(1, n_ep),
        "energy_rate": term_energy / max(1, n_ep),
        "oob_rate": term_oob / max(1, n_ep),
    }


def collect_results() -> List[Dict[str, Any]]:
    rows = []
    for exp_dir in sorted(PARALLEL_DIR.iterdir()):
        if not exp_dir.is_dir() or "__" not in exp_dir.name:
            continue
        case, system = exp_dir.name.split("__", 1)
        for ts_dir in sorted(exp_dir.iterdir()):
            if not ts_dir.name.startswith("Ts"):
                continue
            ts = int(ts_dir.name[2:])
            for seed_dir in sorted(ts_dir.iterdir()):
                if not seed_dir.name.startswith("seed"):
                    continue
                seed = int(seed_dir.name[4:])
                metrics_path = seed_dir / "metrics.jsonl"
                diag_path = seed_dir / "diag.jsonl"
                if not metrics_path.exists() or metrics_path.stat().st_size == 0:
                    continue
                episodes = read_metrics(metrics_path)
                if len(episodes) < EPISODES_EXPECTED:
                    continue
                diag = read_diag(diag_path)
                stats = compute_episode_stats(episodes, diag)
                if not stats:
                    continue
                stats["case"] = case
                stats["system"] = system
                stats["ts"] = ts
                stats["seed"] = seed
                rows.append(stats)
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, int], List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["case"], r["system"], r["ts"])].append(r)
    return dict(groups)


def print_human_table(rows: List[Dict[str, Any]]) -> None:
    groups = aggregate(rows)
    print(f"\n{'Scenario':<45} {'ρ':>4} {'N':>3} {'R_task':>8} {'R_task_σ':>8} {'T_ret':>8} {'P_ret|cov':>10} "
          f"{'Replan':>8} {'Event%':>8} {'Home%':>8} {'OOB%':>8}")
    print("-" * 120)
    for key in sorted(groups.keys()):
        case, system, ts = key
        members = groups[key]
        r_tasks = [m["R_task_mean"] for m in members]
        t_rets = [m["T_ret_mean"] for m in members]
        p_rets = [m["P_ret_given_cov"] for m in members]
        replans = [m["replan_mean"] for m in members]
        event_r = [m["event_ratio"] for m in members]
        home_r = [m["returned_home_rate"] for m in members]
        oob_r = [m["oob_rate"] for m in members]
        label = f"{case} {system}"
        print(f"{label:<45} {ts:>4} {len(members):>3} "
              f"{_safe_mean(r_tasks):>8.3f} {_safe_stdev(r_tasks):>8.3f} "
              f"{_safe_mean(t_rets):>8.2f} {_safe_mean(p_rets):>10.3f} "
              f"{_safe_mean(replans):>8.1f} {_safe_mean(event_r):>8.3f} "
              f"{_safe_mean(home_r):>8.3f} {_safe_mean(oob_r):>8.3f}")
    print()


def print_latex_tables(rows: List[Dict[str, Any]]) -> None:
    """Print LaTeX tables for the paper."""
    groups = aggregate(rows)

    # Map to display names
    display = {
        ("c1_g2_m2", "struct_full_dual_loop_distributed"): r"\texttt{c1\_g2\_m2}",
        ("c2_g2_m2", "struct_full_dual_loop_distributed"): r"\texttt{c2\_g2\_m2}",
        ("c2_g2_m2", "ts_sweep_periodic_goal"): r"\texttt{c2\_g2\_m2}",
    }
    method_labels = {
        "struct_full_dual_loop_distributed": "FDLC",
        "ts_sweep_periodic_goal": "Periodic Goal",
    }

    # ---- Table 1: FDLC time-scale sensitivity ----
    print("\n% === Table 1: FDLC time-scale sensitivity ===")
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{FDLC在不同时间尺度比$\rho$下的任务性能与重规划开销。}")
    print(r"  \label{tab:ts_fdlc}")
    print(r"  \small")
    print(r"  \begin{tabular}{lcccccc}")
    print(r"    \toprule")
    print(r"    场景 & $\rho$ & $R_{\mathrm{task}}$ & $T_{\mathrm{ret}}$ (s) & 总重规划次数 & 事件触发占比 & 返航终止率 \\")
    print(r"    \midrule")

    for case in ["c1_g2_m2", "c2_g2_m2"]:
        key_prefix = (case, "struct_full_dual_loop_distributed")
        ts_vals = sorted([k[2] for k in groups.keys() if k[:2] == key_prefix])
        scene_label = display.get(key_prefix, case)
        print(f"    \\multirow{{{len(ts_vals)}}}{{*}}{{{scene_label}}}")
        for i, ts in enumerate(ts_vals):
            members = groups[key_prefix + (ts,)]
            r_t = _safe_mean([m["R_task_mean"] for m in members])
            t_r = _safe_mean([m["T_ret_mean"] for m in members])
            n_r = _safe_mean([m["replan_mean"] for m in members])
            e_r = _safe_mean([m["event_ratio"] for m in members])
            h_r = _safe_mean([m["returned_home_rate"] for m in members])
            sep = " \\\\" if i < len(ts_vals) - 1 else " \\\\"
            print(f"      & {ts:>2}  & {r_t*100:.1f}\\% & {t_r:.1f} & {n_r:.0f} & {e_r*100:.1f}\\% & {h_r*100:.0f}\\%{sep}")
        if case == "c1_g2_m2":
            print(r"    \midrule")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")

    # ---- Table 2: Periodic Goal vs FDLC ----
    print("\n% === Table 2: Periodic Goal vs FDLC (C2-G2-M2) ===")
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Periodic Goal与FDLC在C2--G2--M2中的时间尺度敏感性对比。}")
    print(r"  \label{tab:ts_periodic}")
    print(r"  \small")
    print(r"  \begin{tabular}{lcccc}")
    print(r"    \toprule")
    print(r"    方法 & $\rho$ & $R_{\mathrm{task}}$ & $T_{\mathrm{ret}}$ (s) & 总重规划次数 \\")
    print(r"    \midrule")

    for system in ["ts_sweep_periodic_goal", "struct_full_dual_loop_distributed"]:
        key_prefix = ("c2_g2_m2", system)
        ts_vals = sorted([k[2] for k in groups.keys() if k[:2] == key_prefix])
        label = method_labels.get(system, system)
        print(f"    \\multirow{{{len(ts_vals)}}}{{*}}{{{label}}}")
        for i, ts in enumerate(ts_vals):
            members = groups[key_prefix + (ts,)]
            r_t = _safe_mean([m["R_task_mean"] for m in members])
            t_r = _safe_mean([m["T_ret_mean"] for m in members])
            n_r = _safe_mean([m["replan_mean"] for m in members])
            sep = " \\\\" if i < len(ts_vals) - 1 else " \\\\"
            print(f"      & {ts:>2}  & {r_t*100:.1f}\\% & {t_r:.1f} & {n_r:.0f}{sep}")
        print(r"    \midrule")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")

    # ---- Table 3: Computation overhead ----
    print("\n% === Table 3: Computation overhead ===")
    total_steps = 10000  # 2000s at 5Hz
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{不同时间尺度比下的规划计算开销。}")
    print(r"  \label{tab:ts_computation}")
    print(r"  \small")
    print(r"  \begin{tabular}{lccccc}")
    print(r"    \toprule")
    print(r"    场景 & $\rho$ & 平均求解时间 (s) & P95求解时间 (s) & 单回合总规划时间 (s) & 计算负载 $\eta_{\mathrm{comp}}$ \\")
    print(r"    \midrule")
    for case in ["c1_g2_m2", "c2_g2_m2"]:
        key_prefix = (case, "struct_full_dual_loop_distributed")
        ts_vals = sorted([k[2] for k in groups.keys() if k[:2] == key_prefix])
        scene_label = display.get(key_prefix, case)
        print(f"    \\multirow{{{len(ts_vals)}}}{{*}}{{{scene_label}}}")
        for i, ts in enumerate(ts_vals):
            members = groups[key_prefix + (ts,)]
            st_mean = _safe_mean([m["solve_time_mean"] for m in members])
            st_p95 = _safe_mean([m["solve_time_p95"] for m in members])
            total_st = _safe_mean([m["total_solve_time"] for m in members])
            comp_load = total_st / total_steps * 100
            sep = " \\\\" if i < len(ts_vals) - 1 else " \\\\"
            print(f"      & {ts:>2}  & {st_mean:.4f} & {st_p95:.4f} & {total_st:.1f} & {comp_load:.2f}\\%{sep}")
        print(r"    \midrule")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex-table", action="store_true")
    args = ap.parse_args()

    rows = collect_results()
    if not rows:
        print("No completed results found. Check:", PARALLEL_DIR)
        print("Sweep still running or data not yet available.")
        sys.exit(1)

    # Print task count summary
    groups = aggregate(rows)
    print(f"Found {len(rows)} seed-runs across {len(groups)} configurations:")
    for key in sorted(groups.keys()):
        case, system, ts = key
        n_seeds = len(groups[key])
        n_ep = sum(m["n_ep"] for m in groups[key])
        print(f"  {case} {system:40s} Ts={ts:>2}  ({n_seeds} seeds, {n_ep} episodes)")

    if args.latex_table:
        print_latex_tables(rows)
    else:
        print_human_table(rows)


if __name__ == "__main__":
    main()
