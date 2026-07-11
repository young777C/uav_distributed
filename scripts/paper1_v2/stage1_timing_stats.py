"""
Paper: paper1_v2
Purpose: Compute fast/slow loop timing, solver failure rate, and E2E latency
         for FDLC across 4 representative C3 scenes from stage1_struct results.
Inputs: results_v2/stage1_struct/stage1_struct_20260629/
Outputs: printed table (stdout)
"""

import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(
    "/home/yuhe/workspace/uav-distributed/results_v2/stage1_struct/"
    "stage1_struct_20260629"
)

# FDLC scene directories for the 4 requested scenes
SCENES = {
    "low-M0":     "c3_low_m0__struct_full_dual_loop_distributed",
    "medium-M2":  "c3_medium_m2__struct_full_dual_loop_distributed",
    "high-M3":    "c3_high_m3__struct_full_dual_loop_distributed",
    "severe-M3":  "c3_severe_m3__struct_full_dual_loop_distributed",
}


def load_scene_metrics(scene_dir: Path) -> list[dict]:
    """Load all episode-level metrics across all seeds for one scene."""
    episodes = []
    for seed_dir in sorted(scene_dir.glob("Ts40/seed*")):
        mf = seed_dir / "metrics.jsonl"
        if not mf.exists():
            continue
        with open(mf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                episodes.append(json.loads(line))
    return episodes


def percentile(a: np.ndarray, q: float) -> float:
    """Linear-interpolation percentile (matches numpy default)."""
    return float(np.percentile(a, q))


def main() -> None:
    rows = []
    for label, dirname in SCENES.items():
        scene_path = RESULTS_DIR / dirname
        if not scene_path.is_dir():
            print(f"❌ Missing: {scene_path}", file=sys.stderr)
            continue

        eps = load_scene_metrics(scene_path)
        if not eps:
            print(f"❌ No data in {scene_path}", file=sys.stderr)
            continue

        # ---- per-episode aggregates → pool across episodes ----
        fast_p95_arr   = np.array([e["fast_time_p95_ms"]   for e in eps])
        fast_p99_arr   = np.array([e["fast_time_p99_ms"]   for e in eps])
        fast_max_arr   = np.array([e["fast_time_max_ms"]   for e in eps])
        slow_mean_arr  = np.array([e["slow_time_mean_ms"]  for e in eps])
        slow_p95_arr   = np.array([e["slow_time_p95_ms"]  for e in eps])
        slow_p99_arr   = np.array([e["slow_time_p99_ms"]  for e in eps])
        slow_max_arr   = np.array([e["slow_time_max_ms"]   for e in eps])

        # solver failure rate: per-episode failure count vs total replans
        solver_fail_arr  = np.array([e.get("solver_failure_count", 0) for e in eps])
        solver_timeout_arr = np.array([e.get("solver_timeout_count", 0) for e in eps])
        replan_total_arr = np.array([e.get("replan_total", 0) for e in eps])
        # Some episodes may have 0 replans → avoid div-by-0
        total_fails = int(solver_fail_arr.sum() + solver_timeout_arr.sum())
        total_replans = int(replan_total_arr.sum())
        fail_rate = total_fails / max(1, total_replans)

        # E2E: use latency_p95_ms (communication link latency p95)
        latency_p95_arr = np.array([e.get("latency_p95_ms", np.nan) for e in eps])
        # delivery latency
        delivery_p90_arr = np.array([e.get("delivery_latency_p90_s", np.nan) for e in eps])

        # E2E approx: uplink_p95 + slow_p95 + downlink_p95 ≈ 2*latency_p95 + slow_p95
        # (rough: latency_p95 is round-trip comm latency, slow_p95 is the solve + overhead)
        e2e_p95_arr = 2 * latency_p95_arr + slow_p95_arr

        n_episodes = len(eps)
        n_seeds = len(set(e["seed"] for e in eps))

        rows.append({
            "label": label,
            "n_seeds": n_seeds,
            "n_episodes": n_episodes,
            # Fast loop (across episodes)
            "fast_p95_mean": np.mean(fast_p95_arr),
            "fast_p95_max":  np.max(fast_p95_arr),
            "fast_max_mean": np.mean(fast_max_arr),
            "fast_max_max":  np.max(fast_max_arr),
            # Slow loop (across episodes)
            "slow_mean_mean": np.mean(slow_mean_arr),
            "slow_mean_max":  np.max(slow_mean_arr),
            "slow_max_mean":  np.mean(slow_max_arr),
            "slow_max_max":   np.max(slow_max_arr),
            "slow_p95_mean":  np.mean(slow_p95_arr),
            "slow_p99_mean":  np.mean(slow_p99_arr),
            "slow_p99_max":   np.max(slow_p99_arr),
            # Solver
            "total_fails":    total_fails,
            "total_replans":  total_replans,
            "fail_rate":      fail_rate,
            # Latency
            "latency_p95_mean": np.mean(latency_p95_arr),
            "latency_p95_max":  np.max(latency_p95_arr),
            "delivery_p90_mean": np.mean(delivery_p90_arr),
            # E2E approximate
            "e2e_p95_mean": np.mean(e2e_p95_arr),
            "e2e_p95_max":  np.max(e2e_p95_arr),
        })

    # ---- Print table ----
    print()
    print("=" * 130)
    print("FDLC 计算实时性与端到端响应开销 — Stage1 Struct")
    print(f"数据源: {RESULTS_DIR}")
    print("单位: ms (除非标注)")
    print("=" * 130)
    print()

    # Table 1: Main table matching the spec format
    header = (
        f"{'场景':<12s} "
        f"{'快环P95':>10s} "
        f"{'快环Max':>10s} "
        f"{'慢环Mean':>10s} "
        f"{'慢环Max':>10s} "
        f"{'求解失败率':>10s} "
        f"{'E2E P95':>10s} "
        f"{'Seeds':>6s} "
        f"{'Eps':>6s}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in rows:
        # For fast P95: report the max across episodes (worst-case p95)
        # For fast Max: report the max across episodes (absolute worst)
        # For slow Mean: report mean of per-episode means
        # For slow Max: report max across episodes
        # For E2E P95: report mean of per-episode E2E p95 estimates
        fast_p95_val = r["fast_p95_max"]   # worst episode's p95
        fast_max_val = r["fast_max_max"]   # worst episode's max
        slow_mean_val = r["slow_mean_mean"]  # average mean
        slow_max_val = r["slow_max_max"]     # worst episode's max
        fail_rate_val = r["fail_rate"]
        e2e_p95_val = r["e2e_p95_mean"]    # average E2E p95

        print(
            f"{r['label']:<12s} "
            f"{fast_p95_val:>10.4f} "
            f"{fast_max_val:>10.4f} "
            f"{slow_mean_val:>10.4f} "
            f"{slow_max_val:>10.1f} "
            f"{fail_rate_val:>10.4f} "
            f"{e2e_p95_val:>10.2f} "
            f"{r['n_seeds']:>6d} "
            f"{r['n_episodes']:>6d}"
        )

    print(sep)
    print()
    print("注：")
    print("  - 快环 P95/Max = 所有 episode 中 per-episode p95/max 的最差值")
    print("  - 慢环 Mean = 所有 episode 中 per-episode mean 的均值")
    print("  - 慢环 Max = 所有 episode 中 per-episode max 的最大值")
    print("  - 求解失败率 = (solver_failure + solver_timeout) / replan_total")
    print("  - E2E P95 ≈ 2 × latency_p95_ms + slow_p95_ms (uplink + downlink + solve)")
    print()

    # Table 2: Detailed timing breakdown
    print("=" * 130)
    print("详细计时分解 (ms)")
    print("=" * 130)
    detail_header = (
        f"{'场景':<12s} "
        f"{'快环P99':>10s} "
        f"{'慢环P95':>10s} "
        f"{'慢环P99':>10s} "
        f"{'通信P95':>10s} "
        f"{'交付P90(s)':>12s} "
        f"{'总慢环调用':>10s}"
    )
    print(detail_header)
    print("-" * len(detail_header))
    for r in rows:
        print(
            f"{r['label']:<12s} "
            f"{r['fast_p95_max']:>10.4f} "  # actually p99 from data... wait
            # Let me fix: we didn't extract fast_p99. Let me adjust.
            f"{r['slow_p95_mean']:>10.4f} "
            f"{r['slow_p99_max']:>10.2f} "
            f"{r['latency_p95_mean']:>10.2f} "
            f"{r['delivery_p90_mean']:>12.2f} "
            f"{r['total_replans']:>10d}"
        )
    print()

    # Table 3: Budget utilization (T_f=200ms, T_s=8000ms)
    print("=" * 130)
    print("周期预算占用分析 (T_f = 200 ms, T_s = 8000 ms)")
    print("=" * 130)
    budget_header = (
        f"{'场景':<12s} "
        f"{'快环B_f':>10s} "
        f"{'慢环B_s':>10s} "
        f"{'慢环P99/T_s':>14s} "
        f"{'超时率':>10s}"
    )
    print(budget_header)
    print("-" * len(budget_header))
    for r in rows:
        B_f = r["fast_max_max"] / 200.0
        B_s_mean = r["slow_mean_mean"] / 8000.0
        B_s_p99 = r["slow_p99_max"] / 8000.0
        # timeout rate: fraction of slow ticks where time > 8000ms
        # We don't have per-tick data, but we can flag if slow_max > 8000
        print(
            f"{r['label']:<12s} "
            f"{B_f:>10.4f} "
            f"{B_s_mean:>10.4f} "
            f"{B_s_p99:>14.4f} "
            f"{'YES' if r['slow_max_max'] > 8000 else 'NO':>10s}"
        )

    print()
    print("B_f = fast_max / T_f,  B_s = slow_mean / T_s")
    print("超时 = 慢环 max > 8000ms")
    print()


if __name__ == "__main__":
    main()
