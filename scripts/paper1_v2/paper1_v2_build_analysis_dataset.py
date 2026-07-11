"""
Paper: paper1_v2
Purpose: Build episode/seed/scene three-layer analysis dataset from Stage 1 metrics.jsonl
Inputs:  results_v2/stage1_baselines/stage1_bl_20260629/
         results_v2/stage1_struct/stage1_struct_20260629/
Outputs: results_v2/summaries/paper1_v2_episode_analysis.csv
         results_v2/summaries/paper1_v2_seed_analysis.csv
         results_v2/summaries/paper1_v2_scene_analysis.csv
         results_v2/summaries/paper1_v2_protocol_report.md
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_ROOTS = [
    PROJECT_ROOT / "results_v2/stage1_baselines/stage1_bl_20260629",
    PROJECT_ROOT / "results_v2/stage1_struct/stage1_struct_20260629",
]
STAGE2_ROOTS = [
    PROJECT_ROOT / "results_v2/stage2_coupling/stage2_coupling_20260701",
]
STAGE1_FDLC_D2_ROOT = PROJECT_ROOT / "results_v2/stage1_fdlc_p0/stage1_fdlc_p0_v4_d2_20260704"
STAGE1_FDLC_ALLFIX_ROOT = PROJECT_ROOT / "results_v2/stage1_fdlc_p0/stage1_fdlc_allfix_m2m3_20260706"
STAGE1_CBCP_TIGHT_ROOT = PROJECT_ROOT / "results_v2/stage1_cbcp_tight/stage1_cbcp_tight_20260705"
# M2-M3 scenes: use AllFixes data instead of P0-d2
M2M3_SCENES = {f"c3_{i}_{c.lower()}" for i in ["medium", "high", "severe"] for c in ["M2", "M3"]}
OUT_DIR = PROJECT_ROOT / "results_v2/summaries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_MAP = {
    "baseline_rhc_inspection": "RHC-Inspection",
    "baseline_cbcp": "CBCP-BL",  # original → renamed; tight replaces it
    "struct_centralized_single_loop": "CDSL",
    "struct_decoupled_dual_loop": "WCDL",
    "struct_full_dual_loop_distributed": "FDLC-BL",  # original baseline → renamed
    "coupling_periodic_goal": "Periodic Goal",
    "coupling_event_driven_goal": "Event-driven Goal",
}
FDLC_D2_METHOD = "FDLC"  # new P0-optimized FDLC with dwell=2
CBCP_METHOD = "CBCP"      # tight CBCP replaces original

INTENSITY_ORDER = ["low", "medium", "high", "severe"]
CONFLICT_ORDER = ["M0", "M1", "M2", "M3"]

# ── helpers ─────────────────────────────────────────────────────────

def parse_scene_id(scene_id: str) -> Dict[str, str]:
    """c3_low_m0 → {intensity: low, conflict: M0}"""
    parts = scene_id.split("_")
    intensity = parts[1] if len(parts) > 1 else "unknown"
    conflict = parts[2].upper() if len(parts) > 2 else "unknown"
    return {"intensity": intensity, "conflict": conflict}


def load_episodes(metrics_path: Path) -> List[Dict[str, Any]]:
    """Load all episode lines from a metrics.jsonl file."""
    rows = []
    if not metrics_path.exists():
        return rows
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ── main ────────────────────────────────────────────────────────────

def main() -> None:
    episode_rows: List[Dict[str, Any]] = []
    protocol_info: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "expected_seeds": 5,
        "expected_episodes_per_seed": 15,
        "roots_scanned": [str(r) for r in STAGE1_ROOTS],
        "combo_details": [],
    }

    # ── scan all metrics.jsonl ──────────────────────────────────
    ALL_ROOTS = STAGE1_ROOTS + STAGE2_ROOTS + [STAGE1_FDLC_D2_ROOT, STAGE1_FDLC_ALLFIX_ROOT, STAGE1_CBCP_TIGHT_ROOT]
    for root in ALL_ROOTS:
        for metrics_path in sorted(root.rglob("metrics.jsonl")):
            if not metrics_path.is_file():
                continue
            try:
                rel = metrics_path.relative_to(root)
            except Exception:
                continue
            parts = rel.parts
            if len(parts) < 4:
                continue

            exp_name = parts[0]  # e.g. c3_low_m0__baseline_cbcp
            ts_part = parts[1]   # e.g. Ts40
            seed_part = parts[2]  # e.g. seed0

            scene_raw, system_raw = exp_name.split("__", 1) if "__" in exp_name else (exp_name, "")
            method = METHOD_MAP.get(system_raw, system_raw)
            # Skip M2-M3 scenes from P0-d2 root (replaced by AllFixes)
            if root == STAGE1_FDLC_D2_ROOT and scene_raw in M2M3_SCENES:
                continue
            # FDLC-d2 root → rename to "FDLC" (P0-optimized, M0-M1 only)
            if root == STAGE1_FDLC_D2_ROOT and method == "FDLC-BL":
                method = FDLC_D2_METHOD
            # AllFixes root → rename to "FDLC" (M2-M3 only, replaces P0-d2)
            if root == STAGE1_FDLC_ALLFIX_ROOT and method == "FDLC-BL":
                method = FDLC_D2_METHOD
            # Tight CBCP root → override original CBCP-BL
            if root == STAGE1_CBCP_TIGHT_ROOT and method == "CBCP-BL":
                method = CBCP_METHOD
            scene_info = parse_scene_id(scene_raw)
            ts = int(ts_part[2:]) if ts_part.startswith("Ts") else None
            seed = int(seed_part[4:]) if seed_part.startswith("seed") else None
            rho = ts

            eps = load_episodes(metrics_path)
            n_eps = len(eps)

            protocol_info["combo_details"].append({
                "exp_name": exp_name,
                "method": method,
                "scene": scene_raw,
                "intensity": scene_info["intensity"],
                "conflict": scene_info["conflict"],
                "rho": rho,
                "seed": seed,
                "n_episodes": n_eps,
            })

            for ep in eps:
                # Use C3 case name (c3_low_m0 etc.) from exp_name, NOT the G2 scene filename
                c3_scene = scene_raw  # e.g. c3_low_m0
                row = {
                    "run_id": ep.get("run_id", ""),
                    "method": method,
                    "scene_id": c3_scene,
                    "intensity": scene_info["intensity"],
                    "conflict": scene_info["conflict"],
                    "rho": rho if rho else ep.get("rho", None),
                    "seed": seed if seed is not None else ep.get("seed", None),
                    "episode_id": ep.get("episode_id", ep.get("episode", None)),
                    # primary
                    "R_task": safe_float(ep.get("R_task", ep.get("effective_ratio"))),
                    "R_cov": safe_float(ep.get("R_cov", ep.get("coverage_ratio"))),
                    "R_fail_given_cov": safe_float(ep.get("R_fail_given_cov", ep.get("R_fail_cov"))),
                    "coverage_ratio": safe_float(ep.get("coverage_ratio")),
                    "effective_ratio": safe_float(ep.get("effective_ratio")),
                    "covered_count": safe_float(ep.get("covered_count")),
                    "delivered_count": safe_float(ep.get("effective_return_count")),
                    # delivery
                    "delivery_latency_mean_s": safe_float(ep.get("delivery_latency_mean_s")),
                    "delivery_latency_median_s": safe_float(ep.get("delivery_latency_median_s")),
                    "delivery_latency_p90_s": safe_float(ep.get("delivery_latency_p90_s")),
                    # safety
                    "returned_home": bool(ep.get("returned_home", False)),
                    "oob_triggered": bool(ep.get("oob_triggered", False)),
                    "energy_triggered": bool(ep.get("energy_triggered", False)),
                    "timeout_triggered": bool(ep.get("timeout_triggered", False)),
                    "termination_reason": str(ep.get("termination_reason", "")),
                    "final_energy": safe_float(ep.get("final_energy")),
                    # comm
                    "loss_mean": safe_float(ep.get("loss_mean")),
                    "latency_mean_ms": safe_float(ep.get("latency_mean_ms")),
                    "link_quality_mean": safe_float(ep.get("link_quality_mean")),
                    # coordination
                    "replan_total": safe_float(ep.get("replan_total")),
                    "replan_periodic": safe_float(ep.get("replan_periodic")),
                    "replan_event": safe_float(ep.get("replan_event")),
                    "event_replan_ratio": safe_float(ep.get("event_replan_ratio")),
                    "feedback_total": safe_float(ep.get("feedback_total")),
                    "target_switch_count": safe_float(ep.get("target_switch_count")),
                    "mode_switch_count": safe_float(ep.get("mode_switch_count")),
                    # timing
                    "fast_time_mean_ms": safe_float(ep.get("fast_time_mean_ms")),
                    "fast_time_p95_ms": safe_float(ep.get("fast_time_p95_ms")),
                    "fast_time_p99_ms": safe_float(ep.get("fast_time_p99_ms")),
                    "slow_time_mean_ms": safe_float(ep.get("slow_time_mean_ms")),
                    "slow_time_p95_ms": safe_float(ep.get("slow_time_p95_ms")),
                    "slow_time_p99_ms": safe_float(ep.get("slow_time_p99_ms")),
                    "solver_timeout_count": safe_float(ep.get("solver_timeout_count")),
                    "solver_failure_count": safe_float(ep.get("solver_failure_count")),
                    "mip_gap_mean": safe_float(ep.get("mip_gap_mean")),
                    "episode_wall_time_s": safe_float(ep.get("episode_wall_time_s")),
                    # other
                    "steps": safe_float(ep.get("steps")),
                    "remaining_energy": safe_float(ep.get("remaining_energy")),
                    "git_commit": ep.get("git_commit", ""),
                    "config_hash": ep.get("config_hash", ""),
                }
                episode_rows.append(row)

    # ── episode dataframe ───────────────────────────────────────
    df_ep = pd.DataFrame(episode_rows)
    df_ep.to_csv(OUT_DIR / "paper1_v2_episode_analysis.csv", index=False)
    print(f"Episode CSV: {len(df_ep)} rows → {OUT_DIR / 'paper1_v2_episode_analysis.csv'}")

    # ── seed aggregation ────────────────────────────────────────
    seed_groups = df_ep.groupby(["method", "scene_id", "intensity", "conflict", "rho", "seed"])
    seed_rows = []
    for (method, scene, intensity, conflict, rho, seed), grp in seed_groups:
        n_eps = len(grp)
        row = {
            "method": method,
            "scene_id": scene,
            "intensity": intensity,
            "conflict": conflict,
            "rho": rho,
            "seed": seed,
            "n_episodes": n_eps,
            "R_task_mean": grp["R_task"].mean(),
            "R_task_std": grp["R_task"].std(ddof=1) if n_eps > 1 else 0.0,
            "R_task_sem": grp["R_task"].sem() if n_eps > 1 else 0.0,
            "R_cov_mean": grp["R_cov"].mean(),
            "R_cov_std": grp["R_cov"].std(ddof=1) if n_eps > 1 else 0.0,
            "R_fail_given_cov_mean": grp["R_fail_given_cov"].mean(),
            "delivery_latency_mean_s": grp["delivery_latency_mean_s"].mean(),
            "delivery_latency_median_s": grp["delivery_latency_median_s"].mean(),
            "delivery_latency_p90_s": grp["delivery_latency_p90_s"].mean(),
            "returned_home_rate": grp["returned_home"].mean(),
            "oob_rate": grp["oob_triggered"].mean(),
            "energy_rate": grp["energy_triggered"].mean(),
            "timeout_rate": grp["timeout_triggered"].mean(),
            "replan_total_mean": grp["replan_total"].mean(),
            "replan_periodic_mean": grp["replan_periodic"].mean(),
            "replan_event_mean": grp["replan_event"].mean(),
            "event_replan_ratio_mean": grp["event_replan_ratio"].mean(),
            "feedback_total_mean": grp["feedback_total"].mean(),
            "target_switch_count_mean": grp["target_switch_count"].mean(),
            "mode_switch_count_mean": grp["mode_switch_count"].mean(),
            "fast_time_mean_ms": grp["fast_time_mean_ms"].mean(),
            "fast_time_p95_ms": grp["fast_time_p95_ms"].mean(),
            "slow_time_mean_ms": grp["slow_time_mean_ms"].mean(),
            "slow_time_p95_ms": grp["slow_time_p95_ms"].mean(),
            "solver_timeout_count_mean": grp["solver_timeout_count"].mean(),
            "mip_gap_mean": grp["mip_gap_mean"].mean(),
            "episode_wall_time_s_mean": grp["episode_wall_time_s"].mean(),
            "final_energy_mean": grp["final_energy"].mean(),
            "link_quality_mean": grp["link_quality_mean"].mean(),
            "loss_mean": grp["loss_mean"].mean(),
            "steps_mean": grp["steps"].mean(),
        }
        seed_rows.append(row)

    df_seed = pd.DataFrame(seed_rows)
    df_seed.to_csv(OUT_DIR / "paper1_v2_seed_analysis.csv", index=False)
    print(f"Seed CSV: {len(df_seed)} rows → {OUT_DIR / 'paper1_v2_seed_analysis.csv'}")

    # ── scene aggregation ───────────────────────────────────────
    scene_groups = df_seed.groupby(["method", "scene_id", "intensity", "conflict", "rho"])
    scene_rows = []
    for (method, scene, intensity, conflict, rho), grp in scene_groups:
        n_seeds = len(grp)
        row = {
            "method": method,
            "scene_id": scene,
            "intensity": intensity,
            "conflict": conflict,
            "rho": rho,
            "n_seeds": n_seeds,
            "n_episodes_total": int(grp["n_episodes"].sum()),
            "R_task_mean": grp["R_task_mean"].mean(),
            "R_task_sem": grp["R_task_mean"].sem() if n_seeds > 1 else 0.0,
            "R_task_std": grp["R_task_mean"].std(ddof=1) if n_seeds > 1 else 0.0,
            "R_cov_mean": grp["R_cov_mean"].mean(),
            "R_cov_sem": grp["R_cov_mean"].sem() if n_seeds > 1 else 0.0,
            "R_fail_given_cov_mean": grp["R_fail_given_cov_mean"].mean(),
            "P_del_given_cov_mean": 1.0 - grp["R_fail_given_cov_mean"].mean() if grp["R_fail_given_cov_mean"].notna().any() else None,
            "delivery_latency_mean_s": grp["delivery_latency_mean_s"].mean(),
            "delivery_latency_p90_s": grp["delivery_latency_p90_s"].mean(),
            "returned_home_rate": grp["returned_home_rate"].mean(),
            "oob_rate": grp["oob_rate"].mean(),
            "energy_rate": grp["energy_rate"].mean(),
            "timeout_rate": grp["timeout_rate"].mean(),
            "replan_total_mean": grp["replan_total_mean"].mean(),
            "replan_event_mean": grp["replan_event_mean"].mean(),
            "event_replan_ratio_mean": grp["event_replan_ratio_mean"].mean(),
            "feedback_total_mean": grp["feedback_total_mean"].mean(),
            "target_switch_count_mean": grp["target_switch_count_mean"].mean(),
            "mode_switch_count_mean": grp["mode_switch_count_mean"].mean(),
            "fast_time_mean_ms": grp["fast_time_mean_ms"].mean(),
            "fast_time_p95_ms": grp["fast_time_p95_ms"].mean(),
            "slow_time_mean_ms": grp["slow_time_mean_ms"].mean(),
            "slow_time_p95_ms": grp["slow_time_p95_ms"].mean(),
            "mip_gap_mean": grp["mip_gap_mean"].mean(),
            "episode_wall_time_s_mean": grp["episode_wall_time_s_mean"].mean(),
            "final_energy_mean": grp["final_energy_mean"].mean(),
            "link_quality_mean": grp["link_quality_mean"].mean(),
        }
        scene_rows.append(row)

    df_scene = pd.DataFrame(scene_rows)
    df_scene.to_csv(OUT_DIR / "paper1_v2_scene_analysis.csv", index=False)
    print(f"Scene CSV: {len(df_scene)} rows → {OUT_DIR / 'paper1_v2_scene_analysis.csv'}")

    # ── protocol report ─────────────────────────────────────────
    n_methods = df_seed["method"].nunique()
    n_scenes = df_seed["scene_id"].nunique()
    n_seeds_total = len(df_seed)
    n_episodes_total = len(df_ep)
    seed_sizes = df_seed.groupby(["method", "scene_id"])["n_episodes"].unique()

    report_lines = [
        "# Stage 1 + Stage 2 Protocol Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        f"- Methods: {n_methods}",
        f"- Scenes: {n_scenes}",
        f"- Total seeds: {n_seeds_total}",
        f"- Total episodes: {n_episodes_total}",
        f"- Seeds per combo: {df_seed.groupby(['method','scene_id']).size().min()}–{df_seed.groupby(['method','scene_id']).size().max()}",
        f"- Episodes per seed: {df_seed['n_episodes'].min()}–{df_seed['n_episodes'].max()}",
        "",
        "## Methods",
    ]
    for m in sorted(df_seed["method"].unique()):
        n = len(df_seed[df_seed["method"] == m])
        report_lines.append(f"- {m}: {n} seeds")

    report_lines += [
        "",
        "## Protocol vs Chapter 4",
        f"- Chapter 4 spec: 10 seeds × 10 episodes = 100 episodes/combo",
        f"- Actual: {int(df_seed['n_episodes'].median())} seeds × {int(df_seed['n_episodes'].mode().iloc[0])} episodes = {n_episodes_total // n_seeds_total if n_seeds_total else 0} episodes/combo avg",
        "- ⚠ Protocol mismatch: fewer seeds, more episodes per seed. Seed is still the independent statistical unit.",
    ]

    report_path = OUT_DIR / "paper1_v2_protocol_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"Protocol report: {report_path}")

    # ── quick stats ─────────────────────────────────────────────
    print("\n=== Quick Summary ===")
    print(f"Methods: {sorted(df_seed['method'].unique())}")
    print(f"Scenes: {sorted(df_seed['scene_id'].unique())}")
    print(f"\nR_task by method (grand mean ± SEM):")
    for m in sorted(df_seed["method"].unique()):
        sub = df_seed[df_seed["method"] == m]
        print(f"  {m:20s}: {sub['R_task_mean'].mean():.4f} ± {sub['R_task_mean'].sem():.4f}")


if __name__ == "__main__":
    main()
