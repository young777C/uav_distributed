"""
Paper: paper1_v2
Purpose: Stage 2 coupling ablation analysis — planned contrasts, per-scene tests,
         decomposition of incremental contributions (Event-driven, Mode-switching).
Inputs:  results_v2/summaries/paper1_v2_seed_analysis.csv
Outputs: results_v2/summaries/paper1_v2_coupling_contrasts.csv
         results_v2/summaries/paper1_v2_coupling_scene_pairwise.csv
         results_v2/summaries/paper1_v2_coupling_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "results_v2/summaries"
OUT_DIR = DATA_DIR
SEED_CSV = DATA_DIR / "paper1_v2_seed_analysis.csv"

# ── Coupling ablation config ──────────────────────────────────────────
COUPLING_STRATEGIES = ["Periodic Goal", "Event-driven Goal", "FDLC"]

# Predetermined high-stress subset per skill §7.4
HIGH_STRESS_SCENES = [
    "c3_high_m2", "c3_high_m3", "c3_severe_m2", "c3_severe_m3",
]

COUPLING_CONTRASTS = [
    ("Event-driven Goal vs Periodic Goal", "Event-driven Goal", "Periodic Goal"),
    ("FDLC vs Event-driven Goal", "FDLC", "Event-driven Goal"),
    ("FDLC vs Periodic Goal", "FDLC", "Periodic Goal"),
]

ALPHA = 0.05
N_BOOT = 5000


# ── Helpers ───────────────────────────────────────────────────────────

def holm_adjust(p_values: List[Tuple[str, float]]) -> List[Tuple[str, float, float]]:
    sorted_pairs = sorted(p_values, key=lambda x: x[1])
    n = len(sorted_pairs)
    adjusted = []
    for rank, (label, p) in enumerate(sorted_pairs):
        adj = min(p * (n - rank), 1.0)
        adjusted.append((label, p, adj))
    result = []
    for label, raw_p in p_values:
        for l2, rp, ap in adjusted:
            if l2 == label and abs(rp - raw_p) < 1e-15:
                result.append((label, raw_p, ap))
                break
        else:
            result.append((label, raw_p, min(raw_p * n, 1.0)))
    return result


def rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    r_plus = np.sum(nonzero > 0)
    r_minus = np.sum(nonzero < 0)
    return (r_plus - r_minus) / (r_plus + r_minus)


def bootstrap_paired_ci(
    x: np.ndarray, y: np.ndarray, n_resamples: int = N_BOOT, ci: float = 0.95,
) -> Tuple[float, float]:
    """Seed-level paired bootstrap CI for mean difference (x - y)."""
    rng = np.random.default_rng(42)
    n = len(x)
    diffs = x - y
    boot_means = np.array([
        np.mean(diffs[rng.integers(0, n, n)]) for _ in range(n_resamples)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return float(lo), float(hi)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(SEED_CSV)

    # Filter to coupling-relevant methods only
    coupling_df = df[df["method"].isin(COUPLING_STRATEGIES)].copy()
    print(f"Coupling subset: {len(coupling_df)} seeds, "
          f"{coupling_df['method'].nunique()} strategies, "
          f"{coupling_df['scene_id'].nunique()} scenes")

    # ── 1. Grand means by strategy ─────────────────────────────────
    print("\n=== Grand R_task by Strategy ===")
    grand = coupling_df.groupby("method").agg(
        R_task_mean=("R_task_mean", "mean"),
        R_task_sem=("R_task_mean", "sem"),
        R_task_std=("R_task_mean", "std"),
        n_seeds=("R_task_mean", "count"),
    ).sort_values("R_task_mean", ascending=False)
    print(grand.to_string())

    # ── 2. Per-scene R_task table ──────────────────────────────────
    print("\n=== Per-scene R_task (seed mean ± SEM) ===")
    scene_table = coupling_df.pivot_table(
        index="scene_id", columns="method", values="R_task_mean",
        aggfunc=["mean", "sem"],
    )
    for scene in sorted(coupling_df["scene_id"].unique()):
        line = f"  {scene:20s}"
        for strat in COUPLING_STRATEGIES:
            m = coupling_df[(coupling_df["scene_id"] == scene) &
                            (coupling_df["method"] == strat)]["R_task_mean"]
            if len(m) > 0:
                line += f" | {strat}: {m.mean():.4f}±{m.sem():.4f}"
            else:
                line += f" | {strat}: N/A"
        print(line)

    # ── 3. Planned contrasts per scene ─────────────────────────────
    print("\n=== Per-scene planned contrasts (Wilcoxon paired, seed-level) ===")
    all_contrast_rows = []
    scene_contrast_summary = []

    for scene in sorted(coupling_df["scene_id"].unique()):
        sub = coupling_df[coupling_df["scene_id"] == scene]
        piv = sub.pivot(index="seed", columns="method", values="R_task_mean")
        piv = piv.dropna()
        n_seeds = len(piv)

        if len(piv.columns) < 2 or n_seeds < 3:
            continue

        # Friedman test
        try:
            f_stat, f_p = friedmanchisquare(
                *[piv[s].values for s in COUPLING_STRATEGIES if s in piv.columns]
            )
        except Exception:
            f_stat, f_p = None, None

        pairs = []
        for label, m1, m2 in COUPLING_CONTRASTS:
            if m1 not in piv.columns or m2 not in piv.columns:
                continue
            x = piv[m1].values
            y = piv[m2].values
            mean_diff = np.mean(x - y)

            if np.allclose(x, y):
                w_p, rbc = 1.0, 0.0
            else:
                try:
                    w = wilcoxon(x, y, alternative="two-sided", method="exact")
                    w_p = float(w.pvalue)
                except Exception:
                    w_p = None
                rbc = rank_biserial_correlation(x, y)

            ci_lo, ci_hi = bootstrap_paired_ci(x, y)
            pairs.append((label, w_p if w_p is not None else 1.0, rbc, mean_diff, ci_lo, ci_hi))

            all_contrast_rows.append({
                "scene_id": scene,
                "contrast": label,
                "mean_diff": mean_diff,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "raw_p": w_p if w_p is not None else 1.0,
                "rank_biserial_corr": rbc,
                "n_seeds": n_seeds,
                "high_stress": scene in HIGH_STRESS_SCENES,
            })

        # Holm correction for this scene
        if pairs:
            p_vals = [(label, p) for label, p, _, _, _, _ in pairs]
            holm_results = holm_adjust(p_vals)
            for (label, raw_p, rbc, md, clo, chi), (_, _, adj_p) in zip(pairs, holm_results):
                scene_contrast_summary.append({
                    "scene_id": scene,
                    "n_seeds": n_seeds,
                    "friedman_stat": f_stat,
                    "friedman_p": f_p,
                    "contrast": label,
                    "mean_diff": md,
                    "ci_lower": clo,
                    "ci_upper": chi,
                    "raw_p": raw_p,
                    "holm_adjusted_p": adj_p,
                    "rank_biserial_corr": rbc,
                    "significant": adj_p < ALPHA,
                    "high_stress": scene in HIGH_STRESS_SCENES,
                })

    # Write per-scene pairwise
    scene_df = pd.DataFrame(scene_contrast_summary)
    scene_df.to_csv(OUT_DIR / "paper1_v2_coupling_scene_pairwise.csv", index=False)
    print(f"Scene pairwise: {len(scene_df)} rows → "
          f"{OUT_DIR / 'paper1_v2_coupling_scene_pairwise.csv'}")

    # ── 4. Overall pooled contrasts ─────────────────────────────────
    print("\n=== Overall pooled coupling contrasts ===")
    contrast_df = pd.DataFrame(all_contrast_rows)
    overall_rows = []

    for label, m1, m2 in COUPLING_CONTRASTS:
        sub = contrast_df[contrast_df["contrast"] == label]
        if len(sub) == 0:
            continue
        mean_d = sub["mean_diff"].mean()
        ci_lo = sub["mean_diff"].mean() - sub["mean_diff"].sem() * 1.96  # crude CI across scenes
        ci_hi = sub["mean_diff"].mean() + sub["mean_diff"].sem() * 1.96
        n_wins = int((sub["mean_diff"] > 0).sum())
        n_scenes = len(sub)

        # High-stress subset
        hs = sub[sub["high_stress"] == True]
        hs_mean_d = hs["mean_diff"].mean() if len(hs) > 0 else None
        hs_n_wins = int((hs["mean_diff"] > 0).sum()) if len(hs) > 0 else 0

        overall_rows.append({
            "contrast": label,
            "mean_diff": mean_d,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_wins": n_wins,
            "n_scenes": n_scenes,
            "win_rate": n_wins / n_scenes if n_scenes > 0 else 0,
            "high_stress_mean_diff": hs_mean_d,
            "high_stress_wins": hs_n_wins,
            "high_stress_n": len(hs),
        })

        print(f"  {label:35s}: Δ={mean_d:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}], "
              f"wins={n_wins}/{n_scenes}, hs_wins={hs_n_wins}/{len(hs)}")

    ov_df = pd.DataFrame(overall_rows)
    ov_df.to_csv(OUT_DIR / "paper1_v2_coupling_contrasts.csv", index=False)
    print(f"Overall contrasts: {len(ov_df)} rows → "
          f"{OUT_DIR / 'paper1_v2_coupling_contrasts.csv'}")

    # ── 5. Auxiliary metrics: replan, feedback ─────────────────────
    print("\n=== Coordination overhead by strategy ===")
    coord_metrics = ["replan_total_mean", "replan_event_mean",
                     "event_replan_ratio_mean", "feedback_total_mean",
                     "target_switch_count_mean", "mode_switch_count_mean"]
    for metric in coord_metrics:
        if metric in coupling_df.columns:
            print(f"\n  {metric}:")
            for strat in COUPLING_STRATEGIES:
                vals = coupling_df[coupling_df["method"] == strat][metric].dropna()
                if len(vals) > 0:
                    print(f"    {strat:20s}: {vals.mean():.2f} ± {vals.sem():.2f}")

    # ── 6. Decomposition: incremental gains ────────────────────────
    print("\n=== Incremental decomposition ===")
    periodic_mean = grand.loc["Periodic Goal", "R_task_mean"]
    event_mean = grand.loc["Event-driven Goal", "R_task_mean"]
    fdlc_mean = grand.loc["FDLC", "R_task_mean"]

    gain_event = event_mean - periodic_mean
    gain_mode = fdlc_mean - event_mean
    gain_total = fdlc_mean - periodic_mean

    print(f"  Periodic Goal base:          {periodic_mean:.4f}")
    print(f"  + Event-driven update:       {gain_event:+.4f} ({gain_event/gain_total*100:.0f}% of total)")
    print(f"  + Mode switching (FDLC):      {gain_mode:+.4f} ({gain_mode/gain_total*100:.0f}% of total)")
    print(f"  = FDLC total:                 {fdlc_mean:.4f}")
    print(f"  Total gain over Periodic:     {gain_total:+.4f}")

    # ── 7. Failed/negative results ─────────────────────────────────
    print("\n=== Scenes where FDLC ≤ Event-driven Goal ===")
    for scene in sorted(coupling_df["scene_id"].unique()):
        sub = coupling_df[coupling_df["scene_id"] == scene]
        piv = sub.pivot(index="seed", columns="method", values="R_task_mean")
        if "FDLC" not in piv.columns or "Event-driven Goal" not in piv.columns:
            continue
        diff = piv["FDLC"].mean() - piv["Event-driven Goal"].mean()
        if diff <= 0:
            print(f"  {scene}: FDLC - Event-driven = {diff:+.4f}")

    print("\n=== Scenes where Event-driven ≤ Periodic Goal ===")
    for scene in sorted(coupling_df["scene_id"].unique()):
        sub = coupling_df[coupling_df["scene_id"] == scene]
        piv = sub.pivot(index="seed", columns="method", values="R_task_mean")
        if "Event-driven Goal" not in piv.columns or "Periodic Goal" not in piv.columns:
            continue
        diff = piv["Event-driven Goal"].mean() - piv["Periodic Goal"].mean()
        if diff <= 0:
            print(f"  {scene}: Event-driven - Periodic = {diff:+.4f}")

    # ── 8. Summary JSON ────────────────────────────────────────────
    summary = {
        "generated": datetime.now().isoformat(),
        "n_seeds_total": len(coupling_df),
        "n_scenes": int(coupling_df["scene_id"].nunique()),
        "strategies": COUPLING_STRATEGIES,
        "grand_r_task": {m: float(v) for m, v in zip(grand.index, grand["R_task_mean"].values)},
        "incremental_gains": {
            "periodic_base": float(periodic_mean),
            "event_driven_gain": float(gain_event),
            "mode_switch_gain": float(gain_mode),
            "total_gain": float(gain_total),
            "event_pct": float(gain_event / gain_total * 100) if gain_total != 0 else 0,
            "mode_pct": float(gain_mode / gain_total * 100) if gain_total != 0 else 0,
        },
        "overall_contrasts": overall_rows,
    }
    (OUT_DIR / "paper1_v2_coupling_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\nSummary JSON: {OUT_DIR / 'paper1_v2_coupling_summary.json'}")

    print("\n✓ Coupling ablation analysis complete.")


if __name__ == "__main__":
    main()
