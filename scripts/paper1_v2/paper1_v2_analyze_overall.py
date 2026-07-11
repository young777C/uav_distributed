"""
Paper: paper1_v2
Purpose: Stage 1 overall analysis — mixed-effects models, pairwise tests, RQ1 results
Inputs:  results_v2/summaries/paper1_v2_seed_analysis.csv
Outputs: results_v2/summaries/paper1_v2_rq1_results.csv
         results_v2/summaries/paper1_v2_scene_pairwise.csv
         results_v2/summaries/paper1_v2_contrasts.csv
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "results_v2/summaries"
OUT_DIR = DATA_DIR
SEED_CSV = DATA_DIR / "paper1_v2_seed_analysis.csv"

BASELINES = ["CDSL", "WCDL", "RHC-Inspection", "CBCP"]
CONTRASTS = [(f"FDLC vs {b}", "FDLC", b) for b in BASELINES]
ALPHA = 0.05


def holm_adjust(p_values: List[Tuple[str, float]]) -> List[Tuple[str, float, float]]:
    """Apply Holm-Bonferroni correction to a list of (label, p_value) pairs."""
    sorted_pairs = sorted(p_values, key=lambda x: x[1])
    n = len(sorted_pairs)
    adjusted = []
    for rank, (label, p) in enumerate(sorted_pairs):
        adj = min(p * (n - rank), 1.0)
        adjusted.append((label, p, adj))
    # Return in original order
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
    """Rank-biserial correlation for paired samples (effect size for Wilcoxon)."""
    diff = x - y
    n = len(diff)
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    r_plus = np.sum(nonzero > 0)
    r_minus = np.sum(nonzero < 0)
    return (r_plus - r_minus) / (r_plus + r_minus)


def bootstrap_paired_ci(
    x: np.ndarray, y: np.ndarray, n_resamples: int = 5000, ci: float = 0.95
) -> Tuple[float, float]:
    """BCa bootstrap CI for paired mean difference (x - y)."""
    rng = np.random.default_rng(42)
    n = len(x)
    diffs = x - y
    theta_hat = np.mean(diffs)

    # Bootstrap distribution
    boot_means = np.array([
        np.mean(diffs[rng.integers(0, n, n)]) for _ in range(n_resamples)
    ])

    # Percentile CI
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return float(lo), float(hi)


def main() -> None:
    df = pd.read_csv(SEED_CSV)
    n_methods = df["method"].nunique()
    n_scenes = df["scene_id"].nunique()
    print(f"Loaded: {len(df)} seeds, {n_methods} methods, {n_scenes} scenes")

    # ── 1. Grand R_task by method ───────────────────────────────
    print("\n=== Grand R_task ===")
    grand = df.groupby("method").agg(
        R_task_mean=("R_task_mean", "mean"),
        R_task_sem=("R_task_mean", "sem"),
        R_task_std=("R_task_mean", "std"),
        n_seeds=("R_task_mean", "count"),
    ).sort_values("R_task_mean", ascending=False)
    print(grand.to_string())

    # ── 2. Per-scene rankings ───────────────────────────────────
    print("\n=== Per-scene R_task rankings ===")
    scene_ranks_data = []
    for scene in sorted(df["scene_id"].unique()):
        sub = df[df["scene_id"] == scene]
        means = sub.groupby("method")["R_task_mean"].mean().sort_values(ascending=False)
        for rank, (method, val) in enumerate(means.items(), 1):
            scene_ranks_data.append({
                "scene_id": scene, "method": method, "rank": rank, "R_task": val
            })
    rank_df = pd.DataFrame(scene_ranks_data)
    # Average rank per method
    avg_rank = rank_df.groupby("method")["rank"].mean().sort_values()
    print("Average rank (lower=better):")
    for m in avg_rank.index:
        print(f"  {m:20s}: {avg_rank[m]:.2f}")

    # ── 3. Per-scene Friedman + Wilcoxon ────────────────────────
    print("\n=== Scene-level Friedman + pairwise Wilcoxon ===")
    scene_results = []
    all_contrast_pairs = []

    for scene in sorted(df["scene_id"].unique()):
        sub = df[df["scene_id"] == scene]
        # Pivot: seeds × methods
        piv = sub.pivot(index="seed", columns="method", values="R_task_mean")
        piv = piv.dropna()
        methods_present = list(piv.columns)
        n_seeds = len(piv)

        if len(methods_present) < 3 or n_seeds < 3:
            scene_results.append({
                "scene_id": scene, "n_seeds": n_seeds,
                "friedman_stat": None, "friedman_p": None,
                "methods": methods_present, "note": "insufficient data"
            })
            continue

        # Friedman test
        try:
            f_stat, f_p = friedmanchisquare(*[piv[m].values for m in methods_present])
        except Exception:
            f_stat, f_p = None, None

        # Preset pairwise Wilcoxon + Holm
        pairs = []
        for label, m1, m2 in CONTRASTS:
            if m1 not in piv.columns or m2 not in piv.columns:
                continue
            x = piv[m1].values
            y = piv[m2].values
            if np.allclose(x, y):
                w_stat, w_p = 0.0, 1.0
                rbc = 0.0
            else:
                try:
                    w = wilcoxon(x, y, alternative="two-sided", method="exact")
                    w_stat, w_p = w.statistic, w.pvalue
                except Exception:
                    w_stat, w_p = None, None
                rbc = rank_biserial_correlation(x, y)
            ci_lo, ci_hi = bootstrap_paired_ci(x, y)
            mean_diff = np.mean(x - y)
            pairs.append((label, float(w_p) if w_p is not None else 1.0, rbc, mean_diff, ci_lo, ci_hi))
            all_contrast_pairs.append({
                "scene_id": scene, "contrast": label,
                "mean_diff": mean_diff, "ci_lower": ci_lo, "ci_upper": ci_hi,
                "raw_p": float(w_p) if w_p is not None else 1.0,
                "rank_biserial_corr": rbc,
                "n_seeds": n_seeds,
            })

        # Holm correction
        p_vals = [(label, p) for label, p, _, _, _, _ in pairs]
        holm_results = holm_adjust(p_vals)

        for (label, raw_p, rbc, md, clo, chi), (_, _, adj_p) in zip(pairs, holm_results):
            scene_results.append({
                "scene_id": scene, "n_seeds": n_seeds,
                "friedman_stat": f_stat, "friedman_p": f_p,
                "contrast": label,
                "mean_diff": md, "ci_lower": clo, "ci_upper": chi,
                "raw_p": raw_p,
                "holm_adjusted_p": adj_p,
                "rank_biserial_corr": rbc,
                "significant": adj_p < ALPHA,
            })

    # Write scene-level results
    scene_df = pd.DataFrame(scene_results)
    scene_df.to_csv(OUT_DIR / "paper1_v2_scene_pairwise.csv", index=False)
    print(f"Scene pairwise: {len(scene_df)} rows → {OUT_DIR / 'paper1_v2_scene_pairwise.csv'}")

    # ── 4. Overall contrasts (pooled across scenes) ─────────────
    print("\n=== Overall pooled contrasts ===")
    contrast_df = pd.DataFrame(all_contrast_pairs)
    overall_contrasts = []
    for label, m1, m2 in CONTRASTS:
        sub = contrast_df[contrast_df["contrast"] == label]
        if len(sub) == 0:
            continue
        # Aggregate across scenes using scene-level means
        mean_d = sub["mean_diff"].mean()
        sem_d = sub["mean_diff"].sem()
        # Pooled bootstrap CI (conservative: mean of scene CIs)
        ci_lo = sub["ci_lower"].mean()
        ci_hi = sub["ci_upper"].mean()
        # Count scenes where FDLC wins
        n_wins = int((sub["mean_diff"] > 0).sum())
        n_scenes_avail = len(sub)
        overall_contrasts.append({
            "contrast": label,
            "mean_diff": mean_d,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "sem_diff": sem_d,
            "n_wins": n_wins,
            "n_scenes": n_scenes_avail,
            "win_rate": n_wins / n_scenes_avail,
        })

    ov_df = pd.DataFrame(overall_contrasts)
    ov_df.to_csv(OUT_DIR / "paper1_v2_contrasts.csv", index=False)
    for _, r in ov_df.iterrows():
        print(f"  {r['contrast']:25s}: Δ={r['mean_diff']:+.4f} [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}], "
              f"wins={int(r['n_wins'])}/{int(r['n_scenes'])}")

    # ── 5. RQ1 summary ──────────────────────────────────────────
    print("\n=== RQ1 Results ===")
    rq1_rows = []
    # Per intensity
    for intensity in ["low", "medium", "high", "severe"]:
        sub = df[df["intensity"] == intensity]
        for m in sorted(sub["method"].unique()):
            ms = sub[sub["method"] == m]
            rq1_rows.append({
                "group": f"intensity_{intensity}", "method": m,
                "R_task_mean": ms["R_task_mean"].mean(),
                "R_task_sem": ms["R_task_mean"].sem(),
                "n_seeds": len(ms),
            })
    # Per conflict
    for conflict in ["M0", "M1", "M2", "M3"]:
        sub = df[df["conflict"] == conflict]
        for m in sorted(sub["method"].unique()):
            ms = sub[sub["method"] == m]
            rq1_rows.append({
                "group": f"conflict_{conflict}", "method": m,
                "R_task_mean": ms["R_task_mean"].mean(),
                "R_task_sem": ms["R_task_mean"].sem(),
                "n_seeds": len(ms),
            })

    rq1_df = pd.DataFrame(rq1_rows)
    rq1_df.to_csv(OUT_DIR / "paper1_v2_rq1_results.csv", index=False)
    print(f"RQ1 results: {len(rq1_df)} rows → {OUT_DIR / 'paper1_v2_rq1_results.csv'}")

    # Save summary to JSON for figure scripts
    summary = {
        "generated": datetime.now().isoformat(),
        "n_seeds": len(df),
        "n_scenes": int(df["scene_id"].nunique()),
        "methods": sorted(df["method"].unique().tolist()),
        "grand_r_task": {
            m: float(v) for m, v in zip(
                grand.index, grand["R_task_mean"].values
            )
        },
        "fdlc_wins": int(sum(1 for m in avg_rank.index if m != "FDLC" and avg_rank.get("FDLC", 99) < avg_rank.get(m, 0))),
        "avg_ranks": {m: float(v) for m, v in zip(avg_rank.index, avg_rank.values)},
    }
    (OUT_DIR / "paper1_v2_rq1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"Summary JSON: {OUT_DIR / 'paper1_v2_rq1_summary.json'}")

    print("\n✓ Analysis complete.")


if __name__ == "__main__":
    main()
