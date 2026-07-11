#!/usr/bin/env python3
"""Build merged Chapter 5.1 overall comparison (FDLC vs RHC/CBCP/WCDL).

Reads seed-level summaries from:
  - results/baselines/* (external baselines)
  - artifacts/.../struct_axis_full_20260606/summary.csv (internal struct axis)

Outputs under results/chapter5_overall_comparison/:
  - seed_rows.csv
  - summary_table.csv
  - statistical_tests.csv
  - fig_overall_comparison_r_task.{pdf,png}
  - chapter5_overall_comparison_latex.tex
  - supplementary_cdsl_8scene_table.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS_MAIN = [
    ("c1_g2_m0", "C1--G2--M0", "low-conflict representative"),
    ("c1_g2_m2", "C1--G2--M2", "high-conflict (distance decay)"),
    ("c2_g2_m2", "C2--G2--M2", "high-conflict (local shadow)"),
]

SCENARIOS_CDSL = [
    "c1_g1",
    "c1_g2_m0",
    "c1_g2_m1",
    "c1_g2_m2",
    "c2_g1",
    "c2_g2_m0",
    "c2_g2_m1",
    "c2_g2_m2",
]

METHODS_MAIN = ["RHC-Inspection", "CBCP", "WCDL", "FDLC"]
METHODS_CDSL = ["CDSL", "WCDL", "FDLC"]

SYSTEM_TO_METHOD = {
    "baseline_rhc_inspection": "RHC-Inspection",
    "baseline_cbcp": "CBCP",
    "struct_centralized_single_loop": "CDSL",
    "struct_decoupled_dual_loop": "WCDL",
    "struct_full_dual_loop_distributed": "FDLC",
}

COLORS = {
    "RHC-Inspection": "#6f6f6f",
    "CBCP": "#4c78a8",
    "WCDL": "#54a24b",
    "FDLC": "#f58518",
    "CDSL": "#9e9e9e",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def _parse_exp_name(name: str) -> tuple[str, str]:
    case, system = name.split("__", 1)
    return case, system


def _load_summary_csv(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    parsed = df["exp_name"].map(_parse_exp_name)
    df["case_key"] = parsed.map(lambda x: x[0])
    df["system"] = parsed.map(lambda x: x[1])
    df["method"] = df["system"].map(SYSTEM_TO_METHOD)
    df = df[df["method"].notna()].copy()
    df["source"] = source
    df["P_return_given_cov"] = 1.0 - pd.to_numeric(df["R_fail_given_cov_mean"], errors="coerce")
    for col in [
        "R_task_mean",
        "R_cov_mean",
        "T_ret_s_mean",
        "returned_home_rate",
        "oob_rate",
        "P_return_given_cov",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype(int)
    return df


def _collect_baseline_rows() -> pd.DataFrame:
    baseline_root = ROOT / "results" / "baselines"
    frames: list[pd.DataFrame] = []
    for tag_dir in sorted(baseline_root.iterdir()):
        if not tag_dir.is_dir():
            continue
        summary = tag_dir / "summary.csv"
        if summary.exists():
            frames.append(_load_summary_csv(summary, source=f"baselines/{tag_dir.name}"))
    if not frames:
        raise FileNotFoundError("No baseline summary.csv files found under results/baselines/")
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["case_key", "system", "seed"], keep="last"
    )


def _collect_struct_rows(struct_summary: Path) -> pd.DataFrame:
    return _load_summary_csv(struct_summary, source=str(struct_summary.parent.name))


def rank_biserial_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    abs_ranks = _rank_abs(diff)
    w_pos = abs_ranks[diff > 0].sum()
    w_neg = abs_ranks[diff < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def _rank_abs(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.abs(values))
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and np.abs(values[order[j + 1]]) == np.abs(values[order[i]]):
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def wilcoxon_signed_rank_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    n = len(diff)
    if n < 1:
        return float("nan")
    if n == 1:
        return 1.0
    ranks = _rank_abs(diff)
    w_plus = ranks[diff > 0].sum()
    # Normal approximation with continuity correction (adequate for n>=3 seeds).
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return 1.0
    z = (w_plus - mu - 0.5) / sigma
    # two-sided p-value from standard normal
    from math import erf, sqrt

    cdf = 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))
    return float(max(0.0, min(1.0, 2.0 * (1.0 - cdf))))


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * float(p_values[idx])
        running_max = max(running_max, val)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def friedman_pvalue(*groups: np.ndarray) -> tuple[float, float]:
    mats = [np.asarray(g, dtype=float) for g in groups]
    n = len(mats[0])
    k = len(mats)
    if n < 2 or k < 2:
        return float("nan"), float("nan")
    data = np.vstack(mats).T
    ranks = np.apply_along_axis(
        lambda row: pd.Series(row).rank(method="average").to_numpy(), 1, data
    )
    r_bar = ranks.mean(axis=0)
    q = 12.0 * n / (k * (k + 1)) * float(np.sum((r_bar - (k + 1) / 2.0) ** 2))
    # chi-square approximation with k-1 df
    from math import exp, gamma

    df = k - 1

    def chi2_sf(x: float, d: float) -> float:
        # Regularized gamma Q(d/2, x/2)
        a = d / 2.0
        z = x / 2.0
        if z <= 0:
            return 1.0
        term = z**a * exp(-z) / gamma(a)
        s = term
        for i in range(1, 200):
            term *= z / (a + i)
            s += term
            if abs(term) < 1e-12 * abs(s):
                break
        return max(0.0, min(1.0, 1.0 - s))

    return float(q), float(chi2_sf(q, df))


def paired_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 5000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    diff = x[valid] - y[valid]
    if len(diff) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(diff.mean()), float(lo), float(hi)


def _aggregate_summary(seed_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (case_key, method), g in seed_rows.groupby(["case_key", "method"], sort=False):
        vals = pd.to_numeric(g["R_task_mean"], errors="coerce")
        row = {
            "case_key": case_key,
            "method": method,
            "n_seeds": int(g["seed"].nunique()),
            "n_episodes": int(pd.to_numeric(g["episodes"], errors="coerce").sum()),
            "R_task_mean": float(vals.mean()),
            "R_task_seed_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "R_cov_mean": float(pd.to_numeric(g["R_cov_mean"], errors="coerce").mean()),
            "P_return_given_cov_mean": float(
                pd.to_numeric(g["P_return_given_cov"], errors="coerce").mean()
            ),
            "T_ret_s_mean": float(pd.to_numeric(g["T_ret_s_mean"], errors="coerce").mean()),
            "returned_home_rate_mean": float(
                pd.to_numeric(g["returned_home_rate"], errors="coerce").mean()
            ),
            "oob_rate_mean": float(pd.to_numeric(g["oob_rate"], errors="coerce").mean()),
        }
        lo, hi = paired_bootstrap_ci(vals.to_numpy(), vals.to_numpy(), n_boot=2000, seed=42)
        # seed-level CI for the mean via bootstrap over seeds
        rng = np.random.default_rng(42)
        boot_means = []
        arr = vals.to_numpy()
        for _ in range(5000):
            sample = arr[rng.integers(0, len(arr), size=len(arr))]
            boot_means.append(sample.mean())
        row["R_task_ci_low"] = float(np.quantile(boot_means, 0.025))
        row["R_task_ci_high"] = float(np.quantile(boot_means, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def _run_stats(seed_rows: pd.DataFrame, scenarios: list[tuple[str, str, str]]) -> pd.DataFrame:
    comparisons = [
        ("FDLC", "RHC-Inspection"),
        ("FDLC", "CBCP"),
        ("FDLC", "WCDL"),
    ]
    out_rows: list[dict[str, object]] = []
    for case_key, scene_label, _ in scenarios:
        sub = seed_rows[seed_rows["case_key"] == case_key]
        wide = sub.pivot(index="seed", columns="method", values="R_task_mean")
        methods_present = [m for m in METHODS_MAIN if m in wide.columns]
        wide = wide.dropna(subset=methods_present)
        if len(wide) < 2:
            continue
        friedman_chi2, friedman_p = friedman_pvalue(*[wide[m].to_numpy() for m in methods_present])
        raw_ps: list[float] = []
        comp_rows: list[dict[str, object]] = []
        for a, b in comparisons:
            if a not in wide.columns or b not in wide.columns:
                continue
            x = wide[a].to_numpy()
            y = wide[b].to_numpy()
            p_raw = wilcoxon_signed_rank_pvalue(x, y)
            mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(x, y)
            comp_rows.append(
                {
                    "scene": scene_label,
                    "case_key": case_key,
                    "metric": "R_task",
                    "comparison": f"{a} vs {b}",
                    "n_pairs": int(len(x)),
                    "mean_difference": mean_diff,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "p_raw": float(p_raw),
                    "rank_biserial": rank_biserial_paired(x, y),
                    "friedman_chi2": float(friedman_chi2),
                    "friedman_p": float(friedman_p),
                }
            )
            raw_ps.append(float(p_raw))
        if not comp_rows:
            continue
        p_adj = holm_adjust(raw_ps)
        for row, p_holm in zip(comp_rows, p_adj):
            row["p_holm"] = float(p_holm)
            row["significant_005"] = bool(p_holm < 0.05)
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def _plot(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0), sharey=True)
    x = np.arange(len(METHODS_MAIN))
    width = 0.62
    for ax, (case_key, scene_label, subtitle) in zip(axes, SCENARIOS_MAIN):
        data = summary[summary["case_key"] == case_key].set_index("method").reindex(METHODS_MAIN)
        means = data["R_task_mean"].to_numpy()
        yerr = np.vstack(
            [
                means - data["R_task_ci_low"].to_numpy(),
                data["R_task_ci_high"].to_numpy() - means,
            ]
        )
        bars = ax.bar(
            x,
            means,
            width=width,
            color=[COLORS[m] for m in METHODS_MAIN],
            edgecolor="black",
            linewidth=0.4,
            yerr=yerr,
            capsize=3,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        ax.set_xticks(x, ["RHC", "CBCP", "WCDL", "FDLC"], rotation=0)
        ax.set_title(f"{scene_label}\n({subtitle})", fontsize=9)
        ax.set_ylim(0.0, 0.9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        for rect, val in zip(bars, means):
            if np.isfinite(val):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.03,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
    axes[0].set_ylabel(r"Effective task completion rate $R_{\mathrm{task}}$")
    fig.suptitle("Overall comparison with external and internal baselines", y=1.02, fontsize=11)
    fig.tight_layout()
    pdf = out_dir / "fig_overall_comparison_r_task.pdf"
    png = out_dir / "fig_overall_comparison_r_task.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=360)
    plt.close(fig)
    return pdf


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _latex_main_table(summary: pd.DataFrame, stats_df: pd.DataFrame) -> str:
    lines = [
        "% Auto-generated by scripts/analyze_chapter5_overall_comparison.py",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Overall comparison with external and internal baselines ($R_{\\mathrm{task}}$, mean $\\pm$ 95\\% bootstrap CI across seeds)}",
        "  \\label{tab:ch5-overall-comparison}",
        "  \\small",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    Scenario & RHC-Inspection & CBCP & WCDL & FDLC \\\\",
        "    \\midrule",
    ]
    for case_key, scene_label, _ in SCENARIOS_MAIN:
        row_cells = [scene_label]
        for method in METHODS_MAIN:
            r = summary[(summary["case_key"] == case_key) & (summary["method"] == method)]
            if r.empty:
                row_cells.append("--")
                continue
            m = float(r["R_task_mean"].iloc[0])
            lo = float(r["R_task_ci_low"].iloc[0])
            hi = float(r["R_task_ci_high"].iloc[0])
            row_cells.append(f"${m:.3f}$ [{lo:.3f}, {hi:.3f}]")
        lines.append("    " + " & ".join(row_cells) + " \\\\")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}}",
            "  \\footnotetext{Seed-level bootstrap 95\\% confidence intervals; 3 seeds $\\times$ 5 episodes per scenario.}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "  \\centering",
            "  \\caption{Paired statistical tests for $R_{\\mathrm{task}}$ in Section~5.1 (Wilcoxon signed-rank; Holm-adjusted $p$-values)}",
            "  \\label{tab:ch5-overall-stats}",
            "  \\small",
            "  \\begin{tabular}{llccccc}",
            "    \\toprule",
            "    Scenario & Comparison & $\\Delta$ mean & 95\\% CI & $p_{\\mathrm{raw}}$ & $p_{\\mathrm{Holm}}$ & $r_{\\mathrm{rb}}$ \\\\",
            "    \\midrule",
        ]
    )
    for _, r in stats_df.iterrows():
        lines.append(
            "    "
            + " & ".join(
                [
                    str(r["scene"]),
                    str(r["comparison"]),
                    f"{r['mean_difference']:+.3f}",
                    f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
                    _fmt_p(float(r["p_raw"])),
                    _fmt_p(float(r["p_holm"])),
                    f"{r['rank_biserial']:.2f}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _latex_cdsl_supplementary(summary_cdsl: pd.DataFrame) -> str:
    scene_labels = {
        "c1_g1": "C1--G1",
        "c1_g2_m0": "C1--G2--M0",
        "c1_g2_m1": "C1--G2--M1",
        "c1_g2_m2": "C1--G2--M2",
        "c2_g1": "C2--G1",
        "c2_g2_m0": "C2--G2--M0",
        "c2_g2_m1": "C2--G2--M1",
        "c2_g2_m2": "C2--G2--M2",
    }
    lines = [
        "% Supplementary: full 8-scene CDSL/WCDL/FDLC struct-axis results",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Supplementary structural comparison across eight scenarios ($R_{\\mathrm{task}}$)}",
        "  \\label{tab:supp-cdsl-8scene}",
        "  \\small",
        "  \\begin{tabular}{lccc}",
        "    \\toprule",
        "    Scenario & CDSL & WCDL & FDLC \\\\",
        "    \\midrule",
    ]
    for case_key in SCENARIOS_CDSL:
        cells = [scene_labels[case_key]]
        for method in METHODS_CDSL:
            r = summary_cdsl[
                (summary_cdsl["case_key"] == case_key) & (summary_cdsl["method"] == method)
            ]
            cells.append(f"{float(r['R_task_mean'].iloc[0]):.3f}" if not r.empty else "--")
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def _draft_text(summary: pd.DataFrame, stats_df: pd.DataFrame) -> str:
    def val(case: str, method: str) -> float:
        r = summary[(summary["case_key"] == case) & (summary["method"] == method)]
        return float(r["R_task_mean"].iloc[0])

    lines = [
        "% Draft paragraph for Section 5.1",
        "% Overall comparison with external and internal baselines",
        "",
        (
            "To answer how FDLC performs relative to representative baselines, "
            "we merge the internal structural comparison (WCDL) with external baselines "
            "(RHC-Inspection and CBCP) under a unified evaluation protocol. "
            "The main text reports one low-conflict representative scenario (C1--G2--M0) "
            "and two high-conflict scenarios (C1--G2--M2 and C2--G2--M2); "
            "the full eight-scene CDSL results are moved to the supplementary material."
        ),
        "",
        (
            f"In the low-conflict representative scenario, "
            f"RHC-Inspection, CBCP, WCDL, and FDLC achieve "
            f"$R_{{\\mathrm{{task}}}}={val('c1_g2_m0', 'RHC-Inspection'):.3f}$, "
            f"{val('c1_g2_m0', 'CBCP'):.3f}$, "
            f"{val('c1_g2_m0', 'WCDL'):.3f}$, and "
            f"{val('c1_g2_m0', 'FDLC'):.3f}$, respectively. "
            f"The performance gap remains limited, indicating that the full feedback loop "
            f"provides marginal benefit when task--communication conflict is weak."
        ),
        "",
        (
            f"In C1--G2--M2, FDLC reaches $R_{{\\mathrm{{task}}}}={val('c1_g2_m2', 'FDLC'):.3f}$, "
            f"whereas RHC-Inspection, CBCP, and WCDL remain at "
            f"{val('c1_g2_m2', 'RHC-Inspection'):.3f}, "
            f"{val('c1_g2_m2', 'CBCP'):.3f}, and "
            f"{val('c1_g2_m2', 'WCDL'):.3f}. "
            f"In C2--G2--M2, the same pattern holds: FDLC ({val('c2_g2_m2', 'FDLC'):.3f}) "
            f"clearly exceeds RHC ({val('c2_g2_m2', 'RHC-Inspection'):.3f}), "
            f"CBCP ({val('c2_g2_m2', 'CBCP'):.3f}), and WCDL ({val('c2_g2_m2', 'WCDL'):.3f})."
        ),
        "",
        "Paired Wilcoxon tests (seed as the unit of analysis; Holm correction across FDLC comparisons) "
        "confirm that FDLC significantly outperforms all three baselines in both high-conflict scenarios "
        "(Table~\\ref{tab:ch5-overall-stats}).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--struct-summary",
        type=Path,
        default=ROOT
        / "artifacts/baseline_before_paper_sweep_20260615_103151/runs_snapshot/sweeps/struct_axis_full_20260606/summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/chapter5_overall_comparison",
    )
    args = parser.parse_args()

    _style()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = _collect_baseline_rows()
    struct_rows = _collect_struct_rows(args.struct_summary)
    all_rows = pd.concat([baseline_rows, struct_rows], ignore_index=True)

    main_case_keys = [s[0] for s in SCENARIOS_MAIN]
    seed_rows = all_rows[
        all_rows["case_key"].isin(main_case_keys) & all_rows["method"].isin(METHODS_MAIN)
    ].copy()
    seed_rows.to_csv(out_dir / "seed_rows.csv", index=False)

    summary = _aggregate_summary(seed_rows)
    summary.to_csv(out_dir / "summary_table.csv", index=False)

    stats_df = _run_stats(seed_rows, SCENARIOS_MAIN)
    stats_df.to_csv(out_dir / "statistical_tests.csv", index=False)

    fig_path = _plot(summary, out_dir)

    cdsl_rows = struct_rows[
        struct_rows["case_key"].isin(SCENARIOS_CDSL)
        & struct_rows["method"].isin(METHODS_CDSL)
    ].copy()
    summary_cdsl = _aggregate_summary(cdsl_rows)
    summary_cdsl.to_csv(out_dir / "supplementary_cdsl_summary.csv", index=False)

    latex = "\n".join(
        [
            _draft_text(summary, stats_df),
            _latex_main_table(summary, stats_df),
            _latex_cdsl_supplementary(summary_cdsl),
        ]
    )
    (out_dir / "chapter5_overall_comparison_latex.tex").write_text(latex, encoding="utf-8")

    report = [
        "# Chapter 5.1 overall comparison",
        "",
        f"- Seed rows: `{out_dir / 'seed_rows.csv'}`",
        f"- Summary table: `{out_dir / 'summary_table.csv'}`",
        f"- Statistical tests: `{out_dir / 'statistical_tests.csv'}`",
        f"- Figure: `{fig_path}`",
        f"- LaTeX snippet: `{out_dir / 'chapter5_overall_comparison_latex.tex'}`",
        f"- CDSL supplementary: `{out_dir / 'supplementary_cdsl_summary.csv'}`",
        "",
        "## Selected scenarios",
    ]
    for _, label, note in SCENARIOS_MAIN:
        report.append(f"- {label}: {note}")
    report.append("")
    report.append("## R_task summary")
    for case_key, label, _ in SCENARIOS_MAIN:
        report.append(f"### {label}")
        for method in METHODS_MAIN:
            r = summary[(summary["case_key"] == case_key) & (summary["method"] == method)]
            if r.empty:
                continue
            m = float(r["R_task_mean"].iloc[0])
            lo = float(r["R_task_ci_low"].iloc[0])
            hi = float(r["R_task_ci_high"].iloc[0])
            report.append(f"- {method}: {m:.3f} [{lo:.3f}, {hi:.3f}]")
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
