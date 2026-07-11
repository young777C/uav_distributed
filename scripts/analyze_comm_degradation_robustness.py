#!/usr/bin/env python3
"""Combine internal and external communication degradation sensitivity scans."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


LEVELS = ["low", "medium", "high"]
LEVEL_LABELS = ["Low", "Medium", "High"]
METHODS = ["RHC-Inspection", "CBCP", "WCDL", "FDLC"]
COLORS = {
    "RHC-Inspection": "#6f6f6f",
    "CBCP": "#4c78a8",
    "WCDL": "#54a24b",
    "FDLC": "#f58518",
}
MARKERS = {
    "RHC-Inspection": "s",
    "CBCP": "D",
    "WCDL": "^",
    "FDLC": "o",
}


def _style() -> None:
    tinos_dir = Path("/usr/share/fonts/truetype/croscore")
    for font_path in sorted(tinos_dir.glob("Tinos-*.ttf")):
        font_manager.fontManager.addfont(font_path)
    mpl.rcParams.update(
        {
            "font.family": "Tinos",
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


def _parse_exp_name(name: str) -> tuple[str, str, str]:
    case_name, system_name = name.split("__", 1)
    parts = case_name.split("_")
    comm_case = parts[0].upper()
    intensity = parts[-1]
    return comm_case, intensity, system_name


def _method_name(system_name: str) -> str | None:
    mapping = {
        "baseline_rhc_inspection": "RHC-Inspection",
        "baseline_cbcp": "CBCP",
        "struct_decoupled_dual_loop": "WCDL",
        "struct_full_dual_loop_distributed": "FDLC",
    }
    return mapping.get(system_name)


def _load_seed_summary(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    parsed = df["exp_name"].map(_parse_exp_name)
    df["comm_case"] = parsed.map(lambda x: x[0])
    df["intensity"] = parsed.map(lambda x: x[1])
    df["system"] = parsed.map(lambda x: x[2])
    df["method"] = df["system"].map(_method_name)
    df = df[df["method"].isin(METHODS)].copy()
    df["source"] = source
    df["P_return_given_cov_mean"] = 1.0 - pd.to_numeric(
        df["R_fail_given_cov_mean"], errors="coerce"
    )
    for col in [
        "R_task_mean",
        "R_cov_mean",
        "T_ret_s_mean",
        "returned_home_rate",
        "oob_rate",
        "T_nf_s_mean",
        "P_return_given_cov_mean",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _aggregate(seed_rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "R_task_mean",
        "R_cov_mean",
        "P_return_given_cov_mean",
        "T_ret_s_mean",
        "returned_home_rate",
        "oob_rate",
        "T_nf_s_mean",
    ]
    rows: list[dict[str, object]] = []
    for (comm_case, intensity, method), g in seed_rows.groupby(
        ["comm_case", "intensity", "method"], sort=False
    ):
        row: dict[str, object] = {
            "comm_case": comm_case,
            "intensity": intensity,
            "method": method,
            "seed_count": int(g["seed"].nunique()),
            "episodes_total": int(pd.to_numeric(g["episodes"], errors="coerce").sum()),
        }
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            row[metric.replace("_mean", "") + "_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[metric.replace("_mean", "") + "_seed_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["level_order"] = out["intensity"].map({x: i for i, x in enumerate(LEVELS)})
    out["method_order"] = out["method"].map({x: i for i, x in enumerate(METHODS)})
    return out.sort_values(["comm_case", "level_order", "method_order"]).drop(
        columns=["level_order", "method_order"]
    )


def _line_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(LEVELS))
    for method in METHODS:
        subset = data[data["method"] == method].set_index("intensity").reindex(LEVELS)
        ax.errorbar(
            x,
            subset[f"{metric}_mean"],
            yerr=subset[f"{metric}_seed_std"],
            marker=MARKERS[method],
            markersize=4.5,
            linewidth=1.5,
            capsize=3,
            color=COLORS[method],
            label=method,
        )
    ax.set_xticks(x, LEVEL_LABELS)
    ax.set_xlabel("Communication degradation intensity")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=360)
    plt.close(fig)


def _plot(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, comm_case in zip(axes, ["C1", "C2"]):
        _line_panel(
            ax,
            summary[summary["comm_case"] == comm_case],
            "R_task",
            "Effective task completion rate",
            (0.0, 0.8),
        )
        ax.set_title(comm_case)
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "fig_main_robustness_r_task")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, comm_case in zip(axes, ["C1", "C2"]):
        _line_panel(
            ax,
            summary[summary["comm_case"] == comm_case],
            "T_ret_s",
            "Mean return latency (s)",
            None,
        )
        ax.set_title(comm_case)
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "fig_main_robustness_return_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, comm_case in zip(axes, ["C1", "C2"]):
        _line_panel(
            ax,
            summary[summary["comm_case"] == comm_case],
            "oob_rate",
            "Out-of-bound termination rate",
            (0.0, 1.05),
        )
        ax.set_title(comm_case)
    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "fig_main_robustness_oob_rate")


def _fmt_series(summary: pd.DataFrame, comm_case: str, method: str, metric: str) -> str:
    subset = (
        summary[(summary["comm_case"] == comm_case) & (summary["method"] == method)]
        .set_index("intensity")
        .reindex(LEVELS)
    )
    return "/".join(f"{float(v):.3f}" for v in subset[f"{metric}_mean"])


def _write_report(summary: pd.DataFrame, out_dir: Path, internal_path: Path, external_path: Path) -> None:
    lines = [
        "# Communication Degradation Robustness Analysis",
        "",
        "## Data Sources",
        "",
        f"- Internal architecture scan: `{internal_path}`.",
        f"- External baseline scan: `{external_path}`.",
        "- Methods compared in the main robustness scan: RHC-Inspection, CBCP, WCDL, FDLC.",
        "- Scenarios: C1--G2--M2 and C2--G2--M2.",
        "- Communication intensity levels: low, medium, high.",
        "- Each method/intensity/scenario entry aggregates 3 seeds x 5 episodes.",
        "",
        "## Main Findings",
        "",
    ]
    for comm_case, mechanism in [("C1", "distance-decay"), ("C2", "local-shadowing")]:
        lines.append(f"### {comm_case} ({mechanism})")
        for method in METHODS:
            lines.append(
                f"- {method}: R_task low/medium/high = "
                f"{_fmt_series(summary, comm_case, method, 'R_task')}; "
                f"P_return|cov = {_fmt_series(summary, comm_case, method, 'P_return_given_cov')}; "
                f"OOB rate = {_fmt_series(summary, comm_case, method, 'oob_rate')}."
            )
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "- FDLC remains the strongest method among the four compared methods in both C1 and C2 high-conflict scans.",
        "- The results do not support a blanket monotonic-decrease claim for every method. Several external baselines are dominated by early out-of-bound termination or conservative task selection, so changing communication intensity alone does not always produce a smooth degradation curve.",
        "- In C1, CBCP shows a non-monotonic result: its medium-intensity task rate is higher than both low and high. This should be reported as an observed boundary phenomenon, not smoothed away.",
        "- In C2, RHC-Inspection remains around 0.270 and CBCP around 0.170 across all three intensities, while FDLC stays higher but also exhibits non-monotonic sensitivity. The stronger conclusion is therefore relative robustness against representative baselines under high-conflict degradation, rather than strictly smaller decline from low to high.",
        "- RHC-Inspection and CBCP mostly end with out-of-bound termination in these high-conflict scans. This indicates that their open-loop or constraint-driven planning does not close the loop between local communication risk, coverage progress, and return-home termination as effectively as FDLC.",
        "",
        "## Generated Files",
        "",
        "- `main_robustness_scan_seed_rows.csv`",
        "- `main_robustness_scan_summary.csv`",
        "- `fig_main_robustness_r_task.pdf/.png`",
        "- `fig_main_robustness_return_metrics.pdf/.png`",
        "- `fig_main_robustness_oob_rate.pdf/.png`",
        "- `chapter5_robustness_latex_draft.tex`",
    ]
    (out_dir / "comm_degradation_robustness_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_latex(summary: pd.DataFrame, out_dir: Path) -> None:
    text = r"""\subsection{通信退化强度敏感性分析}
为进一步评估方法在通信退化增强条件下的鲁棒性，本文在距离衰减主导场景和局部遮挡主导场景中设置 low、medium 和 high 三档通信退化强度，并比较 RHC-Inspection、CBCP、WCDL 与 FDLC。图~\ref{fig:main-robustness-rtask} 给出了有效任务完成率随退化强度变化的结果，图~\ref{fig:main-robustness-return} 进一步给出了平均回传时延。

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.86\textwidth]{figures/chapter5/fig_main_robustness_r_task.pdf}
  \caption{高冲突场景下不同方法的通信退化强度敏感性}
  \label{fig:main-robustness-rtask}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.86\textwidth]{figures/chapter5/fig_main_robustness_return_metrics.pdf}
  \caption{高冲突场景下平均回传时延}
  \label{fig:main-robustness-return}
\end{figure}

在距离衰减主导场景中，FDLC 在 low、medium 和 high 三档下的 $R_{\mathrm{task}}$ 分别为 __C1_FDLC__，高于 RHC-Inspection、CBCP 和 WCDL。需要注意的是，外部基线并未呈现完全单调的退化趋势；例如 CBCP 在 medium 档达到 __C1_CBCP_MED__，而 low 和 high 档分别为 __C1_CBCP_LOW__ 和 __C1_CBCP_HIGH__。该现象表明，本组实验不能被简化为所有方法随通信退化增强而单调下降，而应解释为高冲突任务布局下不同规划机制的失败模式差异。

在局部遮挡主导场景中，RHC-Inspection 在 low、medium 和 high 三档下的 $R_{\mathrm{task}}$ 分别为 __C2_RHC__，CBCP 分别为 __C2_CBCP__，二者主要受越界终止和保守任务推进限制。FDLC 在三档强度下保持较高有效完成率，说明事件反馈、快环模式切换和慢环任务修正能够在局部遮挡条件下维持更强的覆盖--回传闭合能力。与此同时，FDLC 自身也表现出一定非单调敏感性，因此结论应表述为其在代表性高冲突退化条件下相对外部基线更稳健，而不是严格声称其性能随退化强度线性下降或下降斜率始终最小。
"""
    repl = {
        "__C1_FDLC__": _fmt_series(summary, "C1", "FDLC", "R_task").replace("/", "、"),
        "__C1_CBCP_LOW__": f"{float(summary[(summary.comm_case == 'C1') & (summary.method == 'CBCP') & (summary.intensity == 'low')]['R_task_mean'].iloc[0]):.3f}",
        "__C1_CBCP_MED__": f"{float(summary[(summary.comm_case == 'C1') & (summary.method == 'CBCP') & (summary.intensity == 'medium')]['R_task_mean'].iloc[0]):.3f}",
        "__C1_CBCP_HIGH__": f"{float(summary[(summary.comm_case == 'C1') & (summary.method == 'CBCP') & (summary.intensity == 'high')]['R_task_mean'].iloc[0]):.3f}",
        "__C2_RHC__": _fmt_series(summary, "C2", "RHC-Inspection", "R_task").replace("/", "、"),
        "__C2_CBCP__": _fmt_series(summary, "C2", "CBCP", "R_task").replace("/", "、"),
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    (out_dir / "chapter5_robustness_latex_draft.tex").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-summary", type=Path, required=True)
    parser.add_argument("--external-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _style()

    internal = _load_seed_summary(args.internal_summary, "internal")
    external = _load_seed_summary(args.external_summary, "external")
    seed_rows = pd.concat([external, internal], ignore_index=True)
    summary = _aggregate(seed_rows)

    seed_rows.to_csv(out_dir / "main_robustness_scan_seed_rows.csv", index=False)
    summary.to_csv(out_dir / "main_robustness_scan_summary.csv", index=False)
    _plot(summary, out_dir)
    _write_report(summary, out_dir, args.internal_summary, args.external_summary)
    _write_latex(summary, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
