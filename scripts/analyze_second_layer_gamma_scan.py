#!/usr/bin/env python3
"""Analyze second-layer robustness scans over four communication intensities."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


LEVELS = ["low", "mid", "high", "severe"]
LEVEL_LABELS = [
    r"$\Gamma_{\mathrm{low}}$",
    r"$\Gamma_{\mathrm{mid}}$",
    r"$\Gamma_{\mathrm{high}}$",
    r"$\Gamma_{\mathrm{severe}}$",
]

METHOD_LABELS = {
    "struct_full_dual_loop_distributed": "Full Coupling",
    "coupling_periodic_goal": "Periodic Goal",
    "coupling_event_driven_goal": "Event-driven Goal",
    "coupling_full_coupling": "Full Coupling",
    "modelling_comm_aware_decision": "Comm-only",
    "modelling_energy_aware_decision": "Energy-only",
    "modelling_comm_energy_aware_decision": "Comm+Energy",
}

COLORS = {
    "Periodic Goal": "#777777",
    "Event-driven Goal": "#4c78a8",
    "Full Coupling": "#f58518",
    "Comm-only": "#4c78a8",
    "Energy-only": "#777777",
    "Comm+Energy": "#54a24b",
}

METRICS = [
    ("R_task_mean", "Effective task completion rate"),
    ("P_return_given_cov_mean", "Post-coverage return success rate"),
    ("T_ret_s_mean", "Mean return latency (s)"),
]


def style() -> None:
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


def parse_exp_name(exp_name: str) -> tuple[str, str, str]:
    case, system = exp_name.split("__", 1)
    parts = case.split("_")
    comm_case = parts[0].upper()
    intensity = parts[-1]
    if intensity == "medium":
        intensity = "mid"
    return comm_case, intensity, system


def summarize(summary_path: Path, axis: str, *, allowed_methods: set[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    parsed = df["exp_name"].map(parse_exp_name)
    df["comm_case"] = parsed.map(lambda x: x[0])
    df["intensity"] = parsed.map(lambda x: x[1])
    df["system"] = parsed.map(lambda x: x[2])
    df["method"] = df["system"].map(METHOD_LABELS)
    if allowed_methods is not None:
        df = df[df["method"].isin(allowed_methods)].copy()
    df["P_return_given_cov_mean"] = 1.0 - pd.to_numeric(
        df["R_fail_given_cov_mean"], errors="coerce"
    )
    df["P_return_given_cov_std"] = pd.to_numeric(
        df["R_fail_given_cov_std"], errors="coerce"
    )

    rows: list[dict[str, object]] = []
    for group_key, group in df.groupby(["comm_case", "intensity", "method"], sort=False):
        comm_case, intensity, method = group_key
        row: dict[str, object] = {
            "axis": axis,
            "comm_case": comm_case,
            "intensity": intensity,
            "method": method,
            "seed_count": int(group["seed"].nunique()),
            "episodes_total": int(group["episodes"].sum()),
        }
        for metric, _ in METRICS:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_seed_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)

    out = pd.DataFrame(rows)
    out["level_order"] = out["intensity"].map({k: i for i, k in enumerate(LEVELS)})
    return out.sort_values(["level_order", "method"]).drop(columns=["level_order"])


def assemble_from_sources(legacy_root: Path, severe_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coupling_methods = {"Periodic Goal", "Event-driven Goal", "Full Coupling"}
    information_methods = {"Comm-only", "Energy-only", "Comm+Energy"}

    legacy_coupling = summarize(
        legacy_root / "coupling_scan" / "summary.csv",
        "coupling",
        allowed_methods={"Periodic Goal", "Event-driven Goal"},
    )
    legacy_full = summarize(
        legacy_root / "architecture_scan" / "summary.csv",
        "coupling",
        allowed_methods={"Full Coupling"},
    )
    legacy_full = legacy_full[legacy_full["comm_case"] == "C2"].copy()
    legacy_information = summarize(
        legacy_root / "information_scan" / "summary.csv",
        "information",
        allowed_methods=information_methods,
    )
    severe_coupling = summarize(
        severe_root / "coupling_scan" / "summary.csv",
        "coupling",
        allowed_methods=coupling_methods,
    )
    severe_information = summarize(
        severe_root / "information_scan" / "summary.csv",
        "information",
        allowed_methods=information_methods,
    )

    coupling = pd.concat(
        [
            legacy_coupling[legacy_coupling["intensity"].isin(["low", "mid", "high"])],
            legacy_full[legacy_full["intensity"].isin(["low", "mid", "high"])],
            severe_coupling[severe_coupling["intensity"] == "severe"],
        ],
        ignore_index=True,
    )
    information = pd.concat(
        [
            legacy_information[legacy_information["intensity"].isin(["low", "mid", "high"])],
            severe_information[severe_information["intensity"] == "severe"],
        ],
        ignore_index=True,
    )
    return coupling, information


def line_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    methods: list[str],
    metric: str,
    ylabel: str,
) -> None:
    x = np.arange(len(LEVELS))
    for method in methods:
        subset = data[data["method"] == method].set_index("intensity").reindex(LEVELS)
        ax.errorbar(
            x,
            subset[metric],
            yerr=subset[f"{metric}_seed_std"],
            marker="o",
            capsize=3,
            linewidth=1.5,
            color=COLORS[method],
            label=method,
        )
    ax.set_xticks(x, LEVEL_LABELS)
    ax.set_xlabel("Communication degradation intensity")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig: plt.Figure, root: Path, stem: str) -> None:
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(root / f"{stem}.png", bbox_inches="tight", dpi=360)
    plt.close(fig)


def plot_axis(root: Path, data: pd.DataFrame, methods: list[str], stem: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for ax, (metric, ylabel) in zip(axes, METRICS):
        line_panel(ax, data, methods, metric, ylabel)
    axes[0].legend(loc="best")
    fig.tight_layout()
    save(fig, root, stem)


def fmt_series(data: pd.DataFrame, method: str, metric: str) -> str:
    subset = data[data["method"] == method].set_index("intensity").reindex(LEVELS)
    return "/".join(f"{v:.3f}" for v in subset[metric].astype(float).to_numpy())


def markdown_table(data: pd.DataFrame, methods: list[str]) -> str:
    rows = [
        "| Method | Metric | $\\Gamma_{low}$ | $\\Gamma_{mid}$ | $\\Gamma_{high}$ | $\\Gamma_{severe}$ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    metric_names = {
        "R_task_mean": "$R_{task}$",
        "P_return_given_cov_mean": "$P_{return|cov}$",
        "T_ret_s_mean": "$\\bar{T}_{ret}$ / s",
    }
    for method in methods:
        subset = data[data["method"] == method].set_index("intensity").reindex(LEVELS)
        for metric, _ in METRICS:
            vals = [f"{float(v):.3f}" for v in subset[metric].to_numpy()]
            rows.append(f"| {method} | {metric_names[metric]} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_report(root: Path, coupling: pd.DataFrame, information: pd.DataFrame) -> None:
    coupling_methods = ["Periodic Goal", "Event-driven Goal", "Full Coupling"]
    info_methods = ["Comm-only", "Energy-only", "Comm+Energy"]

    lines = [
        "# Second-Layer Gamma Robustness Scan Report",
        "",
        "## Scope",
        "",
        "- Scenario: C2--G2--M2.",
        "- Communication intensity: $\\Gamma_{low}$, $\\Gamma_{mid}$, $\\Gamma_{high}$, $\\Gamma_{severe}$.",
        "- Multipliers: 0.75, 1.00, 1.25, 1.50 relative to the baseline mid profile.",
        "- Seeds: 0, 1, 2; episodes per seed: 5.",
        "- Reported metrics: $R_{task}$, $P_{return|cov}$, and $\\bar{T}_{ret}$.",
        "- No core algorithm code or existing result directory was overwritten.",
        "",
        "## Coupling Mechanism Robustness",
        "",
        markdown_table(coupling, coupling_methods),
        "",
        "## Information-Use Robustness",
        "",
        markdown_table(information, info_methods),
        "",
        "## Key Observations",
        "",
    ]
    for method in coupling_methods:
        lines.append(
            f"- {method}: R_task = {fmt_series(coupling, method, 'R_task_mean')}; "
            f"P_return|cov = {fmt_series(coupling, method, 'P_return_given_cov_mean')}; "
            f"T_ret = {fmt_series(coupling, method, 'T_ret_s_mean')} s."
        )
    for method in info_methods:
        lines.append(
            f"- {method}: R_task = {fmt_series(information, method, 'R_task_mean')}; "
            f"P_return|cov = {fmt_series(information, method, 'P_return_given_cov_mean')}; "
            f"T_ret = {fmt_series(information, method, 'T_ret_s_mean')} s."
        )
    lines += [
        "",
        "## Generated Files",
        "",
        "- `second_layer_coupling_gamma_summary.csv`",
        "- `second_layer_information_gamma_summary.csv`",
        "- `fig_second_layer_coupling_gamma.pdf/.png`",
        "- `fig_second_layer_information_gamma.pdf/.png`",
        "- `second_layer_gamma_latex_draft.tex`",
    ]
    (root / "second_layer_gamma_scan_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(root: Path, coupling: pd.DataFrame, information: pd.DataFrame) -> None:
    text = rf"""% ===== 第二层通信退化机制鲁棒性验证草稿 =====
\subsection{{通信退化强度下的机制鲁棒性验证}}
为检验消融结论是否仅依赖于单一通信退化强度，本文在局部遮挡主导的高冲突场景 C2--G2--M2 中进一步设置
$\Gamma\in\{{\Gamma_{{\mathrm{{low}}}},\Gamma_{{\mathrm{{mid}}}},\Gamma_{{\mathrm{{high}}}},\Gamma_{{\mathrm{{severe}}}}\}}$ 四档通信退化强度，并分别比较双环耦合机制与信息利用方式。评价指标包括任务有效完成率 $R_{{\mathrm{{task}}}}$、覆盖后有效回传率 $P_{{\mathrm{{return}}\mid\mathrm{{cov}}}}$ 和平均回传时延 $\bar{{T}}_{{\mathrm{{ret}}}}$。

图~\ref{{fig:second-layer-coupling-gamma}} 给出了耦合机制扫描结果。Periodic Goal、Event-driven Goal 和 Full Coupling 的 $R_{{\mathrm{{task}}}}$ 分别为 {fmt_series(coupling, "Periodic Goal", "R_task_mean")}、{fmt_series(coupling, "Event-driven Goal", "R_task_mean")} 和 {fmt_series(coupling, "Full Coupling", "R_task_mean")}。该结果应与覆盖后有效回传率和平均回传时延联合解释，以区分任务推进能力、回传闭合能力和覆盖后等待时间之间的差异。

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{{../results/comm_degradation_gamma_second_layer/{root.name}/fig_second_layer_coupling_gamma.pdf}}
  \caption{{四档通信退化强度下的耦合机制鲁棒性}}
  \label{{fig:second-layer-coupling-gamma}}
\end{{figure}}

图~\ref{{fig:second-layer-information-gamma}} 给出了信息利用扫描结果。Comm-only、Energy-only 和 Comm+Energy 的 $R_{{\mathrm{{task}}}}$ 分别为 {fmt_series(information, "Comm-only", "R_task_mean")}、{fmt_series(information, "Energy-only", "R_task_mean")} 和 {fmt_series(information, "Comm+Energy", "R_task_mean")}。其中，通信信息主要影响覆盖结果能否转化为有效回传，能量信息主要约束任务闭合和返航可达性，联合建模结果需要从有效完成率和闭合状态之间的折中进行分析。

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{{../results/comm_degradation_gamma_second_layer/{root.name}/fig_second_layer_information_gamma.pdf}}
  \caption{{四档通信退化强度下的信息利用鲁棒性}}
  \label{{fig:second-layer-information-gamma}}
\end{{figure}}
"""
    (root / "second_layer_gamma_latex_draft.tex").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, default=None)
    parser.add_argument("--severe-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    style()

    if args.legacy_root is not None and args.severe_root is not None:
        coupling, information = assemble_from_sources(
            args.legacy_root.resolve(),
            args.severe_root.resolve(),
        )
    else:
        coupling = summarize(root / "coupling_scan" / "summary.csv", "coupling")
        information = summarize(root / "information_scan" / "summary.csv", "information")

    coupling.to_csv(root / "second_layer_coupling_gamma_summary.csv", index=False)
    information.to_csv(root / "second_layer_information_gamma_summary.csv", index=False)
    plot_axis(
        root,
        coupling,
        ["Periodic Goal", "Event-driven Goal", "Full Coupling"],
        "fig_second_layer_coupling_gamma",
    )
    plot_axis(
        root,
        information,
        ["Comm-only", "Energy-only", "Comm+Energy"],
        "fig_second_layer_information_gamma",
    )
    write_report(root, coupling, information)
    write_latex(root, coupling, information)
    print(root)


if __name__ == "__main__":
    main()
