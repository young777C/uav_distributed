#!/usr/bin/env python3
"""Aggregate communication degradation scans, generate figures, report, and LaTeX draft."""
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
COLORS = {
    "CDSL": "#777777",
    "WCDL": "#4c78a8",
    "FDLC": "#f58518",
    "Periodic Goal": "#777777",
    "Event-driven Goal": "#4c78a8",
    "Full Coupling": "#f58518",
    "Comm-only": "#4c78a8",
    "Energy-only": "#777777",
    "Comm+Energy": "#54a24b",
}


def style() -> None:
    tinos_dir = Path("/usr/share/fonts/truetype/croscore")
    for font_path in sorted(tinos_dir.glob("Tinos-*.ttf")):
        font_manager.fontManager.addfont(font_path)
    mpl.rcParams.update(
        {
            "font.family": "Tinos",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def parse_name(name: str) -> tuple[str, str, str]:
    case, system = name.split("__", 1)
    parts = case.split("_")
    comm_case = parts[0].upper()
    if parts[-1] == "E210":
        intensity = parts[-2]
    else:
        intensity = parts[-1]
    return comm_case, intensity, system


def method_name(system: str) -> str:
    mapping = {
        "struct_centralized_single_loop": "CDSL",
        "struct_decoupled_dual_loop": "WCDL",
        "struct_full_dual_loop_distributed": "FDLC",
        "coupling_periodic_goal": "Periodic Goal",
        "coupling_event_driven_goal": "Event-driven Goal",
        "coupling_full_coupling": "Full Coupling",
        "modelling_comm_aware_decision": "Comm-only",
        "modelling_energy_aware_decision": "Energy-only",
        "modelling_comm_energy_aware_decision": "Comm+Energy",
    }
    return mapping[system]


def summarize(path: Path, axis: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    parsed = df["exp_name"].map(parse_name)
    df["comm_case"] = parsed.map(lambda x: x[0])
    df["intensity"] = parsed.map(lambda x: x[1])
    df["system"] = parsed.map(lambda x: x[2])
    df["method"] = df["system"].map(method_name)
    df["P_return_given_cov_mean"] = 1.0 - df["R_fail_given_cov_mean"]
    metrics = [
        "R_task_mean",
        "R_cov_mean",
        "P_return_given_cov_mean",
        "T_ret_s_mean",
        "returned_home_rate",
        "oob_rate",
        "energy_rate",
        "remaining_energy_mean",
        "T_nf_s_mean",
    ]
    rows: list[dict[str, object]] = []
    keys = ["comm_case", "intensity", "method"]
    for group_key, group in df.groupby(keys, sort=False):
        row = dict(zip(keys, group_key))
        row["axis"] = axis
        row["seed_count"] = int(group["seed"].nunique())
        row["episodes_total"] = int(group["episodes"].sum())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def line_panel(ax: plt.Axes, data: pd.DataFrame, methods: list[str], metric: str, ylabel: str) -> None:
    x = np.arange(3)
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


def trend(values: pd.Series) -> str:
    vals = values.to_numpy(dtype=float)
    if np.all(np.diff(vals) <= 1e-9):
        return "monotonic non-increasing"
    if vals[-1] < vals[0]:
        return "lower at high than low, but non-monotonic"
    return "not lower at high than low"


def level_values(data: pd.DataFrame, method: str, metric: str) -> list[float]:
    subset = data[data["method"] == method].set_index("intensity").reindex(LEVELS)
    return subset[metric].astype(float).tolist()


def architecture_latex_text(arch: pd.DataFrame) -> str:
    paragraphs = []
    for comm_case, mechanism in [("C1", "距离衰减"), ("C2", "局部遮挡")]:
        data = arch[arch["comm_case"] == comm_case]
        entries = []
        changes = {}
        for method in ["CDSL", "WCDL", "FDLC"]:
            values = level_values(data, method, "R_task_mean")
            changes[method] = values[2] - values[0]
            entries.append(
                f"{method} 的 $R_{{\\mathrm{{task}}}}$ 由 {values[0]:.3f} 变化至 {values[2]:.3f}"
            )
        fd_lower_drop = changes["FDLC"] >= changes["CDSL"] and changes["FDLC"] >= changes["WCDL"]
        assessment = (
            "FDLC 从低档到高档的下降幅度不大于两种基线，表明其在该退化机制下具有更稳定的任务收益。"
            if fd_lower_drop
            else "FDLC 从低档到高档的变化幅度并未同时小于两种基线，因此该组结果不支持将其概括为对所有退化变化均更稳定。"
        )
        paragraphs.append(
            f"在{mechanism}主导条件下，" + "，".join(entries) + f"。{assessment}"
        )
    return "\n\n".join(paragraphs)


def coupling_latex_text(coupling: pd.DataFrame) -> str:
    fragments = []
    for method in ["Periodic Goal", "Event-driven Goal", "Full Coupling"]:
        task = level_values(coupling, method, "R_task_mean")
        latency = level_values(coupling, method, "T_ret_s_mean")
        fragments.append(
            f"{method} 的 $R_{{\\mathrm{{task}}}}$ 为 "
            f"{task[0]:.3f}/{task[1]:.3f}/{task[2]:.3f}，"
            f"$\\bar{{T}}_{{\\mathrm{{ret}}}}="
            f"{latency[0]:.2f}/{latency[1]:.2f}/{latency[2]:.2f}\\,\\mathrm{{s}}$"
        )
    full = coupling[coupling["method"] == "Full Coupling"].set_index("intensity")
    best_levels = []
    for level in ["medium", "high"]:
        level_data = coupling[coupling["intensity"] == level]
        if not level_data.empty and full.loc[level, "R_task_mean"] >= level_data["R_task_mean"].max() - 1e-9:
            best_levels.append("中档" if level == "medium" else "高档")
    if len(best_levels) == 2:
        assessment = "Full Coupling 在中、高退化条件下均取得最高完成率。"
    elif best_levels:
        assessment = f"Full Coupling 仅在{'和'.join(best_levels)}退化条件下取得最高完成率。"
    else:
        assessment = "Full Coupling 在中、高退化条件下均未取得最高完成率。"
    return "；".join(fragments) + f"。{assessment}时延结果需与完成率联合解释，以避免将更早终止误判为更快回传。"


def information_latex_text(info: pd.DataFrame) -> str:
    fragments = []
    for method in ["Comm-only", "Energy-only", "Comm+Energy"]:
        task = level_values(info, method, "R_task_mean")
        returned = level_values(info, method, "returned_home_rate")
        oob = level_values(info, method, "oob_rate")
        fragments.append(
            f"{method} 在低、中、高退化下的 $R_{{\\mathrm{{task}}}}$ 分别为 "
            f"{task[0]:.3f}、{task[1]:.3f} 和 {task[2]:.3f}，"
            f"返航率分别为 {returned[0]:.3f}、{returned[1]:.3f} 和 {returned[2]:.3f}，"
            f"越界率分别为 {oob[0]:.3f}、{oob[1]:.3f} 和 {oob[2]:.3f}。"
        )
    return (
        "\n\n".join(fragments)
        + "\n\n这些指标共同刻画通信收益与任务闭合状态；若完成率提高同时伴随越界率上升，"
        "则不能仅依据原始任务收益判断信息利用方式的综合效果。"
    )


def e210_latex_text(info: pd.DataFrame) -> str:
    if info.empty:
        return ""
    fragments = []
    for method in ["Comm-only", "Energy-only", "Comm+Energy"]:
        subset = info[info["method"] == method].set_index("intensity")
        fragments.append(
            f"{method} 在中、高退化下的 $R_{{\\mathrm{{task}}}}$ 分别为 "
            f"{subset.loc['medium', 'R_task_mean']:.3f} 和 "
            f"{subset.loc['high', 'R_task_mean']:.3f}，"
            f"能量终止率分别为 {subset.loc['medium', 'energy_rate']:.3f} 和 "
            f"{subset.loc['high', 'energy_rate']:.3f}。"
        )
    return (
        "在 E210 能量受限条件下，各信息利用方式的结果如下。\n\n"
        + "\n\n".join(fragments)
        + "\n\n现有输出未记录独立的能量约束触发次数，因此此处以能量终止率作为可观测代理。"
    )


def link_markdown_table(link_stats: pd.DataFrame) -> str:
    header = (
        "| Mechanism | Intensity | Mean loss | Mean delay (s) | "
        "Mean bandwidth (Mbps) | Weak-link ratio |"
    )
    separator = "|---|---|---:|---:|---:|---:|"
    rows = [header, separator]
    for row in link_stats.itertuples(index=False):
        rows.append(
            f"| {row.comm_case} | {row.intensity} | {row.mean_loss_p:.4f} | "
            f"{row.mean_delay_s:.4f} | {row.mean_bandwidth_bps / 1e6:.4f} | "
            f"{row.weak_link_ratio_loss_ge_0_5:.4f} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    style()

    arch = summarize(root / "architecture_scan/summary.csv", "architecture")
    coupling = summarize(root / "coupling_scan/summary.csv", "coupling")
    if "Full Coupling" not in set(coupling["method"]):
        reused_full = arch[
            (arch["comm_case"] == "C2") & (arch["method"] == "FDLC")
        ].copy()
        reused_full["axis"] = "coupling"
        reused_full["method"] = "Full Coupling"
        reused_full["system"] = "coupling_full_coupling"
        coupling = pd.concat([coupling, reused_full], ignore_index=True)
    info_parts = [summarize(root / "information_scan/summary.csv", "information")]
    e210_path = root / "information_scan_E210/summary.csv"
    if e210_path.exists():
        e210 = summarize(e210_path, "information_E210")
        e210["energy_tier"] = "E210"
        info_parts.append(e210)
    info = pd.concat(info_parts, ignore_index=True)

    arch.to_csv(root / "architecture_scan_summary.csv", index=False)
    coupling.to_csv(root / "coupling_scan_summary.csv", index=False)
    info.to_csv(root / "information_scan_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, comm_case in zip(axes, ["C1", "C2"]):
        line_panel(
            ax,
            arch[arch["comm_case"] == comm_case],
            ["CDSL", "WCDL", "FDLC"],
            "R_task_mean",
            "Effective task completion rate",
        )
        ax.set_title(comm_case)
    axes[0].legend()
    fig.tight_layout()
    save(fig, root, "fig_architecture_degradation_scan")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    methods = ["Periodic Goal", "Event-driven Goal", "Full Coupling"]
    line_panel(axes[0], coupling, methods, "R_task_mean", "Effective task completion rate")
    line_panel(axes[1], coupling, methods, "T_ret_s_mean", "Mean return latency (s)")
    axes[0].legend()
    fig.tight_layout()
    save(fig, root, "fig_coupling_degradation_scan")

    base_info = info[info["axis"] == "information"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    methods = ["Comm-only", "Energy-only", "Comm+Energy"]
    line_panel(axes[0], base_info, methods, "R_task_mean", "Effective task completion rate")
    line_panel(axes[1], base_info, methods, "returned_home_rate", "Return-home termination rate")
    line_panel(axes[2], base_info, methods, "oob_rate", "Out-of-bound termination rate")
    axes[0].legend()
    fig.tight_layout()
    save(fig, root, "fig_information_degradation_scan")

    notes = []
    for comm_case in ["C1", "C2"]:
        for method in ["CDSL", "WCDL", "FDLC"]:
            s = arch[(arch.comm_case == comm_case) & (arch.method == method)].set_index("intensity").reindex(LEVELS)
            notes.append(
                f"- {comm_case} {method}: R_task is {trend(s['R_task_mean'])}; "
                f"low-to-high change = {s['R_task_mean'].iloc[-1] - s['R_task_mean'].iloc[0]:+.3f}."
            )
    coupling_notes = []
    for method in ["Periodic Goal", "Event-driven Goal", "Full Coupling"]:
        s = coupling[coupling.method == method].set_index("intensity").reindex(LEVELS)
        coupling_notes.append(
            f"- {method}: R_task low/medium/high = "
            f"{s['R_task_mean'].iloc[0]:.3f}/{s['R_task_mean'].iloc[1]:.3f}/{s['R_task_mean'].iloc[2]:.3f}; "
            f"T_ret = {s['T_ret_s_mean'].iloc[0]:.2f}/{s['T_ret_s_mean'].iloc[1]:.2f}/{s['T_ret_s_mean'].iloc[2]:.2f} s."
        )
    info_notes = []
    for method in ["Comm-only", "Energy-only", "Comm+Energy"]:
        s = base_info[base_info.method == method].set_index("intensity").reindex(LEVELS)
        info_notes.append(
            f"- {method}: R_task = {s['R_task_mean'].iloc[0]:.3f}/{s['R_task_mean'].iloc[1]:.3f}/{s['R_task_mean'].iloc[2]:.3f}; "
            f"return-home = {s['returned_home_rate'].iloc[0]:.3f}/{s['returned_home_rate'].iloc[1]:.3f}/{s['returned_home_rate'].iloc[2]:.3f}; "
            f"out-of-bound = {s['oob_rate'].iloc[0]:.3f}/{s['oob_rate'].iloc[1]:.3f}/{s['oob_rate'].iloc[2]:.3f}."
        )

    link_stats = pd.read_csv(root / "link_statistics.csv")
    backup_dir = (
        root.parent.parent
        / "artifacts"
        / root.name.replace("comm_degradation_scan_", "comm_degradation_scan_backup_", 1)
    )
    checks = []
    for comm_case in ["C1", "C2"]:
        data = arch[arch["comm_case"] == comm_case]
        monotonic = all(
            np.all(np.diff(level_values(data, method, "R_task_mean")) <= 1e-9)
            for method in ["CDSL", "WCDL", "FDLC"]
        )
        declines = {
            method: level_values(data, method, "R_task_mean")[0]
            - level_values(data, method, "R_task_mean")[2]
            for method in ["CDSL", "WCDL", "FDLC"]
        }
        robust = declines["FDLC"] <= min(declines["CDSL"], declines["WCDL"]) + 1e-9
        checks.append(
            f"- {comm_case}: all three methods are"
            f"{'' if monotonic else ' not'} monotonic non-increasing; "
            f"FDLC low-to-high decline ({declines['FDLC']:.3f}) is"
            f"{'' if robust else ' not'} smaller than or equal to both baselines."
        )

    full_advantage = []
    for level in ["medium", "high"]:
        data = coupling[coupling["intensity"] == level]
        full_value = float(data.loc[data["method"] == "Full Coupling", "R_task_mean"].iloc[0])
        full_advantage.append(full_value >= float(data["R_task_mean"].max()) - 1e-9)
    checks.append(
        "- Full Coupling "
        + ("has" if all(full_advantage) else "does not have")
        + " the highest R_task at both medium and high degradation."
    )

    info_indexed = base_info.set_index(["method", "intensity"])
    comm_closure_worse = all(
        (
            info_indexed.loc[("Comm-only", level), "returned_home_rate"]
            <= info_indexed.loc[("Comm+Energy", level), "returned_home_rate"] + 1e-9
        )
        and (
            info_indexed.loc[("Comm-only", level), "oob_rate"]
            >= info_indexed.loc[("Comm+Energy", level), "oob_rate"] - 1e-9
        )
        for level in LEVELS
    )
    checks.append(
        "- Comm-only "
        + ("shows" if comm_closure_worse else "does not consistently show")
        + " poorer closure than Comm+Energy across all three intensities."
    )

    e210_notes = []
    e210_data = info[info["axis"] == "information_E210"]
    if not e210_data.empty:
        for method in ["Comm-only", "Energy-only", "Comm+Energy"]:
            subset = e210_data[e210_data["method"] == method].set_index("intensity")
            e210_notes.append(
                f"- E210 {method}: medium/high R_task = "
                f"{subset.loc['medium', 'R_task_mean']:.3f}/{subset.loc['high', 'R_task_mean']:.3f}; "
                f"energy termination = "
                f"{subset.loc['medium', 'energy_rate']:.3f}/{subset.loc['high', 'energy_rate']:.3f}."
            )

    report = f"""# Communication Degradation Scan Report

## Configuration

- Output root: `{root}`
- Pre-run backup: `{backup_dir}`
- Task distribution and conflict: G2--M2.
- Intensities: low/medium/high use 0.75/1.00/1.25 times the current medium parameters. Calibration showed clear separation without universal high-tier failure, so the initial multipliers were retained.
- Seeds: 0, 1, 2; episodes per seed: 5.
- C1 represents distance-decay-dominant degradation; C2 represents local-shadow-dominant degradation.
- POIs, no-fly and degradation regions, start state, energy settings, and random seeds remain unchanged across intensity levels.
- Architecture and information scans use `Ts=40`. In the established coupling ablation presets, Periodic Goal uses `Ts=200`, while Event-driven Goal and Full Coupling use `Ts=40`; each method's period remains fixed across all three intensity levels.
- The active simulator has no independent bandwidth attenuation coefficient. Bandwidth is computed as `1e6 * (1 - loss_p)`, so bandwidth changes indirectly with loss.
- The Full Coupling system file is an alias of the FDLC full-stack configuration. Its C2 results reuse the configuration-identical architecture runs with the same seeds and episodes instead of duplicating stochastic runs.
- Link calibration samples the actual link model on a 31-by-31 spatial grid with five random seeds. The weak-link ratio uses `loss_p >= 0.5`.
- The existing metrics do not expose a separate energy-constraint activation counter. `energy_rate` is reported as the available energy-termination proxy, together with mean remaining energy.
- No core algorithm code or pre-existing result directory was modified.

## Run Completeness

- Architecture: 18 configurations, 54 seed runs, 270 episodes.
- Coupling: 6 independently executed configurations plus 3 configuration-identical Full Coupling groups reused from the architecture scan; 9 summarized configurations and 135 represented episodes.
- Information: 9 configurations, 27 seed runs, 135 episodes.
- E210 supplement: 6 configurations, 18 seed runs, 90 episodes.
- One interrupted duplicate Full Coupling attempt was preserved under `interrupted_runs/` and excluded from every summary.

## Link Calibration

{link_markdown_table(link_stats)}

## Architecture Scan

{chr(10).join(notes)}

## Coupling Scan

{chr(10).join(coupling_notes)}

## Information Scan

{chr(10).join(info_notes)}

## E210 Information Scan

{chr(10).join(e210_notes) if e210_notes else "- Not run."}

## Result Integrity Checks

{chr(10).join(checks)}

- FDLC has the highest absolute R_task in both C1 and C2 at every tested intensity, but the claim that its low-to-high decline is always the smallest is supported only for C2.
- The C2 FDLC result is non-monotonic (medium exceeds low and high). This is reported as observed rather than adjusted to fit a monotonic narrative.
- Comm-only has higher raw R_task than Comm+Energy at all three base-energy intensities, while Comm+Energy has better return-home and out-of-bound closure metrics.
- The report records observed trends without altering data.
- A non-monotonic result should be interpreted as interaction between communication conditions, task selection, termination behavior, and stochastic execution, not as a failed data point.
- Representative claims should use the CSV summaries and error bars; no single episode is used as aggregate evidence.

## Generated Files

- `architecture_scan_summary.csv`
- `coupling_scan_summary.csv`
- `information_scan_summary.csv`
- `link_statistics.csv`
- `fig_architecture_degradation_scan.pdf/.png`
- `fig_coupling_degradation_scan.pdf/.png`
- `fig_information_degradation_scan.pdf/.png`
- `comm_degradation_scan_xelatex_draft.tex`
"""
    (root / "comm_degradation_scan_report.md").write_text(report, encoding="utf-8")

    link_rows = []
    level_cn = {"low": "低", "medium": "中", "high": "高"}
    for row in link_stats.itertuples(index=False):
        link_rows.append(
            f"    {row.comm_case} & {level_cn[row.intensity]} & "
            f"{row.mean_loss_p:.3f} & {row.mean_delay_s:.3f} & "
            f"{row.mean_bandwidth_bps / 1e6:.3f} & "
            f"{row.weak_link_ratio_loss_ge_0_5:.3f} \\\\"
        )

    figure_dir = f"../results/{root.name}"
    latex = r"""% ===== 建议插入第4章实验设置 =====
\subsection{通信退化强度设置}
在保持兴趣点分布、空间风险区域、初始位置、能量参数、慢环周期和随机种子不变的条件下，本文设置低、中、高三档通信退化强度。中档采用当前基准通信配置，低档和高档分别按 $0.75$ 和 $1.25$ 的倍率调整距离衰减或局部遮挡导致的丢包、时延均值及随机扰动幅值。对于局部遮挡机制，倍率同时作用于遮挡区域的附加丢包。当前仿真器中的带宽由瞬时丢包率通过 $b_t=10^6(1-\ell_t)$ 计算，因此带宽随丢包退化同步变化，而未单独设置带宽衰减系数。三档链路统计见表~\ref{tab:comm-degradation-link-stats}。

\begin{table}[htbp]
  \centering
  \caption{不同通信退化强度下的链路统计}
  \label{tab:comm-degradation-link-stats}
  \begin{tabular}{llcccc}
    \toprule
    通信机制 & 强度 & 平均丢包率 & 平均时延/s & 平均带宽/Mbps & 弱链路比例 \\
    \midrule
__LINK_TABLE_ROWS__
    \bottomrule
  \end{tabular}
\end{table}

% ===== 建议插入第5章实验分析 =====
\subsection{通信退化鲁棒性分析}
图~\ref{fig:comm-scan-architecture} 比较了两类通信退化机制下三种决策结构的有效任务完成率。误差线表示不同随机种子结果的标准差。

__ARCHITECTURE_TEXT__

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\textwidth]{__FIGURE_DIR__/fig_architecture_degradation_scan.pdf}
  \caption{不同通信退化强度下的决策结构对比}
  \label{fig:comm-scan-architecture}
\end{figure}

图~\ref{fig:comm-scan-coupling} 给出了局部遮挡主导场景下不同耦合机制的有效任务完成率和平均回传时延。

__COUPLING_TEXT__

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\textwidth]{__FIGURE_DIR__/fig_coupling_degradation_scan.pdf}
  \caption{不同通信退化强度下的耦合机制消融结果}
  \label{fig:comm-scan-coupling}
\end{figure}

图~\ref{fig:comm-scan-information} 比较了三种信息利用方式的有效任务完成率、返航终止率和越界终止率。

__INFORMATION_TEXT__

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.98\textwidth]{__FIGURE_DIR__/fig_information_degradation_scan.pdf}
  \caption{不同通信退化强度下的信息利用消融结果}
  \label{fig:comm-scan-information}
\end{figure}

__E210_TEXT__
"""
    latex = (
        latex.replace("__LINK_TABLE_ROWS__", "\n".join(link_rows))
        .replace("__FIGURE_DIR__", figure_dir)
        .replace("__ARCHITECTURE_TEXT__", architecture_latex_text(arch))
        .replace("__COUPLING_TEXT__", coupling_latex_text(coupling))
        .replace("__INFORMATION_TEXT__", information_latex_text(base_info))
        .replace("__E210_TEXT__", e210_latex_text(e210_data))
    )
    (root / "comm_degradation_scan_xelatex_draft.tex").write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    main()
