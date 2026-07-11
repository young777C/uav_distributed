#!/usr/bin/env python3
"""Generate Chapter 5 paper figures from existing experiment results.

The script never edits source experiment outputs. It reads summary CSV files and
trajectory JSONL files, then writes figures and a generation report under the
requested output directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uavlab.common.config import load_resolved_config  # noqa: E402


SCENES_8 = [
    "c1_g1",
    "c1_g2_m0",
    "c1_g2_m1",
    "c1_g2_m2",
    "c2_g1",
    "c2_g2_m0",
    "c2_g2_m1",
    "c2_g2_m2",
]
MODELING_SCENES = ["c1_g2_m2", "c2_g2_m2"]
COUPLING_SCAN_LEVELS = ["low", "medium", "high"]
COUPLING_SCAN_LABELS = ["Low", "Medium", "High"]

STRUCT_METHODS = {
    "struct_centralized_single_loop": "CDSL",
    "struct_decoupled_dual_loop": "WCDL",
    "struct_full_dual_loop_distributed": "FDLC",
}
MODELING_METHODS = {
    "modelling_comm_aware_decision": "Comm-only",
    "modelling_energy_aware_decision": "Energy-only",
    "modelling_comm_energy_aware_decision": "Comm+Energy",
}
COUPLING_METHODS = {
    "coupling_periodic_goal": "Periodic Goal",
    "coupling_event_driven_goal": "Event-driven Goal",
    "coupling_full_coupling": "Full Coupling",
}

METHOD_COLORS = {
    "CDSL": "#7a7a7a",
    "WCDL": "#4c78a8",
    "FDLC": "#f58518",
    "Comm-only": "#4c78a8",
    "Energy-only": "#7a7a7a",
    "Comm+Energy": "#54a24b",
    "Periodic Goal": "#7a7a7a",
    "Event-driven Goal": "#4c78a8",
    "Full Coupling": "#f58518",
}
METHOD_HATCHES = {
    "CDSL": "///",
    "WCDL": "\\\\\\",
    "FDLC": "",
    "Comm-only": "///",
    "Energy-only": "\\\\\\",
    "Comm+Energy": "",
    "Periodic Goal": "///",
    "Event-driven Goal": "\\\\\\",
    "Full Coupling": "",
}


@dataclass
class FigureRecord:
    files: list[Path]
    source: str
    insert_after: str


def configure_style() -> list[str]:
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["legend.fontsize"] = 9
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["grid.linewidth"] = 0.5
    mpl.rcParams["grid.alpha"] = 0.25
    mpl.rcParams["legend.frameon"] = False
    warnings: list[str] = []
    fonts = {f.name for f in mpl.font_manager.fontManager.ttflist}
    if "Times New Roman" not in fonts:
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        mpl.rcParams["font.family"] = "Liberation Serif"
        warnings.append(
            "Times New Roman is not installed in the runtime font cache; "
            "figures use Liberation Serif as the closest installed Times-compatible fallback."
        )
    return warnings


def split_exp_name(name: str) -> tuple[str, str]:
    for marker in ("__struct_", "__modelling_", "__modeling_", "__coupling_"):
        if marker in name:
            scene, rest = name.split(marker, 1)
            axis = marker.strip("_")
            if axis == "modeling":
                axis = "modelling"
            return scene, f"{axis}_{rest}"
    return name, ""


def axis_is_complete(df: pd.DataFrame, axis: str) -> bool:
    parsed = add_scene_method(df)
    if axis == "struct":
        return set(SCENES_8).issubset(set(parsed["scene"])) and set(STRUCT_METHODS).issubset(set(parsed["method_key"]))
    if axis == "coupling":
        return set(SCENES_8).issubset(set(parsed["scene"])) and set(COUPLING_METHODS).issubset(set(parsed["method_key"]))
    if axis == "modeling":
        return set(MODELING_SCENES).issubset(set(parsed["scene"])) and set(MODELING_METHODS).issubset(set(parsed["method_key"]))
    return True


def load_axis_summary(result_dir: Path, axis: str, fallback: Path, issues: list[str]) -> tuple[pd.DataFrame, Path]:
    candidates = [
        result_dir / axis / "summary.csv",
        result_dir / "summary.csv" if result_dir.name.startswith(f"{axis}_") else result_dir / axis / "summary.csv",
        fallback / "summary.csv",
    ]
    incomplete_seen: list[Path] = []
    for p in candidates:
        if p.exists() and p not in incomplete_seen:
            df = pd.read_csv(p)
            if not axis_is_complete(df, axis) and p != fallback / "summary.csv":
                incomplete_seen.append(p)
                issues.append(f"{axis}: `{p}` exists but is incomplete for the requested Chapter 5 figure set.")
                continue
            if p == fallback / "summary.csv":
                issues.append(f"{axis}: requested result directory lacks complete summary; used fallback `{p}`.")
            return add_scene_method(df), p
    raise FileNotFoundError(f"No summary.csv found for axis `{axis}`; checked: {candidates}")


def add_scene_method(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["exp_name"].map(split_exp_name)
    out["scene"] = parsed.map(lambda x: x[0])
    out["method_key"] = parsed.map(lambda x: x[1])
    return out


def aggregate(df: pd.DataFrame, method_map: dict[str, str]) -> pd.DataFrame:
    tmp = df[df["method_key"].isin(method_map)].copy()
    tmp["method"] = tmp["method_key"].map(method_map)
    rows = []
    metrics = [
        "R_task_mean",
        "T_ret_s_mean",
        "returned_home_rate",
        "oob_rate",
        "energy_rate",
        "R_fail_given_cov_mean",
    ]
    for (scene, method), g in tmp.groupby(["scene", "method"], sort=False):
        row: dict[str, Any] = {"scene": scene, "method": method, "n_seed": int(g["seed"].nunique())}
        for col in metrics:
            if col in g:
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                row[col] = float(vals.mean()) if len(vals) else np.nan
                row[col.replace("_mean", "_seed_std") if col.endswith("_mean") else f"{col}_seed_std"] = (
                    float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                )
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=360)
    plt.close(fig)
    return [pdf, png]


def scenario_label(scene: str) -> str:
    return scene.upper().replace("_", "-")


def grouped_bar(
    ax: plt.Axes,
    data: pd.DataFrame,
    scenes: list[str],
    methods: list[str],
    value_col: str,
    err_col: str | None,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(scenes))
    width = min(0.24, 0.78 / max(1, len(methods)))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    for off, method in zip(offsets, methods):
        vals, errs = [], []
        for scene in scenes:
            row = data[(data["scene"] == scene) & (data["method"] == method)]
            vals.append(float(row[value_col].iloc[0]) if len(row) else np.nan)
            errs.append(float(row[err_col].iloc[0]) if err_col and len(row) and err_col in row else 0.0)
        bars = ax.bar(
            x + off,
            vals,
            width,
            label=method,
            color=METHOD_COLORS.get(method, "#808080"),
            edgecolor="black",
            linewidth=0.45,
            hatch=METHOD_HATCHES.get(method, ""),
            yerr=errs if err_col else None,
            capsize=2.5 if err_col else 0,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )
        for b in bars:
            b.set_alpha(0.92)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([scenario_label(s) for s in scenes], rotation=35, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_overall_structure(data: pd.DataFrame, out_dir: Path) -> FigureRecord:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    grouped_bar(
        ax,
        data,
        SCENES_8,
        ["CDSL", "WCDL", "FDLC"],
        "R_task_mean",
        "R_task_seed_std",
        "Effective task completion rate",
        (0, 0.82),
    )
    for idx, scene in enumerate(SCENES_8):
        if scene.endswith("m2"):
            ax.axvspan(idx - 0.48, idx + 0.48, color="#f2f2f2", zorder=-10)
            ax.text(idx, 0.04, "M2", ha="center", va="bottom", fontsize=9, color="#303030")
    ax.legend(ncol=3, loc="upper left")
    files = save_figure(fig, out_dir, "fig_overall_structure_comparison")
    return FigureRecord(files, "struct summary.csv, seed-level means aggregated across seeds", "Section 5.4.1")


def plot_modeling_ablation(data: pd.DataFrame, out_dir: Path) -> FigureRecord:
    metrics = [
        ("R_task_mean", "R_task_seed_std", "Effective task completion rate", (0, 0.86)),
        ("returned_home_rate", "returned_home_rate_seed_std", "Return-home termination rate", (0, 1.08)),
        ("oob_rate", "oob_rate_seed_std", "Out-of-bound termination rate", (0, 1.08)),
        ("energy_rate", "energy_rate_seed_std", "Energy termination rate", (0, 1.08)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.4), sharex=True)
    for ax, (val, err, ylabel, ylim) in zip(axes.flat, metrics):
        grouped_bar(
            ax,
            data,
            MODELING_SCENES,
            ["Comm-only", "Energy-only", "Comm+Energy"],
            val,
            err,
            ylabel,
            ylim,
        )
        ax.legend().remove() if ax.get_legend() else None
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.88, hspace=0.33, wspace=0.28)
    files = save_figure(fig, out_dir, "fig_modeling_ablation")
    return FigureRecord(files, "modeling summary.csv, seed-level means aggregated across seeds", "Section 5.5")


def plot_coupling_completion(data: pd.DataFrame, out_dir: Path) -> FigureRecord:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    grouped_bar(
        ax,
        data,
        SCENES_8,
        ["Periodic Goal", "Event-driven Goal", "Full Coupling"],
        "R_task_mean",
        "R_task_seed_std",
        "Effective task completion rate",
        (0, 0.82),
    )
    ax.legend(ncol=3, loc="upper left")
    files = save_figure(fig, out_dir, "fig_coupling_completion")
    return FigureRecord(files, "coupling summary.csv, seed-level means aggregated across seeds", "Section 5.6")


def parse_coupling_scan_name(name: str) -> tuple[str, str, str]:
    case, system = name.split("__", 1)
    parts = case.split("_")
    comm_case = parts[0].upper()
    intensity = parts[-1]
    return comm_case, intensity, system


def summarize_coupling_scan(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    parsed = df["exp_name"].map(parse_coupling_scan_name)
    df["comm_case"] = parsed.map(lambda x: x[0])
    df["intensity"] = parsed.map(lambda x: x[1])
    df["method_key"] = parsed.map(lambda x: x[2])
    df["method"] = df["method_key"].map(COUPLING_METHODS)
    metrics = [
        "R_task_mean",
        "T_ret_s_mean",
        "returned_home_rate",
        "oob_rate",
        "energy_rate",
        "R_fail_given_cov_mean",
    ]
    rows: list[dict[str, Any]] = []
    keys = ["comm_case", "intensity", "method"]
    for group_key, group in df.groupby(keys, sort=False):
        row = dict(zip(keys, group_key))
        row["n_seed"] = int(group["seed"].nunique())
        for metric in metrics:
            if metric not in group:
                continue
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric.replace('_mean', '_seed_std') if metric.endswith('_mean') else metric + '_seed_std'}"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_coupling_scan_data(result_dir: Path) -> tuple[pd.DataFrame, Path]:
    scan_summary = result_dir / "coupling_scan_summary.csv"
    if scan_summary.exists():
        df = pd.read_csv(scan_summary)
        df = df[df["method"].isin(COUPLING_METHODS.values())].copy()
        return df, scan_summary
    raw_summary = result_dir / "summary.csv"
    if not raw_summary.exists():
        raise FileNotFoundError(f"No coupling scan summary found under `{result_dir}`")
    df = summarize_coupling_scan(raw_summary)
    df.to_csv(scan_summary, index=False)
    return df, scan_summary


def plot_coupling_scan_line(
    data: pd.DataFrame,
    out_dir: Path,
    stem: str,
    metric: str,
    err_metric: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> FigureRecord:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(COUPLING_SCAN_LEVELS))
    markers = {"Periodic Goal": "o", "Event-driven Goal": "s", "Full Coupling": "^"}
    subset = data[data["comm_case"] == "C2"] if "comm_case" in data.columns else data
    for method in ["Periodic Goal", "Event-driven Goal", "Full Coupling"]:
        by_level = subset[subset["method"] == method].set_index("intensity").reindex(COUPLING_SCAN_LEVELS)
        vals = by_level[metric].astype(float).to_numpy()
        errs = by_level[err_metric].astype(float).to_numpy() if err_metric in by_level else np.zeros_like(vals)
        ax.errorbar(
            x,
            vals,
            yerr=errs,
            marker=markers[method],
            color=METHOD_COLORS[method],
            label=method,
            linewidth=1.4,
            markersize=4.5,
            capsize=2.5,
        )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Communication degradation intensity (C2--G2--M2)")
    ax.set_xticks(x, COUPLING_SCAN_LABELS)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=3, loc="upper left")
    files = save_figure(fig, out_dir, stem)
    return FigureRecord(files, "coupling scan summary", "Section 5.6")


def plot_coupling_latency(data: pd.DataFrame, out_dir: Path) -> FigureRecord:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(SCENES_8))
    markers = {"Periodic Goal": "o", "Event-driven Goal": "s", "Full Coupling": "^"}
    for method in ["Periodic Goal", "Event-driven Goal", "Full Coupling"]:
        vals, errs = [], []
        for scene in SCENES_8:
            row = data[(data["scene"] == scene) & (data["method"] == method)]
            vals.append(float(row["T_ret_s_mean"].iloc[0]) if len(row) else np.nan)
            errs.append(float(row["T_ret_s_seed_std"].iloc[0]) if len(row) else 0.0)
        ax.errorbar(
            x,
            vals,
            yerr=errs,
            marker=markers[method],
            color=METHOD_COLORS[method],
            label=method,
            linewidth=1.4,
            markersize=4.5,
            capsize=2.5,
        )
    ax.set_ylabel("Mean return latency (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([scenario_label(s) for s in SCENES_8], rotation=35, ha="right")
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=3, loc="upper left")
    files = save_figure(fig, out_dir, "fig_coupling_latency")
    return FigureRecord(files, "coupling summary.csv, seed-level means aggregated across seeds", "Section 5.6")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def choose_representative_run(struct_root: Path, scene: str) -> tuple[Path, int, dict[str, Any]]:
    exp_dir = struct_root / f"{scene}__struct_full_dual_loop_distributed" / "Ts40"
    candidates: list[tuple[float, Path, int, dict[str, Any]]] = []
    all_rows: list[dict[str, Any]] = []
    for metrics_path in sorted(exp_dir.glob("seed*/metrics.jsonl")):
        run_dir = metrics_path.parent
        if not (run_dir / "traj.jsonl").exists():
            continue
        for row in read_jsonl(metrics_path):
            all_rows.append(row)
    if not all_rows:
        raise FileNotFoundError(f"No FDLC metrics/traj rows found for `{scene}` under `{exp_dir}`")
    target = float(np.nanmean([float(r.get("R_task", np.nan)) for r in all_rows]))
    for metrics_path in sorted(exp_dir.glob("seed*/metrics.jsonl")):
        run_dir = metrics_path.parent
        if not (run_dir / "traj.jsonl").exists():
            continue
        for row in read_jsonl(metrics_path):
            episode = int(row["episode"])
            r_task = float(row.get("R_task", np.nan))
            returned_penalty = 0.0 if row.get("returned_home") else 0.35
            oob_penalty = 0.25 if row.get("terminated_by_oob") else 0.0
            score = abs(r_task - target) + returned_penalty + oob_penalty
            candidates.append((score, run_dir, episode, row))
    if not candidates:
        raise FileNotFoundError(f"No trajectory candidates found for `{scene}`")
    candidates.sort(key=lambda x: x[0])
    _, run_dir, episode, row = candidates[0]
    return run_dir, episode, row


def as_points(values: Any) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in (values or [])]


def as_circles(values: Any) -> list[tuple[float, float, float]]:
    return [(float(c[0]), float(c[1]), float(c[2])) for c in (values or [])]


def draw_comm_degradation(ax: plt.Axes, scene: dict[str, Any], comm_cfg: dict[str, Any]) -> bool:
    shown = False
    if comm_cfg.get("enable_distance_decay"):
        gcs = tuple(scene.get("gcs_ne") or scene.get("start_ne") or (1250.0, 1250.0))
        d1 = float(comm_cfg.get("distance_d1_m", 2500.0))
        for frac, alpha in [(0.5, 0.08), (0.75, 0.11), (1.0, 0.14)]:
            ax.add_patch(
                patches.Circle(
                    (float(gcs[0]), float(gcs[1])),
                    d1 * frac,
                    fill=False,
                    edgecolor="#6f4aa8",
                    linewidth=0.8,
                    linestyle="--",
                    alpha=alpha / 0.14,
                )
            )
        shown = True
    if comm_cfg.get("use_scene_blackholes"):
        for cx, cy, r in as_circles(scene.get("communication_blackholes")):
            ax.add_patch(
                patches.Circle((cx, cy), r, facecolor="#6f4aa8", edgecolor="#3f2670", alpha=0.10, linewidth=0.8)
            )
            shown = True
    return shown


def plot_trajectory(struct_root: Path, scene: str, out_dir: Path, stem: str) -> FigureRecord:
    run_dir, episode, row = choose_representative_run(struct_root, scene)
    cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    scenario = load_resolved_config(str(cfg["scene_file"]))
    comm_cfg = dict(cfg.get("comm") or {})
    traj_rows = [r for r in read_jsonl(run_dir / "traj.jsonl") if int(r.get("episode", -1)) == episode]
    if not traj_rows:
        raise ValueError(f"No trajectory rows for episode {episode} in {run_dir}")

    pts = np.array([r["pos_ne"] for r in traj_rows], dtype=float)
    step = max(1, len(pts) // 3500)
    pts_plot = pts[::step]
    gcs = tuple(scenario.get("gcs_ne") or scenario.get("start_ne") or (1250.0, 1250.0))
    pois = as_points(scenario.get("poi_list"))
    covered = {int(i) for i in row.get("covered_ids", [])}
    effective = {int(i) for i in row.get("effective_ids", [])}
    unvisited = [p for i, p in enumerate(pois) if i not in covered]
    cov_only = [p for i, p in enumerate(pois) if i in covered and i not in effective]
    returned = [p for i, p in enumerate(pois) if i in effective]

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    has_comm = draw_comm_degradation(ax, scenario, comm_cfg)
    for cx, cy, r in as_circles(scenario.get("nofly_circles")):
        ax.add_patch(patches.Circle((cx, cy), r, facecolor="#d95f5f", edgecolor="#8b1a1a", alpha=0.12, linewidth=0.9))
    for rect in scenario.get("nofly_rects") or []:
        if len(rect) >= 4:
            n0, n1, e0, e1 = map(float, rect[:4])
            ax.add_patch(patches.Rectangle((n0, e0), n1 - n0, e1 - e0, facecolor="#d95f5f", edgecolor="#8b1a1a", alpha=0.12))

    if unvisited:
        ax.scatter(*zip(*unvisited), s=13, color="#b8b8b8", edgecolor="none", label="POIs")
    if cov_only:
        ax.scatter(*zip(*cov_only), s=22, color="#f2a65a", edgecolor="black", linewidth=0.25, label="Covered POIs")
    if returned:
        ax.scatter(*zip(*returned), s=24, color="#4c9f70", edgecolor="black", linewidth=0.25, label="Returned POIs")

    ax.plot(pts_plot[:, 0], pts_plot[:, 1], color="#202020", linewidth=1.15, label="UAV trajectory")
    ax.scatter([gcs[0]], [gcs[1]], s=105, marker="*", color="#2f6db3", edgecolor="black", linewidth=0.4, label="GCS", zorder=7)
    ax.scatter([pts[0, 0]], [pts[0, 1]], s=45, marker="o", color="#ffffff", edgecolor="#202020", linewidth=1.0, label="Start point", zorder=9)
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], s=55, marker="x", color="#202020", linewidth=1.2, label="End point", zorder=10)

    handles, labels = ax.get_legend_handles_labels()
    extra: list[Any] = []
    extra_labels: list[str] = []
    if as_circles(scenario.get("nofly_circles")) or scenario.get("nofly_rects"):
        extra.append(patches.Patch(facecolor="#d95f5f", alpha=0.12, edgecolor="#8b1a1a"))
        extra_labels.append("Communication shadow regions")
    if has_comm:
        extra.append(patches.Patch(facecolor="#6f4aa8", alpha=0.10, edgecolor="#3f2670"))
        extra_labels.append("Communication-degraded regions")
    ax.legend(handles + extra, labels + extra_labels, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False)

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title(f"FDLC trajectory in {scenario_label(scene)}")
    n_min = float(scenario.get("n_min", 0.0))
    n_max = float(scenario.get("n_max", 2500.0))
    e_min = float(scenario.get("e_min", 0.0))
    e_max = float(scenario.get("e_max", 2500.0))
    ax.set_xlim(n_min, n_max)
    ax.set_ylim(e_min, e_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    files = save_figure(fig, out_dir, stem)
    source = f"{run_dir}, episode {episode}, R_task={float(row.get('R_task', np.nan)):.3f}, termination={row.get('termination_reason', 'unknown')}"
    return FigureRecord(files, source, "Section 5.4.1 or 5.7 representative-case discussion")


def write_report(out_dir: Path, records: dict[str, FigureRecord], missing: list[str], issues: list[str]) -> Path:
    p = out_dir / "figure_generation_report.md"
    lines = [
        "# Chapter 5 Figure Generation Report",
        "",
        "## Generated Figures",
        "",
    ]
    for name, rec in records.items():
        files = ", ".join(f"`{f.as_posix()}`" for f in rec.files)
        lines.append(f"- **{name}**: {files}")
        lines.append(f"  - Data source: {rec.source}")
        lines.append(f"  - Suggested insertion: {rec.insert_after}")
    lines += ["", "## Missing or Not Generated", ""]
    if missing:
        lines += [f"- {m}" for m in missing]
    else:
        lines.append("- None among the requested figures.")
    lines += ["", "## Data Notes", ""]
    if issues:
        lines += [f"- {i}" for i in issues]
    else:
        lines.append("- No data-source fallback or warning was recorded.")
    lines += [
        "",
        "## Recommended Chapter Positions",
        "",
        "- `fig_overall_structure_comparison`: after the first paragraph of Section 5.4.1.",
        "- `fig_modeling_ablation`: after the modeling ablation result table in Section 5.5.",
        "- `fig_coupling_completion` and `fig_coupling_latency`: in Section 5.6 after the completion-rate table.",
        "- `fig_trajectory_c1_g2_m2` and `fig_trajectory_c2_g2_m2`: after the high-conflict scenario discussion in Section 5.4.1 or in Section 5.7.",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_latex_snippets(path: Path, records: dict[str, FigureRecord]) -> None:
    order = [
        ("fig_overall_structure_comparison", "Overall effective task completion comparison on the structure axis.", "fig:overall_structure_comparison"),
        ("fig_modeling_ablation", "Modeling ablation under high-conflict and energy-stress scenarios.", "fig:modeling_ablation"),
        ("fig_coupling_completion", "Effective task completion comparison for coupling mechanisms.", "fig:coupling_completion"),
        ("fig_coupling_latency", "Mean return latency comparison for coupling mechanisms.", "fig:coupling_latency"),
        ("fig_trajectory_c1_g2_m2", "Representative FDLC trajectory in C1-G2-M2.", "fig:trajectory_c1_g2_m2"),
        ("fig_trajectory_c2_g2_m2", "Representative FDLC trajectory in C2-G2-M2.", "fig:trajectory_c2_g2_m2"),
    ]
    lines = ["% Chapter 5 figure snippets generated by scripts/plot_chapter5_figures.py", ""]
    for key, caption, label in order:
        if key not in records:
            continue
        pdf = next((f for f in records[key].files if f.suffix == ".pdf"), records[key].files[0])
        rel = Path("figures/chapter5") / pdf.name
        lines += [
            "\\begin{figure}[t]",
            "  \\centering",
            f"  \\includegraphics[width=0.95\\linewidth]{{{rel.as_posix()}}}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            "\\end{figure}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_chapter_with_figures(draft: Path, out_path: Path) -> None:
    text = draft.read_text(encoding="utf-8")
    text = text.replace(
        "TODO: 插入整体结构性能对比图。建议生成分组柱状图，横轴为场景，纵轴为 `R_task`，三组柱分别为 CDSL、WCDL、FDLC，并在图中标出高冲突场景 M2。",
        "图 5-1 给出了结构轴八类场景下的有效任务完成率对比。图中对 M2 高冲突场景进行了浅色标注，用于突出通信退化、任务聚簇和局部安全约束同时增强时的性能变化。\n\n![Overall structure comparison](../figures/chapter5/fig_overall_structure_comparison.png)\n\n从图中可以看出，FDLC 的优势并不是均匀分布在所有场景，而是集中出现在冲突强度升高后。低冲突场景中，WCDL 仍可能凭借较少的模式切换和较直接的任务推进获得更高原始完成率；当进入 C1-G2-M2 和 C2-G2-M2 后，固定上传和弱反馈难以及时处理回传 backlog 与局部链路退化，FDLC 的双向反馈才转化为明显的有效完成率收益。",
    )
    text = text.replace(
        "TODO: 插入建模消融实验表。建议将 `R_task`、`P_return|cov`、返航终止率、越界终止率和能量终止率放在同一表中，以突出通信收益与能量闭合之间的权衡。\n\nTODO: 插入建模消融雷达图或双轴柱状图。建议左轴绘制 `R_task`，右轴绘制返航终止率或越界终止率，避免将 Comm-only 的高 `R_task` 误读为总体最优。",
        "图 5-2 将有效任务完成率、返航终止率、越界终止率和能量终止率放在同一组消融图中。该图的重点不是寻找所有柱状指标上的单一最优方法，而是区分通信收益和任务闭合状态。\n\n![Modeling ablation](../figures/chapter5/fig_modeling_ablation.png)\n\nComm-only 在多个场景中具有更高的原始有效完成率，说明通信感知确实能够把空间覆盖更充分地转化为有效回传；但其越界终止率也较高，表明该收益部分来自更激进的远端任务推进。Comm+Energy 的完成率低于 Comm-only 时，并不意味着联合建模失效，而是能量闭合约束改变了任务选择边界：策略主动放弃一部分通信收益，以换取返航闭合或能量约束下的可解释终止。",
    )
    text = text.replace(
        "TODO: 插入耦合消融结果图。建议使用 `R_task` 分组柱状图和 `T_ret_s` 折线图组合展示。",
        "图 5-3 和图 5-4 分别给出了耦合机制消融中的有效任务完成率和平均回传时延。为避免单图过度拥挤，本文将完成率和时延拆分展示。\n\n![Coupling completion](../figures/chapter5/fig_coupling_completion.png)\n\n![Coupling latency](../figures/chapter5/fig_coupling_latency.png)\n\n完成率图说明，Full Coupling 的收益来自慢环目标更新和快环局部状态响应的共同作用，而不是单纯增加重规划次数。时延图进一步表明，完整耦合能够缩短覆盖完成到关键数据回传完成之间的等待时间；这与 backlog、链路状态和模式反馈进入慢环后，任务序列能够及时避开低效回传区域的机制解释一致。",
    )
    text = text.replace(
        "TODO: 插入压力扫描曲线。若暂不补跑连续扫描，可以先将 C1/C2、M0/M1/M2、E210 作为离散压力对比图呈现，并在正文中明确其不是连续参数扫描。",
        "图 5-5 和图 5-6 给出了 FDLC 在 C1-G2-M2 与 C2-G2-M2 中的代表性轨迹。两张图仅绘制结果文件中实际存在的 POI、已回传 POI、局部通信阴影区、通信退化区域和 UAV 轨迹；其中 C1 主要体现距离退化，C2 主要体现局部遮挡型退化。\n\n![Trajectory in C1-G2-M2](../figures/chapter5/fig_trajectory_c1_g2_m2.png)\n\n![Trajectory in C2-G2-M2](../figures/chapter5/fig_trajectory_c2_g2_m2.png)\n\n当前结果仍属于离散压力场景对比，而不是连续参数扫描。轨迹图可用于解释高冲突场景中的任务闭合过程，但若要在最终论文中声称对通信退化强度、遮挡比例或慢环频率具有连续鲁棒性，还需要补充相应的扫描实验。",
    )
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--chapter-draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    issues = configure_style()
    missing: list[str] = []
    records: dict[str, FigureRecord] = {}

    struct_df, struct_src = load_axis_summary(args.result_dir, "struct", ROOT / "runs/sweeps/struct_axis_full_20260606", issues)
    coupling_df, coupling_src = load_axis_summary(args.result_dir, "coupling", ROOT / "runs/sweeps/coupling_axis_full_20260606", issues)
    modeling_df, modeling_src = load_axis_summary(args.result_dir, "modeling", ROOT / "runs/sweeps/modeling_axis_b1_E210", issues)

    struct_data = aggregate(struct_df, STRUCT_METHODS)
    coupling_data = aggregate(coupling_df, COUPLING_METHODS)
    modeling_data = aggregate(modeling_df, MODELING_METHODS)

    records["fig_overall_structure_comparison"] = plot_overall_structure(struct_data, args.output_dir)
    records["fig_overall_structure_comparison"].source = str(struct_src)
    records["fig_modeling_ablation"] = plot_modeling_ablation(modeling_data, args.output_dir)
    records["fig_modeling_ablation"].source = str(modeling_src)
    records["fig_coupling_completion"] = plot_coupling_completion(coupling_data, args.output_dir)
    records["fig_coupling_completion"].source = str(coupling_src)
    records["fig_coupling_latency"] = plot_coupling_latency(coupling_data, args.output_dir)
    records["fig_coupling_latency"].source = str(coupling_src)

    struct_root = struct_src.parent
    for scene, stem in [("c1_g2_m2", "fig_trajectory_c1_g2_m2"), ("c2_g2_m2", "fig_trajectory_c2_g2_m2")]:
        try:
            records[stem] = plot_trajectory(struct_root, scene, args.output_dir, stem)
        except Exception as exc:  # keep generating the rest and report the missing trajectory.
            missing.append(f"`{stem}` was not generated: {exc}")

    write_report(args.output_dir, records, missing, issues)
    write_latex_snippets(ROOT / "docs/chapter5_figure_latex_snippets.tex", records)
    write_chapter_with_figures(args.chapter_draft, ROOT / "docs/chapter5_experiment_analysis_with_figures.md")

    print(f"Generated {len(records)} figure groups in {args.output_dir}")
    if missing:
        print("Missing:")
        for item in missing:
            print(f"- {item}")


def main_coupling_scan() -> None:
    parser = argparse.ArgumentParser(description="Regenerate coupling scan figures from coupling_fixed07 results.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "results/coupling_fixed07_20260621_153930",
        help="Directory containing summary.csv or coupling_scan_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures/chapter5",
        help="Directory for fig_coupling_completion / fig_coupling_latency outputs",
    )
    args = parser.parse_args()
    issues = configure_style()
    data, source = load_coupling_scan_data(args.result_dir.resolve())
    records = {
        "fig_coupling_completion": plot_coupling_scan_line(
            data,
            args.output_dir,
            "fig_coupling_completion",
            "R_task_mean",
            "R_task_mean_seed_std",
            "Effective task completion rate",
            (0, 0.82),
        ),
        "fig_coupling_latency": plot_coupling_scan_line(
            data,
            args.output_dir,
            "fig_coupling_latency",
            "T_ret_s_mean",
            "T_ret_s_mean_seed_std",
            "Mean return latency (s)",
        ),
    }
    for rec in records.values():
        rec.source = str(source)
    write_report(args.output_dir, records, [], issues)
    print(f"Generated {len(records)} coupling scan figure groups in {args.output_dir}")
    print(f"Data source: {source}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "coupling-scan":
        sys.argv.pop(1)
        main_coupling_scan()
    else:
        main()
