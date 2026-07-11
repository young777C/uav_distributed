#!/usr/bin/env python3
"""
Paper: paper1_v2
Purpose: Generate LaTeX tables and PDF figures for Level 2 SITL validation.
         Style follows Autonomous Robots 图片规范 and matches all_scenes_c3.png
         (Okabe–Ito palette, Arial/Helvetica, 174mm double-column width).
Inputs:  results_v2/level2_sitl/<run_dir>/
Outputs: <run_dir>/figures/
         - fig_level2_trajectory_3d.pdf      (3D perspective: SITL vs SIM)
         - fig_level2_timing_comparison.pdf   (bar chart: per-component timing)
         - fig_level2_budget_utilization.pdf  (horizontal bar: P99/T_f %)
         - fig_level2_trajectory_comparison.pdf (2D top-down overlay)
         - fig_level2_fsm_timeline.pdf         (FSM mode timeline)
         - table_level2_timing.tex / table_level2_budget.tex / table_deployment_levels.tex
"""

from __future__ import annotations

import argparse
import json
import logging
import string
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Style configuration — matches all_scenes_c3.png (plot_scenes.py)
# ═══════════════════════════════════════════════════════════════════

_AR_DOUBLE_COL_MM = 174.0
_AR_SINGLE_COL_MM = 84.0
_AR_MAX_HEIGHT_MM = 234.0
_AR_AXIS_LABEL_PT = 9.5
_AR_TICK_PT = 8.5
_AR_PANEL_PT = 9.5
_AR_LEGEND_PT = 8.0
_AR_LINE_PT = 0.9
_AR_PNG_DPI = 600

# Okabe–Ito palette (distinguishable in colour and greyscale)
_POI_COLOR = "#E69F00"
_POI_EDGE = "#4D4D4D"
_GCS_COLOR = "#0072B2"
_GCS_EDGE = "#FFFFFF"
_SITL_COLOR = "#0072B2"      # blue — SITL (PX4+Gazebo)
_SIM_COLOR = "#D55E00"       # vermillion — pure simulation
_DEADLINE_COLOR = "#999999"  # grey — deadline reference
_GRID_ALPHA = 0.18
_GRID_LW = 0.5

# FSM mode colours (Okabe–Ito derived, distinguishable)
_FSM_COLORS = {
    "sins":  "#009E73",   # bluish green — inspection
    "stx":   "#E69F00",   # orange — transmit
    "srec":  "#56B4E9",   # sky blue — recovery
    "ssafe": "#CC79A7",   # reddish purple — safety
    "sback": "#F0E442",   # yellow — backtrack
}

_FSM_LABELS = {
    "sins":  "Sins (Inspection)",
    "stx":   "Stx (Transmit)",
    "srec":  "Srec (Recovery)",
    "ssafe": "Ssafe (Safety)",
    "sback": "Sback (Backtrack)",
}

# Comm heatmap defaults (distance-decay, matching phase1_px4 scene)
_COMM_D0 = 0.0
_COMM_D1 = 150.0
_COMM_LMIN = 0.05
_COMM_LMAX = 0.50

_FONT_WARNED = False


def _mm_to_in(mm: float) -> float:
    return float(mm) / 25.4


def _register_arial_if_needed() -> None:
    global _FONT_WARNED
    fonts = {f.name for f in mpl.font_manager.fontManager.ttflist}
    if "Arial" in fonts or "Helvetica" in fonts:
        return
    for path in (
        Path("/mnt/c/Windows/Fonts/arial.ttf"),
        Path("/mnt/c/Windows/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ):
        if not path.is_file():
            continue
        mpl.font_manager.fontManager.addfont(str(path))
        if "Arial" in {f.name for f in mpl.font_manager.fontManager.ttflist}:
            return


def configure_ar_style() -> None:
    """Arial-first rcParams matching all_scenes_c3.png."""
    global _FONT_WARNED
    _register_arial_if_needed()
    fonts = {f.name for f in mpl.font_manager.fontManager.ttflist}
    if "Arial" in fonts:
        family = ["Arial", "Helvetica", "DejaVu Sans"]
    elif "Helvetica" in fonts:
        family = ["Helvetica", "DejaVu Sans"]
    else:
        family = ["DejaVu Sans"]
        if not _FONT_WARNED:
            logging.getLogger(__name__).warning(
                "Arial/Helvetica not found; using DejaVu Sans."
            )
            _FONT_WARNED = True

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": family,
        "font.size": _AR_TICK_PT,
        "axes.labelsize": _AR_AXIS_LABEL_PT,
        "axes.titlesize": _AR_AXIS_LABEL_PT,
        "xtick.labelsize": _AR_TICK_PT,
        "ytick.labelsize": _AR_TICK_PT,
        "legend.fontsize": _AR_LEGEND_PT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": _AR_LINE_PT,
        "grid.linewidth": _GRID_LW,
        "lines.linewidth": _AR_LINE_PT,
        "savefig.dpi": _AR_PNG_DPI,
        "savefig.bbox": "tight",
        "xtick.major.width": _AR_LINE_PT,
        "ytick.major.width": _AR_LINE_PT,
    })


def _panel_tag(index: int) -> str:
    return f"({string.ascii_lowercase[index]})"


# ═══════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════

TIMING_KEYS = ["t_read_telemetry_ms", "t_fdlc_decision_ms",
               "t_send_command_ms", "t_loop_total_ms"]

COMPONENT_LABELS = {
    "t_read_telemetry_ms":  "MAVLink Telemetry",
    "t_fdlc_decision_ms":   "FDLC Decision",
    "t_send_command_ms":    "MAVLink Send",
    "t_loop_total_ms":      "Loop Total",
}


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def timing_stats(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    result = {}
    for key in TIMING_KEYS:
        arr = np.array([r[key] for r in records])
        result[key] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }
    return result


def _reconstruct_altitude(traj: List[Dict], takeoff_steps: int = 27,
                          cruise_alt_m: float = 8.0) -> np.ndarray:
    z = np.zeros(len(traj))
    for i in range(len(traj)):
        if i < takeoff_steps:
            z[i] = cruise_alt_m * (i / max(1, takeoff_steps))
        else:
            z[i] = cruise_alt_m
    return z


def _comm_quality_grid(
    n_min: float, n_max: float, e_min: float, e_max: float,
    gcs: Tuple[float, float], grid_n: int = 200,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Compute expected link-quality (1−loss) grid via distance-decay model."""
    xs = np.linspace(n_min, n_max, grid_n, dtype=np.float64)
    ys = np.linspace(e_min, e_max, grid_n, dtype=np.float64)
    x, y = np.meshgrid(xs, ys, indexing="xy")
    dist = np.hypot(x - gcs[0], y - gcs[1])
    t = np.clip((dist - _COMM_D0) / max(1e-6, _COMM_D1 - _COMM_D0), 0.0, 1.0)
    loss = _COMM_LMIN + t * (_COMM_LMAX - _COMM_LMIN)
    loss = np.clip(loss, 0.0, 1.0)
    quality = 1.0 - loss  # 1 = perfect, 0 = unusable
    return quality, (n_min, n_max, e_min, e_max)


# ═══════════════════════════════════════════════════════════════════
# Figure 1: 3D trajectory perspective   (main figure)
# ═══════════════════════════════════════════════════════════════════

def plot_trajectory_3d(
    traj_sitl: List[Dict], traj_sim: List[Dict],
    pois: List[Tuple[float, float]], gcs: Tuple[float, float],
    output_path: Path,
    cruise_alt_m: float = 8.0,
    n_min: float = -100, n_max: float = 100,
    e_min: float = -100, e_max: float = 100,
) -> None:
    """3D perspective: SITL vs SIM with altitude + comm heatmap on ground plane."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import to_rgba

    pos_sitl = np.array([r["pos_ne"] for r in traj_sitl])
    pos_sim = np.array([r["pos_ne"] for r in traj_sim])
    loss_sitl = np.array([r.get("comm_loss_p", 0.0) for r in traj_sitl])

    sitl_takeoff = 30  # ~6s @ 5Hz
    sim_takeoff = 27   # 8m / 1.5 m/s * 5Hz
    z_sitl = _reconstruct_altitude(traj_sitl, sitl_takeoff, cruise_alt_m)
    z_sim = _reconstruct_altitude(traj_sim, sim_takeoff, cruise_alt_m)

    fig_w = _mm_to_in(_AR_DOUBLE_COL_MM)
    fig_h = fig_w * 0.65
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111, projection="3d")

    # ── Ground-plane: comm quality heatmap ───────────────────
    quality, _ = _comm_quality_grid(n_min, n_max, e_min, e_max, gcs)
    xs_q = np.linspace(n_min, n_max, quality.shape[0])
    ys_q = np.linspace(e_min, e_max, quality.shape[1])
    x_q, y_q = np.meshgrid(xs_q, ys_q, indexing="xy")

    # Map RdYlGn values to face colours for the ground surface (range 0.2–1.0)
    from matplotlib.cm import RdYlGn
    quality_clipped = np.clip((quality - 0.2) / 0.8, 0.0, 1.0)
    rgba = RdYlGn(quality_clipped)  # shape (N, N, 4)
    rgba[..., 3] = 0.40             # lower alpha so heatmap stays behind trajectories
    ax.plot_surface(x_q, y_q, np.zeros_like(x_q),
                    facecolors=rgba, rstride=4, cstride=4,
                    shade=False, antialiased=True, zorder=0)

    # ── SITL trajectory (per-segment colour by loss) ──────────
    from matplotlib.cm import coolwarm_r
    norm_loss = mpl.colors.Normalize(vmin=0.0, vmax=0.6)
    for i in range(len(pos_sitl) - 1):
        c = coolwarm_r(norm_loss(loss_sitl[i + 1]))
        ax.plot(pos_sitl[i:i+2, 0], pos_sitl[i:i+2, 1], z_sitl[i:i+2],
                color=c, linewidth=_AR_LINE_PT * 1.5, alpha=0.92,
                solid_capstyle="round", zorder=10)

    # ── SIM trajectory (dashed) ───────────────────────────────
    ax.plot(pos_sim[:, 0], pos_sim[:, 1], z_sim,
            color=_SIM_COLOR, linewidth=_AR_LINE_PT * 1.2,
            linestyle="--", dashes=(6, 4),
            label="Simulation", alpha=0.85, zorder=10)

    # ── Ground projection (faint) ────────────────────────────
    ax.plot(pos_sitl[:, 0], pos_sitl[:, 1], 0.0,
            color="#AAAAAA", linewidth=_AR_LINE_PT * 0.5, alpha=0.22, zorder=1)
    ax.plot(pos_sim[:, 0], pos_sim[:, 1], 0.0,
            color=_SIM_COLOR, linewidth=_AR_LINE_PT * 0.5, alpha=0.22, zorder=1)

    # ── Start / GCS at ground (large, on top) ────────────────
    ax.scatter(*gcs, 0.05, c=_GCS_COLOR, marker="s", s=80,
               edgecolors="white", linewidths=_AR_LINE_PT * 1.2,
               zorder=100, label="Start / GCS")

    # ── Takeoff vertical stem at origin ──────────────────────
    ax.plot([0, 0], [0, 0], [0, cruise_alt_m],
            color="#AAAAAA", linewidth=_AR_LINE_PT * 0.6,
            alpha=0.50, linestyle=":", zorder=1)

    # ── POIs with vertical stems ─────────────────────────────
    poi_arr = np.array(pois)
    for i, (px, py) in enumerate(pois):
        ax.plot([px, px], [py, py], [0, cruise_alt_m],
                color=_POI_COLOR, linewidth=_AR_LINE_PT * 0.6,
                alpha=0.35, linestyle=":", zorder=1)
        ax.scatter(px, py, cruise_alt_m,
                   c=_POI_COLOR, marker="o", s=40,
                   edgecolors=_POI_EDGE, linewidths=_AR_LINE_PT * 1.2,
                   zorder=98)
        ax.text(px, py, cruise_alt_m + 1.5, f"POI{i}",
                fontsize=7.5, color="#8B6914", ha="center", va="bottom",
                fontweight="bold")

    # ── Axes ─────────────────────────────────────────────────
    ax.set_xlabel("East (m)", labelpad=4)
    ax.set_ylabel("North (m)", labelpad=4)
    ax.set_zlabel("Altitude (m)", labelpad=4)
    ax.view_init(elev=28, azim=-55)
    ax.xaxis.set_pane_color((1, 1, 1, 0.6))
    ax.yaxis.set_pane_color((1, 1, 1, 0.6))
    ax.zaxis.set_pane_color((1, 1, 1, 0.6))
    ax.grid(True, alpha=_GRID_ALPHA, linewidth=_GRID_LW)
    ax.set_xlim(n_min, n_max)
    ax.set_ylim(e_min, e_max)
    ax.set_zlim(0, cruise_alt_m * 1.5)

    # ── Legend (all elements with explicit proxies) ───────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    heat_proxy = Patch(facecolor="#88CCEE", edgecolor="#4D4D4D",
                        linewidth=_AR_LINE_PT, alpha=0.65,
                        label="Link quality heatmap")
    sitl_proxy = Line2D([0], [0], color="#7F7F7F", linewidth=_AR_LINE_PT * 1.5,
                        label="SITL (colour = loss)")
    sim_proxy = Line2D([0], [0], color=_SIM_COLOR, linewidth=_AR_LINE_PT * 1.2,
                       linestyle="--", dashes=(6, 4), label="Simulation")
    gcs_proxy = Line2D([0], [0], marker="s", color="w",
                        markerfacecolor=_GCS_COLOR, markeredgecolor="white",
                        markeredgewidth=1.2, markersize=7,
                        linestyle="none", label="Start / GCS")
    poi_proxy = Line2D([0], [0], marker="o", color="w",
                        markerfacecolor=_POI_COLOR, markeredgecolor=_POI_EDGE,
                        markeredgewidth=1.2, markersize=6,
                        linestyle="none", label="POI")

    ax.legend(handles=[heat_proxy, gcs_proxy, poi_proxy, sitl_proxy, sim_proxy],
              loc="upper left", framealpha=0.94, fontsize=_AR_LEGEND_PT,
              edgecolor="#666666", fancybox=False, handlelength=1.6,
              borderpad=0.5, labelspacing=0.3)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path / "fig_level2_trajectory_3d.pdf",
                format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[SAVE] {output_path / 'fig_level2_trajectory_3d.pdf'}")


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Timing comparison bar chart
# ═══════════════════════════════════════════════════════════════════

def plot_timing_comparison(
    stats_sitl: Dict, stats_sim: Dict, output_path: Path,
) -> None:
    """Bar chart: SITL vs SIM per-component mean timing with P95 whiskers."""
    components = ["t_read_telemetry_ms", "t_fdlc_decision_ms",
                  "t_send_command_ms", "t_loop_total_ms"]
    labels = [COMPONENT_LABELS[c] for c in components]

    sitl_means = [stats_sitl[c]["mean"] for c in components]
    sitl_p95s = [stats_sitl[c]["p95"] for c in components]
    sim_means = [stats_sim[c]["mean"] for c in components]
    sim_p95s = [stats_sim[c]["p95"] for c in components]

    fig_w = _mm_to_in(_AR_DOUBLE_COL_MM)
    fig_h = fig_w * 0.45
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x = np.arange(len(components))
    width = 0.33

    b1 = ax.bar(x - width / 2, sitl_means, width,
                color=_SITL_COLOR, edgecolor="white", linewidth=0.4,
                label="SITL (mean)", zorder=3)
    for i, (m, p) in enumerate(zip(sitl_means, sitl_p95s)):
        ax.plot([x[i] - width / 2, x[i] - width / 2], [m, p],
                color="#333333", linewidth=_AR_LINE_PT * 0.9,
                marker="_", markersize=3.5, zorder=4)

    b2 = ax.bar(x + width / 2, sim_means, width,
                color=_SIM_COLOR, edgecolor="white", linewidth=0.4,
                label="Simulation (mean)", zorder=3)
    for i, (m, p) in enumerate(zip(sim_means, sim_p95s)):
        ax.plot([x[i] + width / 2, x[i] + width / 2], [m, p],
                color="#333333", linewidth=_AR_LINE_PT * 0.9,
                marker="_", markersize=3.5, zorder=4)

    for bar, val in zip(b1, sitl_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, color=_SITL_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=_AR_LEGEND_PT,
              edgecolor="#666666", fancybox=False, handlelength=1.4)
    ax.grid(axis="y", alpha=_GRID_ALPHA, linewidth=_GRID_LW)
    ax.set_ylim(bottom=0)
    for spine in ax.spines.values():
        spine.set_linewidth(_AR_LINE_PT)

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path / "fig_level2_timing_comparison.pdf",
                format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[SAVE] {output_path / 'fig_level2_timing_comparison.pdf'}")


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Budget utilization
# ═══════════════════════════════════════════════════════════════════

def plot_budget_utilization(
    stats_sitl: Dict, stats_sim: Dict,
    T_f_ms: float, output_path: Path,
) -> None:
    """Horizontal bar: P99/T_f for SITL and SIM, with deadline reference."""
    sitl_p99 = stats_sitl["t_loop_total_ms"]["p99"]
    sim_p99 = stats_sim["t_loop_total_ms"]["p99"]
    sitl_pct = sitl_p99 / T_f_ms * 100
    sim_pct = sim_p99 / T_f_ms * 100

    fig_w = _mm_to_in(_AR_DOUBLE_COL_MM)
    fig_h = fig_w * 0.22
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    categories = ["SITL\n(PX4+Gazebo)", "Simulation\n(Python only)"]
    values = [sitl_pct, sim_pct]
    colors = [_SITL_COLOR, _SIM_COLOR]

    bars = ax.barh(categories, values, height=0.52, color=colors,
                   edgecolor="white", linewidth=0.4, zorder=3)

    for bar, val, p99_ms in zip(bars, values, [sitl_p99, sim_p99]):
        ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%  ({p99_ms:.2f} ms)",
                va="center", fontsize=_AR_TICK_PT, color="#333333")

    ax.axvline(x=100.0, color=_DEADLINE_COLOR, linestyle="--",
               linewidth=_AR_LINE_PT, alpha=0.75, zorder=2)
    ax.text(100.0 + 0.5, len(categories) - 0.7,
            f"Deadline  T_f = {T_f_ms:.0f} ms",
            fontsize=7, color=_DEADLINE_COLOR, va="top")

    ax.set_xlabel("P99 loop time / fast-loop budget (%)")
    ax.grid(axis="x", alpha=_GRID_ALPHA, linewidth=_GRID_LW)
    ax.set_xlim(0, max(125, max(values) * 1.4))
    for spine in ax.spines.values():
        spine.set_linewidth(_AR_LINE_PT)


    fig.tight_layout(pad=0.5)
    fig.savefig(output_path / "fig_level2_budget_utilization.pdf",
                format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[SAVE] {output_path / 'fig_level2_budget_utilization.pdf'}")


# ═══════════════════════════════════════════════════════════════════
# Figure 4: 2D trajectory overlay + FSM timeline (combined)
# ═══════════════════════════════════════════════════════════════════

def plot_trajectory_2d(
    traj_sitl: List[Dict], traj_sim: List[Dict],
    pois: List[Tuple[float, float]], gcs: Tuple[float, float],
    output_path: Path,
    n_min: float = -100, n_max: float = 100,
    e_min: float = -100, e_max: float = 100,
) -> None:
    """2D top-down trajectory with comm-quality heatmap (like all_scenes_c3)."""
    pos_sitl = np.array([r["pos_ne"] for r in traj_sitl])
    pos_sim = np.array([r["pos_ne"] for r in traj_sim])

    # Per-step link loss along SITL trajectory
    loss_sitl = np.array([r.get("comm_loss_p", 0.0) for r in traj_sitl])

    fig_w = _mm_to_in(_AR_DOUBLE_COL_MM)
    fig_h = fig_w * 0.55
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Background: comm quality heatmap (RdYlGn) ────────────
    quality, _ = _comm_quality_grid(n_min, n_max, e_min, e_max, gcs)
    im = ax.imshow(
        quality, extent=(n_min, n_max, e_min, e_max),
        origin="lower", aspect="equal", cmap="RdYlGn",
        vmin=0.2, vmax=1.0, alpha=0.40, interpolation="bilinear",
        zorder=0,
    )

    # ── SITL trajectory — colour-coded by loss ────────────────
    points_sitl = pos_sitl.T.reshape(-1, 1, 2)
    segments_sitl = np.concatenate([points_sitl[:-1], points_sitl[1:]], axis=1)
    from matplotlib.collections import LineCollection
    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.6)
    lc_sitl = LineCollection(segments_sitl, array=loss_sitl[1:],
                              cmap="coolwarm_r", norm=norm,
                              linewidths=_AR_LINE_PT * 1.5, alpha=0.92,
                              capstyle="round", zorder=10)
    ax.add_collection(lc_sitl)

    # ── SIM trajectory (dashed, single colour) ────────────────
    ax.plot(pos_sim[:, 0], pos_sim[:, 1],
            color=_SIM_COLOR, linewidth=_AR_LINE_PT * 1.2,
            linestyle="--", dashes=(6, 4),
            label="Simulation", alpha=0.85, zorder=9)

    # ── Start / GCS ──────────────────────────────────────────
    ax.scatter(*gcs, c=_GCS_COLOR, marker="s", s=80,
               edgecolors="white", linewidths=_AR_LINE_PT * 1.2,
               zorder=100, label="Start / GCS")

    # ── POIs ─────────────────────────────────────────────────
    poi_arr = np.array(pois)
    ax.scatter(poi_arr[:, 0], poi_arr[:, 1],
               c=_POI_COLOR, marker="o", s=40,
               edgecolors=_POI_EDGE, linewidths=_AR_LINE_PT * 1.2,
               zorder=100, label="POI")
    for i, (px, py) in enumerate(pois):
        ax.annotate(str(i), (px, py), textcoords="offset points",
                    xytext=(5, 6), fontsize=7.5, color="#8B6914", fontweight="bold")

    # ── Colourbar for link quality ───────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label("Expected link quality (1 − loss)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5, width=_AR_LINE_PT, length=3)
    cbar.outline.set_linewidth(_AR_LINE_PT)

    # ── Legend proxies (all elements) ─────────────────────────
    from matplotlib.lines import Line2D
    heat_proxy = patches.Patch(facecolor="#88CCEE", edgecolor="#4D4D4D",
                                linewidth=_AR_LINE_PT, alpha=0.65,
                                label="Link quality heatmap")
    sitl_proxy = Line2D([0], [0], color="#7F7F7F", linewidth=_AR_LINE_PT * 1.5,
                        label="SITL (colour = loss)")
    sim_proxy = Line2D([0], [0], color=_SIM_COLOR, linewidth=_AR_LINE_PT * 1.2,
                       linestyle="--", dashes=(6, 4), label="Simulation")
    gcs_proxy = Line2D([0], [0], marker="s", color="w",
                        markerfacecolor=_GCS_COLOR, markeredgecolor="white",
                        markeredgewidth=1.2, markersize=7,
                        linestyle="none", label="Start / GCS")
    poi_proxy = Line2D([0], [0], marker="o", color="w",
                        markerfacecolor=_POI_COLOR, markeredgecolor=_POI_EDGE,
                        markeredgewidth=1.2, markersize=6,
                        linestyle="none", label="POI")

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_xlim(n_min, n_max)
    ax.set_ylim(e_min, e_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=_GRID_ALPHA, linewidth=_GRID_LW)
    for spine in ax.spines.values():
        spine.set_linewidth(_AR_LINE_PT)

    ax.legend(handles=[heat_proxy, gcs_proxy, poi_proxy, sitl_proxy, sim_proxy],
              loc="lower left", framealpha=0.94, fontsize=_AR_LEGEND_PT,
              edgecolor="#666666", fancybox=False, handlelength=1.4,
              borderpad=0.5, labelspacing=0.3)


    fig.tight_layout(pad=0.5)
    fig.savefig(output_path / "fig_level2_trajectory_comparison.pdf",
                format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[SAVE] {output_path / 'fig_level2_trajectory_comparison.pdf'}")


# ═══════════════════════════════════════════════════════════════════
# Figure 5: FSM mode timeline
# ═══════════════════════════════════════════════════════════════════

def plot_fsm_timeline(
    traj_sitl: List[Dict], output_path: Path,
) -> None:
    """FSM mode colour strip over wall-clock time."""
    modes = [r["fsm_mode"] for r in traj_sitl]
    times = [r["t_wall_s"] for r in traj_sitl]

    segments = []
    cur = modes[0]; t0 = times[0]
    for i in range(1, len(modes)):
        if modes[i] != cur:
            segments.append((cur, t0, times[i - 1]))
            cur = modes[i]; t0 = times[i]
    segments.append((cur, t0, times[-1]))

    fig_w = _mm_to_in(_AR_DOUBLE_COL_MM)
    fig_h = fig_w * 0.13
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    seen = set()
    for mode, t0, t1 in segments:
        c = _FSM_COLORS.get(mode, "#AAAAAA")
        lbl = _FSM_LABELS.get(mode, mode)
        kw = {"label": lbl} if mode not in seen else {}
        ax.barh(0, t1 - t0, left=t0, height=0.55, color=c,
                edgecolor="white", linewidth=0.3, **kw)
        seen.add(mode)

    ax.legend(loc="upper right", framealpha=0.92, fontsize=7,
              edgecolor="#666666", fancybox=False,
              handlelength=1.2, ncol=len(seen))

    ax.set_xlabel("Wall-clock time (s)")
    ax.set_yticks([])
    ax.set_xlim(0, times[-1])
    for spine in ax.spines.values():
        spine.set_linewidth(_AR_LINE_PT)


    fig.tight_layout(pad=0.4)
    fig.savefig(output_path / "fig_level2_fsm_timeline.pdf",
                format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[SAVE] {output_path / 'fig_level2_fsm_timeline.pdf'}")


# ═══════════════════════════════════════════════════════════════════
# LaTeX tables
# ═══════════════════════════════════════════════════════════════════

def write_timing_table(stats_sitl: Dict, stats_sim: Dict, output_path: Path) -> None:
    rows = [
        ("MAVLink Telemetry", "t_read_telemetry_ms"),
        ("FDLC Decision",      "t_fdlc_decision_ms"),
        ("MAVLink Send",       "t_send_command_ms"),
        ("Loop Total",         "t_loop_total_ms"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Level~1 (task-level simulation) vs.\ Level~2 (PX4 SITL) fast-loop timing.}",
        r"\label{tab:level2_timing}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Level~2 --- PX4 SITL}"
        r"& \multicolumn{3}{c}{Level~1 --- Simulation} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Component & Mean (ms) & P95 (ms) & P99 (ms) & Mean (ms) & P95 (ms) & P99 (ms) \\",
        r"\midrule",
    ]
    for label, key in rows:
        s = stats_sitl[key]; p = stats_sim[key]
        lines.append(
            f"{label} & {s['mean']:.3f} & {s['p95']:.3f} & {s['p99']:.3f} & "
            f"{p['mean']:.3f} & {p['p95']:.3f} & {p['p99']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output_path / "table_level2_timing.tex").write_text("\n".join(lines) + "\n")
    print(f"[SAVE] {output_path / 'table_level2_timing.tex'}")


def write_budget_table(stats_sitl: Dict, stats_sim: Dict,
                       T_f_ms: float, timing_sitl: List[Dict],
                       timing_sim: List[Dict], output_path: Path) -> None:
    sitl_p99 = stats_sitl["t_loop_total_ms"]["p99"]
    sim_p99 = stats_sim["t_loop_total_ms"]["p99"]
    sitl_total = np.array([r["t_loop_total_ms"] for r in timing_sitl])
    sim_total = np.array([r["t_loop_total_ms"] for r in timing_sim])
    dl_sitl = int(np.sum(sitl_total > T_f_ms))
    dl_sim = int(np.sum(sim_total > T_f_ms))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fast-loop budget utilisation (T_f = " + f"{T_f_ms:.0f}" + r"\,ms @ 5\,Hz).}",
        r"\label{tab:level2_budget}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & Level~2 (SITL) & Level~1 (Sim.) & Budget & Utilisation \\",
        r"\midrule",
        f"P99 loop time (ms) & {sitl_p99:.2f} & {sim_p99:.2f} & {T_f_ms:.0f} & {sitl_p99/T_f_ms*100:.2f}\\% \\\\",
        f"Max loop time (ms) & {stats_sitl['t_loop_total_ms']['max']:.2f} & "
        f"{stats_sim['t_loop_total_ms']['max']:.2f} & {T_f_ms:.0f} & --- \\\\",
        f"Deadline misses & {dl_sitl} & {dl_sim} & --- & "
        f"{dl_sitl/max(1,len(sitl_total))*100:.2f}\\% \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (output_path / "table_level2_budget.tex").write_text("\n".join(lines) + "\n")
    print(f"[SAVE] {output_path / 'table_level2_budget.tex'}")


def write_deployment_levels_table(output_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{System-level deployment evidence for the FDLC framework.}",
        r"\label{tab:deployment_levels}",
        r"\begin{tabular}{clll}",
        r"\toprule",
        r"Level & Verification Method & Status & Supported Conclusion \\",
        r"\midrule",
        r"Level~1 & Task-level simulation (Python) & Completed & "
        r"Computational cost acceptable at current scale \\",
        r"Level~2 & SITL, PX4, Gazebo (this work) & Completed & "
        r"Software interface and closed-loop timing feasible \\",
        r"Level~3 & Hardware-in-the-loop & Future & "
        r"Real computation and communication link feasible \\",
        r"Level~4 & Real UAV flight test & Future & "
        r"Real-world deployment validity \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (output_path / "table_deployment_levels.tex").write_text("\n".join(lines) + "\n")
    print(f"[SAVE] {output_path / 'table_deployment_levels.tex'}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(description="Level 2 paper figures (all_scenes_c3 style)")
    p.add_argument("--run_dir", type=str,
                   default="results_v2/level2_sitl/run_6poi_120s")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--T_f_ms", type=float, default=200.0)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[ERROR] Not found: {run_dir}"); sys.exit(1)

    out = Path(args.output_dir) if args.output_dir else (run_dir / "figures")
    out.mkdir(parents=True, exist_ok=True)

    configure_ar_style()

    sitl_t = run_dir / "timing_sitl.jsonl"
    sim_t = run_dir / "timing_sim.jsonl"
    sitl_traj = run_dir / "traj_sitl.jsonl"
    sim_traj = run_dir / "traj_sim.jsonl"

    for f in [sitl_t, sim_t]:
        if not f.exists():
            print(f"[ERROR] Missing: {f}"); sys.exit(1)

    timing_sitl = load_jsonl(sitl_t)
    timing_sim = load_jsonl(sim_t)
    stats_sitl = timing_stats(timing_sitl)
    stats_sim = timing_stats(timing_sim)

    # ── Figures ──────────────────────────────────────────────
    # 3D trajectory (main figure)
    if sitl_traj.exists() and sim_traj.exists():
        traj_sitl = load_jsonl(sitl_traj)
        traj_sim = load_jsonl(sim_traj)
        pois = [(50,30), (-60,50), (-70,-40), (30,-60), (80,-20), (-20,80)]
        gcs = (0.0, 0.0)

        plot_trajectory_3d(traj_sitl, traj_sim, pois, gcs, out,
                           n_min=-100, n_max=100, e_min=-100, e_max=100)
        plot_trajectory_2d(traj_sitl, traj_sim, pois, gcs, out,
                           n_min=-100, n_max=100, e_min=-100, e_max=100)
        plot_fsm_timeline(traj_sitl, out)

    plot_timing_comparison(stats_sitl, stats_sim, out)
    plot_budget_utilization(stats_sitl, stats_sim, args.T_f_ms, out)

    # ── Tables ───────────────────────────────────────────────
    write_timing_table(stats_sitl, stats_sim, out)
    write_budget_table(stats_sitl, stats_sim, args.T_f_ms, timing_sitl, timing_sim, out)
    write_deployment_levels_table(out)

    print(f"\n[✓] Done → {out}")


if __name__ == "__main__":
    main()
