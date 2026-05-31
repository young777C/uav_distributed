from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from uavlab.common.config import load_config, load_resolved_config
from uavlab.viz.comm_heatmap import imshow_comm_field

# Figure typography (avoid cramped legend / overlapping handle–text)
_TICK_FS = 11
_TITLE_FS = 11
_ROW_LABEL_FS = 10
_SUPTITLE_FS = 14
_LEGEND_FS = 11
_CBAR_TICK_FS = 11
_CBAR_LABEL_FS = 11


def _circle(ax, x: float, y: float, r: float, *, fc: str, ec: str, alpha: float, lw: float = 1.0) -> None:
    ax.add_patch(patches.Circle((x, y), r, facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=lw))


def _plot_comm_quality_heatmap(ax, scene: Dict[str, Any], comm: Dict[str, Any], *, grid_n: int = 200) -> Any:
    im, _lbl = imshow_comm_field(ax, scene, comm, grid_n=grid_n, alpha=0.52, metric="quality")
    return im


def _rect(ax, n0: float, n1: float, e0: float, e1: float, *, fc: str, ec: str, alpha: float, lw: float = 1.0) -> None:
    # scene uses [n_min, n_max, e_min, e_max]
    x = n0
    y = e0
    w = n1 - n0
    h = e1 - e0
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=lw))


def _plot_scene(ax, scene: Dict[str, Any], *, show_blackholes: bool) -> None:
    n_min, n_max = float(scene["n_min"]), float(scene["n_max"])
    e_min, e_max = float(scene["e_min"]), float(scene["e_max"])

    # bounds
    ax.set_xlim(n_min, n_max)
    ax.set_ylim(e_min, e_max)
    ax.set_aspect("equal", adjustable="box")

    # no-fly rects
    for rect in list(scene.get("nofly_rects") or []):
        n0, n1, e0, e1 = map(float, rect)
        _rect(ax, n0, n1, e0, e1, fc="#ff6666", ec="#aa0000", alpha=0.15, lw=1.0)

    # no-fly disks (legacy obstacles_circles kept for old scene files)
    for c in list(scene.get("nofly_circles") or []) + list(scene.get("obstacles_circles") or []):
        x, y, r = map(float, c)
        _circle(ax, x, y, r, fc="#ff6666", ec="#aa0000", alpha=0.15, lw=1.0)

    # comm blackholes (only for C2)
    if show_blackholes:
        for c in list(scene.get("communication_blackholes") or []):
            x, y, r = map(float, c)
            _circle(ax, x, y, r, fc="#7e57c2", ec="#4a148c", alpha=0.12, lw=0.8)

    # POIs
    poi = scene.get("poi_list") or []
    xs = [float(p[0]) for p in poi]
    ys = [float(p[1]) for p in poi]
    ax.scatter(xs, ys, s=26, c="#ffb300", edgecolors="#6d4c41", linewidths=0.6, zorder=5, label="POI")

    # start/goal/GCS
    sx, sy = map(float, scene["start_ne"])
    gx, gy = map(float, scene.get("gcs_ne", scene["start_ne"]))
    ax.scatter([sx], [sy], s=70, c="#1e88e5", edgecolors="white", linewidths=0.8, zorder=6, label="Start/GCS")
    ax.scatter([gx], [gy], s=70, c="#1e88e5", edgecolors="white", linewidths=0.8, zorder=6)

    # goal radius
    goal_x, goal_y = map(float, scene["goal_ne"])
    goal_r = float(scene.get("goal_radius_m", 8.0))
    _circle(ax, goal_x, goal_y, goal_r, fc="none", ec="#1e88e5", alpha=0.9, lw=1.2)

    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.tick_params(axis="both", which="major", labelsize=_TICK_FS)


def main() -> None:
    # uavlab/viz/plot_scenes.py -> repo root is parents[3]
    root = Path(__file__).resolve().parents[3]
    scenes_dir = root / "configs" / "scenes"
    scene_paths = sorted(p for p in scenes_dir.glob("*.yaml") if p.name != "base.yaml")
    if not scene_paths:
        raise SystemExit(f"No scene YAMLs found under {scenes_dir} (expected *.yaml except base.yaml).")

    base_cfg_path = root / "configs" / "base.yaml"
    comm_profiles: List[Tuple[str, Path]] = [
        ("C1", root / "configs" / "comm_profiles" / "c1.yaml"),
        ("C2", root / "configs" / "comm_profiles" / "c2.yaml"),
    ]

    n = len(scene_paths)
    # Manual margins so bottom legend + colorbar do not collide with subplot ticks
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n + 1.4, 9.6), constrained_layout=False)
    if n == 1:
        axes_arr = np.asarray(axes).reshape(2, 1)
    else:
        axes_arr = np.asarray(axes)
        if axes_arr.shape != (2, n):
            raise RuntimeError(f"Unexpected axes shape {axes_arr.shape}, expected (2, {n})")

    first_im = None
    for ci, (comm_label, prof_path) in enumerate(comm_profiles):
        merged = load_config(base_cfg_path, prof_path)
        comm = merged.get("comm") or {}
        for j, scene_path in enumerate(scene_paths):
            ax = axes_arr[ci, j]
            scene = load_resolved_config(scene_path)
            im = _plot_comm_quality_heatmap(ax, scene, comm)
            if first_im is None:
                first_im = im
            show_bh = bool(comm.get("use_scene_blackholes", False))
            _plot_scene(ax, scene, show_blackholes=show_bh)
            ax.set_title(f"{scene_path.stem.replace('_', ' ')} · {comm_label}", fontsize=_TITLE_FS)
            if j == 0:
                row_desc = (
                    "C1: distance decay (F1) + baseline loss"
                    if comm_label == "C1"
                    else "C2: scene blackholes (F2) + baseline loss"
                )
                ax.set_ylabel(row_desc, fontsize=_ROW_LABEL_FS)

    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.20, top=0.88, wspace=0.20, hspace=0.28)

    cbar = fig.colorbar(first_im, ax=axes_arr.ravel().tolist(), fraction=0.042, pad=0.03)
    cbar.set_label("Expected link quality (1 − loss_p), no fading jitter", fontsize=_CBAR_LABEL_FS)
    cbar.ax.tick_params(labelsize=_CBAR_TICK_FS)

    # shared legend (use proxy artists)
    heat_proxy = patches.Patch(facecolor="#88c", edgecolor="none", alpha=0.55)
    poi_proxy = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#ffb300",
        markeredgecolor="#6d4c41",
        markersize=9,
        linestyle="none",
    )
    gcs_proxy = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#1e88e5",
        markeredgecolor="white",
        markersize=10,
        linestyle="none",
    )
    obs_proxy = patches.Patch(facecolor="#666666", edgecolor="#333333", alpha=0.25)
    nf_proxy = patches.Patch(facecolor="#ff6666", edgecolor="#aa0000", alpha=0.15)
    bh_proxy = patches.Patch(facecolor="#7e57c2", edgecolor="#4a148c", alpha=0.12)
    fig.legend(
        handles=[heat_proxy, poi_proxy, gcs_proxy, obs_proxy, nf_proxy, bh_proxy],
        labels=[
            "Heatmap (RdYlGn)",
            "POI",
            "Start / GCS / goal",
            "Obstacle",
            "No-fly",
            "Blackhole (C2)",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=3,
        fontsize=_LEGEND_FS,
        frameon=True,
        fancybox=False,
        edgecolor="#bbbbbb",
        facecolor="#fafafa",
        handlelength=2.6,
        handleheight=1.15,
        handletextpad=1.05,
        columnspacing=3.2,
        borderpad=0.85,
        labelspacing=0.55,
    )

    out = root / "all_scenes.png"
    fig.suptitle(
        "All scenes × comm profiles (C1 / C2): geometry + expected link quality heatmap",
        fontsize=_SUPTITLE_FS,
        y=0.965,
    )
    fig.savefig(out, dpi=180, bbox_inches="tight", pad_inches=0.15)
    print(str(out))


if __name__ == "__main__":
    main()

