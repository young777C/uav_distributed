#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from uavlab.common.config import load_resolved_config
from uavlab.viz.comm_heatmap import imshow_comm_field


Point2D = Tuple[float, float]


def _as_points(x: Any) -> List[Point2D]:
    pts = list(x or [])
    out: List[Point2D] = []
    for p in pts:
        out.append((float(p[0]), float(p[1])))
    return out


def _as_circles(x: Any) -> List[Tuple[float, float, float]]:
    cs = list(x or [])
    out: List[Tuple[float, float, float]] = []
    for c in cs:
        out.append((float(c[0]), float(c[1]), float(c[2])))
    return out


def _in_any_circle(p: Point2D, circles: List[Tuple[float, float, float]]) -> bool:
    x, y = float(p[0]), float(p[1])
    for cx, cy, r in circles:
        if (x - float(cx)) ** 2 + (y - float(cy)) ** 2 <= float(r) ** 2:
            return True
    return False


def _expected_loss_p(
    *,
    pos_ne: Point2D,
    gcs_ne: Point2D,
    comm_cfg: Dict[str, Any],
    blackholes: List[Tuple[float, float, float]],
) -> float:
    """Deterministic plot-only loss_p (distance decay + optional blackhole bump)."""
    enable_distance = bool(comm_cfg.get("enable_distance_decay", False))
    d0 = float(comm_cfg.get("distance_d0_m", 0.0))
    d1 = float(comm_cfg.get("distance_d1_m", 250.0))
    lmin = float(comm_cfg.get("distance_loss_min", 0.05))
    lmax = float(comm_cfg.get("distance_loss_max", 0.60))
    base_loss = float(comm_cfg.get("base_loss", 0.15))

    x, y = float(pos_ne[0]), float(pos_ne[1])
    gx, gy = float(gcs_ne[0]), float(gcs_ne[1])
    d = float(math.hypot(x - gx, y - gy))

    if enable_distance:
        if d <= d0:
            loss = lmin
        elif d >= d1:
            loss = lmax
        else:
            t = (d - d0) / max(1e-9, (d1 - d0))
            loss = lmin + (lmax - lmin) * float(t)
    else:
        loss = base_loss

    if bool(comm_cfg.get("use_scene_blackholes", False)) and _in_any_circle(pos_ne, blackholes):
        loss = float(loss) + float(comm_cfg.get("blackhole_extra_loss", 0.5))

    return float(min(max(loss, 0.0), 0.99))


def _add_figure_north_arrow(fig: Any) -> None:
    # pos_ne[0]=N on x-axis, pos_ne[1]=E on y-axis → geographic north is +x (arrow points right).
    y0 = 0.902
    x0, x1 = 0.828, 0.868
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y0),
        transform=fig.transFigure,
        arrowstyle="-|>,head_width=0.28,head_length=0.26",
        mutation_scale=12,
        color="#202124",
        linewidth=1.35,
        zorder=250,
        clip_on=False,
    )
    fig.add_artist(arr)
    fig.text(
        min(x1 + 0.01, 0.97),
        y0,
        "N",
        transform=fig.transFigure,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#202124",
        zorder=251,
    )


def _mosaic_bottom_legend(
    fig: Any,
    *,
    show_obstacles: bool,
    show_blackholes: bool,
    has_numpy_comm_heatmap: bool,
    heatmap_metric: str,
    legend_anchor_y: float,
) -> None:
    handles: List[Any] = []
    labels: List[str] = []

    if has_numpy_comm_heatmap:
        m = str(heatmap_metric or "loss").strip().lower()
        if m == "quality":
            handles.append(mpatches.Patch(facecolor="#88ccee", edgecolor="none", alpha=0.55))
            labels.append("Heatmap (RdYlGn, 1 − loss_p)")
        else:
            handles.append(mpatches.Patch(facecolor="#f59e0b", edgecolor="#7c2d12", alpha=0.65))
            labels.append("Heatmap (inferno, loss_p)")

    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#9aa0a6", markeredgecolor="#5f6368", markersize=8, linestyle="none"))
    labels.append("POI unvisited")
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#fbbc04", markeredgecolor="#b06000", markersize=8, linestyle="none"))
    labels.append("POI visited, not returned")
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#34a853", markeredgecolor="#137333", markersize=8, linestyle="none"))
    labels.append("POI visited & returned")
    handles.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="#1a73e8", markeredgecolor="#174ea6", markersize=12, linestyle="none"))
    labels.append("GCS / home")
    handles.append(Line2D([0], [0], color="#202124", lw=2.2, linestyle="-"))
    labels.append("Trajectory")
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#ea4335", markeredgecolor="#c5221f", markersize=6, linestyle="none"))
    labels.append("Sample in no-fly")

    if show_obstacles:
        handles.append(mpatches.Circle((0.5, 0.5), 0.25, fill=False, ec="#ea4335", lw=1.2))
        labels.append("No-fly zone")
    if show_blackholes:
        handles.append(mpatches.Circle((0.5, 0.5), 0.25, fill=False, ec="#a142f4", lw=1.0))
        labels.append("Comm blackhole zone")

    ncol = min(4, len(handles))
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, float(legend_anchor_y)),
        ncol=ncol,
        fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor="#bbbbbb",
        facecolor="#fafafa",
        handlelength=2.2,
        handleheight=1.05,
        handletextpad=0.9,
        columnspacing=2.4,
        borderpad=0.65,
        labelspacing=0.45,
    )


def _load_metrics_row(metrics_path: Path, episode: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if int(obj.get("episode", -1)) == int(episode):
                rows.append(obj)
    if not rows:
        raise ValueError(f"No row found for episode={episode} in {metrics_path}")
    return rows[-1]


def _load_traj(
    traj_path: Path, episode: int
) -> Tuple[List[Point2D], List[bool], List[bool], List[float]]:
    pts: List[Point2D] = []
    flags: List[bool] = []
    bh: List[bool] = []
    loss: List[float] = []
    with traj_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if int(obj.get("episode", -1)) != int(episode):
                continue
            p = obj.get("pos_ne")
            if not isinstance(p, list) or len(p) != 2:
                continue
            pts.append((float(p[0]), float(p[1])))
            flags.append(bool(obj.get("in_nofly", obj.get("in_obstacle", False))))
            bh.append(bool(obj.get("in_blackhole", False)))
            try:
                loss.append(float(obj.get("link_loss_p", float("nan"))))
            except Exception:
                loss.append(float("nan"))
    if not pts:
        raise ValueError(f"No trajectory points found for episode={episode} in {traj_path}")
    return pts, flags, bh, loss


def _discover_sweep_run_dirs(sweep_root: Path) -> List[Path]:
    root_res = sweep_root.resolve()
    seen: set[str] = set()
    out: List[Path] = []
    for p in sorted(sweep_root.rglob("traj.jsonl")):
        if p.parent == root_res or "plot_trajectory" in p.parts:
            continue
        d = p.parent.resolve()
        k = str(d)
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def _default_out_path_for_sweep_leaf(*, sweep_root: Path, run_dir: Path, episode: int) -> Path:
    rel = run_dir.relative_to(sweep_root.resolve())
    safe = rel.as_posix().replace("/", "__")
    return Path(f"{safe}__trajectory_ep{int(episode)}.png")


# Paper §6.3.1 structure axis (left → right): Baseline-1 CDSL, Baseline-2 WCDL, Proposed FDLC.
_STRUCT_LABEL_ORDER: Tuple[str, ...] = (
    "struct_centralized_single_loop",
    "struct_decoupled_dual_loop",
    "struct_full_dual_loop_distributed",
)

_STRUCT_DISPLAY_TITLES: Dict[str, str] = {
    "struct_centralized_single_loop": "CDSL\n(Center-dominant Single-loop)",
    "struct_decoupled_dual_loop": "WCDL\n(Weakly Coupled Dual-loop)",
    "struct_full_dual_loop_distributed": "FDLC\n(Feedback-driven Dual-loop Collaboration)",
}

_MODELLING_LABEL_ORDER: Tuple[str, ...] = (
    "modelling_comm_energy_aware_decision",
    "modelling_comm_aware_decision",
    "modelling_energy_aware_decision",
)

# Paper §4.2.2 main sweep (three profiles).
_COUPLING_LABEL_ORDER: Tuple[str, ...] = (
    "coupling_periodic_goal",
    "coupling_event_driven_goal",
    "coupling_full_coupling",
)

# Legacy / appendix run dirs (e.g. ``runs/sweeps/test_couplling/*__coupling_no_*``) — after main three.
_COUPLING_LEGACY_APPENDIX_ORDER: Tuple[str, ...] = (
    "coupling_no_feedback",
    "coupling_no_replan",
    "coupling_no_fast_switching",
    "coupling_full",
)


def _parse_case_struct_from_exp_dir(exp_dir_name: str) -> Tuple[str, str]:
    """
    Split sweep leaf parent name ``{case}__{axis}_{variant}`` into (case, variant_key).

    ``test_struct`` uses ``__struct_*``; ``test_model`` uses ``__modelling_*``; coupling sweeps use ``__coupling_*``.
    If no known axis prefix is found, the whole string becomes ``case`` and ``struct`` is ``default`` (one column mosaic).
    """
    if "__struct_" in exp_dir_name:
        case, rest = exp_dir_name.split("__struct_", 1)
        return case, "struct_" + rest
    if "__modelling_" in exp_dir_name:
        case, rest = exp_dir_name.split("__modelling_", 1)
        return case, "modelling_" + rest
    if "__modeling_" in exp_dir_name:
        case, rest = exp_dir_name.split("__modeling_", 1)
        return case, "modeling_" + rest
    if "__coupling_" in exp_dir_name:
        case, rest = exp_dir_name.split("__coupling_", 1)
        return case, "coupling_" + rest
    return exp_dir_name, "default"


def _struct_sort_key(struct_label: str) -> Tuple[int, str]:
    orders = (
        _STRUCT_LABEL_ORDER,
        _MODELLING_LABEL_ORDER,
        _COUPLING_LABEL_ORDER,
        _COUPLING_LEGACY_APPENDIX_ORDER,
    )
    base = sum(len(o) for o in orders)
    for order in orders:
        try:
            idx = order.index(struct_label)
            return (idx, struct_label)
        except ValueError:
            continue
    return (base, struct_label)


def _mosaic_variant_short_title(struct_label: str) -> str:
    """Human-readable mosaic titles (struct axis uses paper CDSL / WCDL / FDLC names)."""
    s = str(struct_label)
    if s in _STRUCT_DISPLAY_TITLES:
        return _STRUCT_DISPLAY_TITLES[s]
    for prefix in ("struct_", "modelling_", "modeling_", "coupling_"):
        if s.startswith(prefix):
            return s[len(prefix) :]
    return s


def _exp_dir_name_from_run_dir(run_dir: Path) -> str:
    return str(run_dir.parent.parent.name)


def _scene_stem_from_cfg(cfg: Dict[str, Any]) -> str:
    sf = str(cfg.get("scene_file", "") or "").strip()
    if not sf:
        return "unknown_scene"
    return Path(sf).stem


def _load_run_plot_bundle(run_dir: Path, episode: int) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.jsonl"
    traj_path = run_dir / "traj.jsonl"
    resolved_path = run_dir / "resolved_config.json"
    if not resolved_path.exists():
        raise FileNotFoundError(f"resolved_config.json not found under: {run_dir}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl not found under: {run_dir}")
    if not traj_path.exists():
        raise FileNotFoundError(
            f"traj.jsonl not found under: {run_dir}. "
            "Run with runner module `uavlab.paper1.runner.run_vis` (it writes traj.jsonl by default)."
        )

    cfg = json.loads(resolved_path.read_text(encoding="utf-8"))
    scene_file = str(cfg.get("scene_file", ""))
    if not scene_file:
        raise ValueError("scene_file missing from resolved_config.json")
    scene = load_resolved_config(scene_file)
    comm_cfg = dict(cfg.get("comm") or {})

    row = _load_metrics_row(metrics_path, episode=int(episode))
    covered_ids = row.get("covered_ids")
    effective_ids = row.get("effective_ids")
    if covered_ids is None or effective_ids is None:
        raise ValueError(
            "metrics.jsonl missing covered_ids/effective_ids. "
            "Run with runner module `uavlab.paper1.runner.run_vis`."
        )
    covered = set(int(i) for i in covered_ids)
    effective = set(int(i) for i in effective_ids)

    poi_list = _as_points(scene.get("poi_list"))
    gcs = tuple(scene.get("gcs_ne") or scene.get("start_ne") or (0.0, 0.0))
    gcs_ne: Point2D = (float(gcs[0]), float(gcs[1]))
    blackholes = _as_circles(scene.get("communication_blackholes"))

    eff_xy = [poi_list[i] for i in range(len(poi_list)) if i in effective]
    cov_only_xy = [poi_list[i] for i in range(len(poi_list)) if (i in covered and i not in effective)]
    unvisited_xy = [poi_list[i] for i in range(len(poi_list)) if i not in covered]

    pts, in_obs, in_bh, loss_p = _load_traj(traj_path, episode=int(episode))

    return {
        "cfg": cfg,
        "scene": scene,
        "scene_file": scene_file,
        "comm_cfg": comm_cfg,
        "row": row,
        "gcs_ne": gcs_ne,
        "blackholes": blackholes,
        "eff_xy": eff_xy,
        "cov_only_xy": cov_only_xy,
        "unvisited_xy": unvisited_xy,
        "pts": pts,
        "in_obs": in_obs,
        "in_bh": in_bh,
        "loss_p": loss_p,
    }


def _draw_bundle_on_ax(
    ax: Any,
    fig: Any,
    bundle: Dict[str, Any],
    *,
    title: str,
    show_obstacles: bool,
    show_blackholes: bool,
    heatmap_engine: str,
    heatmap_res: int,
    heatmap_alpha: float,
    heatmap_numpy_metric: str,
    overlay_comm: str,
    legend_fontsize: float = 9,
    axis_lim: Optional[Tuple[float, float, float, float]] = None,
    compact_legend: bool = False,
    show_subplot_legend: bool = True,
    defer_comm_cbar: bool = False,
    show_title: bool = True,
    show_axis_labels: bool = True,
    show_dist_home_text: bool = True,
) -> Optional[Tuple[Any, str]]:
    scene = bundle["scene"]
    comm_cfg = bundle["comm_cfg"]
    row = bundle["row"]
    gcs_ne = bundle["gcs_ne"]
    blackholes = bundle["blackholes"]
    eff_xy = bundle["eff_xy"]
    cov_only_xy = bundle["cov_only_xy"]
    unvisited_xy = bundle["unvisited_xy"]
    pts = bundle["pts"]
    in_obs = bundle["in_obs"]
    in_bh = bundle["in_bh"]
    loss_p = bundle["loss_p"]

    if show_title:
        ax.set_title(title, fontsize=max(7.0, legend_fontsize + 0.5))

    comm_cbar_meta: Optional[Tuple[Any, str]] = None
    eng = str(heatmap_engine or "off").strip().lower()
    if eng == "numpy":
        gn = int(max(80, min(240, heatmap_res)))
        im, cbl = imshow_comm_field(
            ax, scene, comm_cfg, grid_n=gn, alpha=float(heatmap_alpha), metric=heatmap_numpy_metric
        )
        if defer_comm_cbar:
            comm_cbar_meta = (im, cbl)
        else:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbl)
    elif eng == "loop":
        n_min = float(scene.get("n_min", 0.0))
        n_max = float(scene.get("n_max", 250.0))
        e_min = float(scene.get("e_min", 0.0))
        e_max = float(scene.get("e_max", 250.0))
        res = int(max(20, heatmap_res))
        xs = [n_min + (n_max - n_min) * (i / (res - 1)) for i in range(res)]
        ys = [e_min + (e_max - e_min) * (j / (res - 1)) for j in range(res)]
        q: List[List[float]] = []
        for yy in ys:
            rowq: List[float] = []
            for xx in xs:
                lp = _expected_loss_p(
                    pos_ne=(xx, yy),
                    gcs_ne=gcs_ne,
                    comm_cfg=comm_cfg,
                    blackholes=blackholes,
                )
                rowq.append(float(max(0.0, 1.0 - lp)))
            q.append(rowq)
        im = ax.imshow(
            q,
            origin="lower",
            extent=(n_min, n_max, e_min, e_max),
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            alpha=float(heatmap_alpha),
            interpolation="nearest",
        )
        cbl = "1 - loss_p" if compact_legend else "Expected link quality (1 - loss_p)"
        if defer_comm_cbar:
            comm_cbar_meta = (im, cbl)
        else:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbl)

    if compact_legend:
        lu, lc, le = "unvisited", "visited, !returned", "visited & returned"
        lh, lt, lo, lbh, llp = "GCS/home", "trajectory", "in obstacle", "in blackhole", "link loss_p"
        s_uv, s_co, s_ef, s_hm, s_tr, s_obs = 14, 18, 20, 55, 0.9, 6
    else:
        lu, lc, le = "unvisited", "visited, not returned", "visited & returned"
        lh, lt, lo, lbh, llp = "GCS/home", "trajectory", "in obstacle", "in blackhole", "link loss_p"
        s_uv, s_co, s_ef, s_hm, s_tr, s_obs = 22, 30, 32, 90, 1.2, 10

    if unvisited_xy:
        ax.scatter([p[0] for p in unvisited_xy], [p[1] for p in unvisited_xy], s=s_uv, c="#9aa0a6", label=lu)
    if cov_only_xy:
        ax.scatter([p[0] for p in cov_only_xy], [p[1] for p in cov_only_xy], s=s_co, c="#fbbc04", label=lc)
    if eff_xy:
        ax.scatter([p[0] for p in eff_xy], [p[1] for p in eff_xy], s=s_ef, c="#34a853", label=le)

    ax.scatter([gcs_ne[0]], [gcs_ne[1]], s=s_hm, c="#1a73e8", marker="*", label=lh)

    if show_obstacles:
        olw = 0.8 if compact_legend else 1.0
        for cx, cy, r in _as_circles(scene.get("nofly_circles")):
            circ = plt.Circle((cx, cy), r, fill=False, lw=olw, ec="#ea4335", alpha=0.75)
            ax.add_patch(circ)
        for rect in list(scene.get("nofly_rects") or []):
            if not rect or len(rect) < 4:
                continue
            n_min, n_max, e_min, e_max = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
            ax.add_patch(
                mpatches.Rectangle(
                    (n_min, e_min),
                    n_max - n_min,
                    e_max - e_min,
                    fill=False,
                    lw=olw,
                    ec="#ea4335",
                    alpha=0.45,
                )
            )

    if show_blackholes:
        olw = 0.8 if compact_legend else 1.0
        for cx, cy, r in blackholes:
            circ = plt.Circle((cx, cy), r, fill=False, lw=olw, ec="#a142f4", alpha=0.7)
            ax.add_patch(circ)

    ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=s_tr, c="#202124", alpha=0.75, label=lt)
    obs_pts = [p for p, f in zip(pts, in_obs) if f]
    if obs_pts:
        ax.scatter([p[0] for p in obs_pts], [p[1] for p in obs_pts], s=s_obs, c="#ea4335", alpha=0.9, label=lo)

    overlay = str(overlay_comm or "").strip().lower()
    if overlay == "blackhole":
        bh_pts = [p for p, f in zip(pts, in_bh) if f]
        if bh_pts:
            ax.scatter([p[0] for p in bh_pts], [p[1] for p in bh_pts], s=s_obs, c="#a142f4", alpha=0.9, label=lbh)
    elif overlay == "loss":
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        sc = ax.scatter(xs, ys, s=max(4, s_obs // 2), c=loss_p, cmap="viridis", alpha=0.85, label=llp)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("link loss_p")

    if show_dist_home_text and "dist_to_home_m" in row:
        if compact_legend:
            dh_txt = f"d_home={float(row['dist_to_home_m']):.0f}m"
            tx, fs = 0.02, max(6.0, legend_fontsize - 1)
        else:
            dh_txt = f"dist_to_home={float(row['dist_to_home_m']):.1f} m"
            tx, fs = 0.01, 9
        ax.text(
            tx,
            0.02 if compact_legend else 0.01,
            dh_txt,
            transform=ax.transAxes,
            fontsize=fs,
            ha="left",
            va="bottom",
        )

    n_min = float(scene.get("n_min", 0.0))
    n_max = float(scene.get("n_max", 250.0))
    e_min = float(scene.get("e_min", 0.0))
    e_max = float(scene.get("e_max", 250.0))
    if axis_lim is not None:
        n_min, n_max, e_min, e_max = axis_lim
    ax.set_xlim(n_min, n_max)
    ax.set_ylim(e_min, e_max)
    ax.set_aspect("equal", adjustable="box")
    if show_axis_labels:
        ax.set_xlabel("N", fontsize=max(7.0, legend_fontsize - 1))
        ax.set_ylabel("E", fontsize=max(7.0, legend_fontsize - 1))
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=max(6.0, legend_fontsize - 2))
    if show_subplot_legend:
        leg_y = 0.02 if compact_legend else 0.15
        ax.legend(loc="lower left", frameon=True, fontsize=legend_fontsize, bbox_to_anchor=(0, leg_y))
    return comm_cbar_meta


def _group_run_dirs_by_scene_stem(run_dirs: List[Path]) -> Dict[str, List[Path]]:
    groups: DefaultDict[str, List[Path]] = defaultdict(list)
    for rd in run_dirs:
        cfg = json.loads((rd / "resolved_config.json").read_text(encoding="utf-8"))
        k = _scene_stem_from_cfg(cfg)
        groups[k].append(rd)
    for k in groups:
        groups[k].sort(key=lambda p: (_exp_dir_name_from_run_dir(p),))
    return dict(groups)


def render_scene_mosaics_for_sweep(
    *,
    out_dir: Path,
    run_dirs: List[Path],
    episode: int,
    show_obstacles: bool,
    show_blackholes: bool,
    heatmap_comm: bool,
    heatmap_res: int,
    heatmap_alpha: float,
    overlay_comm: str,
    mosaic_comm_heatmap: bool,
    mosaic_comm_metric: str,
    mosaic_subplot_bottom: float,
    mosaic_legend_anchor_y: Optional[float],
    mosaic_full_labels: bool,
) -> List[Path]:
    by_scene = _group_run_dirs_by_scene_stem(run_dirs)
    written: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for scene_stem in sorted(by_scene.keys()):
        members = by_scene[scene_stem]
        cells: Dict[Tuple[str, str], Path] = {}
        for rd in members:
            case, struct = _parse_case_struct_from_exp_dir(_exp_dir_name_from_run_dir(rd))
            cells[(case, struct)] = rd

        cases = sorted({c for c, _ in cells.keys()}, key=lambda x: (len(x), x))
        structs = sorted({s for _, s in cells.keys()}, key=_struct_sort_key)
        if not cases or not structs:
            continue

        nrows, ncols = len(cases), len(structs)
        fig_w = min(32.0, 3.2 * ncols + 1.0)
        fig_h = min(24.0, 3.0 * nrows + 1.2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=140, squeeze=False)
        ax_grid: List[List[Any]] = [[axes[i, j] for j in range(ncols)] for i in range(nrows)]

        axis_lim: Optional[Tuple[float, float, float, float]] = None
        ref_bundle: Optional[Dict[str, Any]] = None
        for rd in members:
            b = _load_run_plot_bundle(rd, episode=episode)
            ref_bundle = b
            sc = b["scene"]
            axis_lim = (
                float(sc.get("n_min", 0.0)),
                float(sc.get("n_max", 250.0)),
                float(sc.get("e_min", 0.0)),
                float(sc.get("e_max", 250.0)),
            )
            break

        leg_fs = max(5.0, min(8.0, 11.0 - 0.45 * max(nrows, ncols)))

        use_numpy_heatmap = bool(mosaic_comm_heatmap or heatmap_comm)
        if mosaic_comm_heatmap:
            hm_metric = str(mosaic_comm_metric or "loss").strip().lower()
            if hm_metric not in {"loss", "quality"}:
                hm_metric = "loss"
        elif heatmap_comm:
            hm_metric = "quality"
        else:
            hm_metric = "loss"
        heatmap_engine = "numpy" if use_numpy_heatmap else "off"

        first_cb: Optional[Tuple[Any, str]] = None
        used_axes: List[Any] = []

        for i, case in enumerate(cases):
            for j, struct in enumerate(structs):
                ax = ax_grid[i][j]
                key = (case, struct)
                if key not in cells:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes)
                    continue
                rd = cells[key]
                bundle = _load_run_plot_bundle(rd, episode=episode)
                struct_short = _mosaic_variant_short_title(struct)
                title = f"{case}\n{struct_short}"
                use_full = bool(mosaic_full_labels)
                meta = _draw_bundle_on_ax(
                    ax,
                    fig,
                    bundle,
                    title=title,
                    show_obstacles=show_obstacles,
                    show_blackholes=show_blackholes,
                    heatmap_engine=heatmap_engine,
                    heatmap_res=heatmap_res,
                    heatmap_alpha=heatmap_alpha,
                    heatmap_numpy_metric=hm_metric,
                    overlay_comm=overlay_comm,
                    legend_fontsize=leg_fs,
                    axis_lim=axis_lim,
                    compact_legend=True,
                    show_subplot_legend=False,
                    defer_comm_cbar=(heatmap_engine == "numpy"),
                    show_title=use_full,
                    show_axis_labels=False,
                    show_dist_home_text=use_full,
                )
                used_axes.append(ax)
                if meta is not None and first_cb is None:
                    first_cb = meta

        if not mosaic_full_labels:
            for j, struct in enumerate(structs):
                struct_short = _mosaic_variant_short_title(struct)
                ax_top = ax_grid[0][j]
                ax_top.set_title(struct_short, fontsize=max(8.0, leg_fs + 1.0))
            for i, case in enumerate(cases):
                ax_left = ax_grid[i][0]
                ax_left.set_ylabel(case, fontsize=max(8.0, leg_fs + 1.0))
            for i in range(nrows):
                for j in range(ncols):
                    ax_grid[i][j].set_xlabel("")
            for i in range(nrows):
                for j in range(ncols):
                    ax = ax_grid[i][j]
                    key = (cases[i], structs[j])
                    if key not in cells:
                        continue
                    ax.tick_params(labelbottom=(i == nrows - 1), labelleft=(j == 0))
                    if j > 0:
                        ax.set_ylabel("")

        sf_label = ""
        if ref_bundle:
            sf_label = str(ref_bundle.get("scene_file", ""))
        fig.suptitle(f"Scene: {scene_stem}  ({sf_label})", fontsize=12, y=0.965)

        bottom_m = float(mosaic_subplot_bottom)
        bottom_m = min(max(bottom_m, 0.10), 0.40)
        leg_anchor = mosaic_legend_anchor_y
        if leg_anchor is None:
            leg_anchor = max(0.04, bottom_m - 0.05)
        else:
            leg_anchor = float(leg_anchor)
            leg_anchor = min(max(leg_anchor, 0.02), bottom_m - 0.012)

        left_m = 0.10 if not mosaic_full_labels else 0.07
        # Leave room for a vertical shared colorbar; otherwise it steals space from axes unevenly.
        right_m = 0.84 if (first_cb is not None and used_axes) else 0.90
        fig.subplots_adjust(left=left_m, right=right_m, bottom=bottom_m, top=0.88, wspace=0.20, hspace=0.30)
        if first_cb is not None and used_axes:
            im0, lbl0 = first_cb
            cbar = fig.colorbar(im0, ax=used_axes, fraction=0.042, pad=0.03)
            cbar.set_label(lbl0, fontsize=10)
            cbar.ax.tick_params(labelsize=9)

        _mosaic_bottom_legend(
            fig,
            show_obstacles=show_obstacles,
            show_blackholes=show_blackholes,
            has_numpy_comm_heatmap=use_numpy_heatmap,
            heatmap_metric=hm_metric,
            legend_anchor_y=leg_anchor,
        )
        _add_figure_north_arrow(fig)

        out_path = out_dir / f"mosaic_scene__{scene_stem}__ep{int(episode)}.png"
        # Do not use bbox_inches="tight" here: it recomputes the canvas and typically squashes the
        # manually tuned subplots_adjust + shared colorbar + bottom fig.legend layout.
        fig.savefig(out_path, dpi=160, pad_inches=0.08)
        plt.close(fig)
        written.append(out_path)

    return written


def render_trajectory_png(
    *,
    run_dir: Path,
    episode: int,
    out: Path,
    show_obstacles: bool,
    show_blackholes: bool,
    heatmap_comm: bool,
    heatmap_res: int,
    heatmap_alpha: float,
    overlay_comm: str,
    title_run_label: str,
) -> None:
    bundle = _load_run_plot_bundle(run_dir, episode=int(episode))
    fig, ax = plt.subplots(figsize=(6.8, 6.8), dpi=160)
    title = f"Trajectory + POI outcomes (ep={episode})\n{title_run_label}"
    h_eng = "loop" if heatmap_comm else "off"
    _draw_bundle_on_ax(
        ax,
        fig,
        bundle,
        title=title,
        show_obstacles=show_obstacles,
        show_blackholes=show_blackholes,
        heatmap_engine=h_eng,
        heatmap_res=heatmap_res,
        heatmap_alpha=heatmap_alpha,
        heatmap_numpy_metric="quality",
        overlay_comm=overlay_comm,
        legend_fontsize=9,
        axis_lim=None,
        compact_legend=False,
        show_subplot_legend=True,
        defer_comm_cbar=False,
        show_title=True,
        show_axis_labels=True,
        show_dist_home_text=True,
    )
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot UAV trajectory with POI outcomes and obstacles.")
    ap.add_argument("--run_dir", type=str, default="", help="A sweep leaf dir: .../Ts*/seed*/")
    ap.add_argument(
        "--sweep_root",
        type=str,
        default="",
        help="If set, plot every leaf under this dir that contains traj.jsonl (batch mode). Ignores --run_dir.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Batch mode: directory for PNGs (default: <sweep_root>/plot_trajectory).",
    )
    ap.add_argument(
        "--mosaic_by_scene",
        action="store_true",
        help="With --sweep_root: one large figure per scene_file (rows=case, cols=struct); "
        "writes mosaic_scene__<stem>__ep*.png under --out_dir instead of per-run PNGs.",
    )
    ap.add_argument(
        "--mosaic_comm_heatmap",
        action="store_true",
        help="With --mosaic_by_scene: draw comm field like uavlab.viz.plot_scenes (numpy, bilinear); "
        "one shared colorbar + bottom fig.legend. Default metric is loss (inferno).",
    )
    ap.add_argument(
        "--mosaic_comm_metric",
        type=str,
        default="loss",
        help="With --mosaic_comm_heatmap: 'loss' (inferno, degradation) or 'quality' (RdYlGn, 1−loss_p).",
    )
    ap.add_argument(
        "--mosaic_subplot_bottom",
        type=float,
        default=0.24,
        help="With --mosaic_by_scene: fig.subplots_adjust(bottom=...), fraction of figure height (0.10–0.40). "
        "Larger value reserves more space below axes for the shared bottom legend (default 0.24). "
        "Legend anchor auto-follows unless --mosaic_legend_anchor_y is set.",
    )
    ap.add_argument(
        "--mosaic_legend_anchor_y",
        type=float,
        default=None,
        help="With --mosaic_by_scene: fig.legend(..., bbox_to_anchor=(0.5, Y)) in figure coords (Y up = higher). "
        "Omit to auto-set just below mosaic_subplot_bottom. Raise Y to pull legend closer to subplots.",
    )
    ap.add_argument(
        "--mosaic_full_labels",
        action="store_true",
        help="With --mosaic_by_scene: repeat case+struct in every subplot title and dist_to_home on each panel "
        "(default is compact: column titles = struct, left column = case, tick labels on outer edges only).",
    )
    ap.add_argument(
        "--mosaic_publish",
        action="store_true",
        help="With --sweep_root: shorthand for publication-style scene mosaics — sets "
        "--mosaic_by_scene --show_obstacles --mosaic_comm_heatmap --mosaic_comm_metric quality --episode 0. "
        "Requires --sweep_root. For other metrics or no heatmap, omit this and pass the individual flags.",
    )
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out", type=str, default="", help="Output PNG path (default: <run_dir>/trajectory_ep*.png)")
    ap.add_argument("--show_obstacles", action="store_true")
    ap.add_argument("--show_blackholes", action="store_true", help="Overlay comm blackhole circles if present in scene.")
    ap.add_argument(
        "--heatmap_comm", action="store_true", help="Draw background heatmap of expected link quality (1-loss_p).",
    )
    ap.add_argument("--heatmap_res", type=int, default=140, help="Heatmap grid resolution per axis.")
    ap.add_argument("--heatmap_alpha", type=float, default=0.55, help="Heatmap alpha.")
    ap.add_argument(
        "--overlay_comm",
        type=str,
        default="",
        help="Overlay comm state on trajectory: 'blackhole' or 'loss' (colors by link_loss_p).",
    )
    args = ap.parse_args()

    argv = sys.argv[1:]
    if bool(getattr(args, "mosaic_publish", False)):
        if not str(args.sweep_root or "").strip():
            raise SystemExit("--mosaic_publish requires --sweep_root")
        args.mosaic_by_scene = True
        args.show_obstacles = True
        args.mosaic_comm_heatmap = True
        args.episode = 0
        if "--mosaic_comm_metric" not in argv:
            args.mosaic_comm_metric = "quality"

    sweep_root_s = str(args.sweep_root or "").strip()
    run_dir_s = str(args.run_dir or "").strip()

    if sweep_root_s:
        sweep_root = Path(sweep_root_s).resolve()
        if not sweep_root.is_dir():
            raise FileNotFoundError(f"--sweep_root is not a directory: {sweep_root}")
        out_dir = Path(args.out_dir).resolve() if str(args.out_dir or "").strip() else (sweep_root / "plot_trajectory")
        if str(args.out or "").strip():
            raise ValueError("In --sweep_root batch mode, do not pass --out; use --out_dir instead.")
        if run_dir_s:
            raise ValueError("Pass either --sweep_root (batch) or --run_dir (single), not both.")
        run_dirs = _discover_sweep_run_dirs(sweep_root)
        if not run_dirs:
            raise FileNotFoundError(f"No traj.jsonl found under: {sweep_root}")
        if bool(args.mosaic_by_scene):
            written = render_scene_mosaics_for_sweep(
                out_dir=out_dir,
                run_dirs=run_dirs,
                episode=int(args.episode),
                show_obstacles=bool(args.show_obstacles),
                show_blackholes=bool(args.show_blackholes),
                heatmap_comm=bool(args.heatmap_comm),
                heatmap_res=int(args.heatmap_res),
                heatmap_alpha=float(args.heatmap_alpha),
                overlay_comm=str(args.overlay_comm or ""),
                mosaic_comm_heatmap=bool(args.mosaic_comm_heatmap),
                mosaic_comm_metric=str(args.mosaic_comm_metric or "loss"),
                mosaic_subplot_bottom=float(args.mosaic_subplot_bottom),
                mosaic_legend_anchor_y=(None if args.mosaic_legend_anchor_y is None else float(args.mosaic_legend_anchor_y)),
                mosaic_full_labels=bool(args.mosaic_full_labels),
            )
            for p in written:
                print(str(p))
            return
        for run_dir in run_dirs:
            out_path = out_dir / _default_out_path_for_sweep_leaf(
                sweep_root=sweep_root, run_dir=run_dir, episode=int(args.episode)
            )
            title_lbl = run_dir.relative_to(sweep_root).as_posix()
            render_trajectory_png(
                run_dir=run_dir,
                episode=int(args.episode),
                out=out_path,
                show_obstacles=bool(args.show_obstacles),
                show_blackholes=bool(args.show_blackholes),
                heatmap_comm=bool(args.heatmap_comm),
                heatmap_res=int(args.heatmap_res),
                heatmap_alpha=float(args.heatmap_alpha),
                overlay_comm=str(args.overlay_comm or ""),
                title_run_label=title_lbl,
            )
            print(str(out_path))
        return

    if not run_dir_s:
        raise SystemExit("Provide --run_dir for a single plot, or --sweep_root to batch all traj.jsonl under a sweep.")

    run_dir = Path(run_dir_s).resolve()
    out = Path(args.out).resolve() if str(args.out or "").strip() else (run_dir / f"trajectory_ep{int(args.episode)}.png")
    render_trajectory_png(
        run_dir=run_dir,
        episode=int(args.episode),
        out=out,
        show_obstacles=bool(args.show_obstacles),
        show_blackholes=bool(args.show_blackholes),
        heatmap_comm=bool(args.heatmap_comm),
        heatmap_res=int(args.heatmap_res),
        heatmap_alpha=float(args.heatmap_alpha),
        overlay_comm=str(args.overlay_comm or ""),
        title_run_label=run_dir.name,
    )
    print(str(out))


if __name__ == "__main__":
    main()

