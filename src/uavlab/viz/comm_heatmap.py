"""
Deterministic mean loss_p field (no comm jitter) for background heatmaps.

Used by ``uavlab.viz.plot_scenes`` and ``uavlab.viz.plot_trajectory`` so the
grid math stays in one place (aligned with Paper1Lite comm knobs).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def gcs_ne_from_scene(scene: Dict[str, Any]) -> Tuple[float, float]:
    ne = scene.get("gcs_ne", scene.get("start_ne", (0.0, 0.0)))
    return float(ne[0]), float(ne[1])


def blackholes_for_comm(scene: Dict[str, Any], comm: Dict[str, Any]) -> List[List[float]]:
    if not bool(comm.get("use_scene_blackholes", False)):
        return []
    return [list(map(float, c)) for c in list(scene.get("communication_blackholes") or [])]


def expected_loss_grid(
    scene: Dict[str, Any],
    comm: Dict[str, Any],
    *,
    grid_n: int = 200,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Return ``(loss, (n_min, n_max, e_min, e_max))`` with ``loss`` shape ``(grid_n, grid_n)``."""
    n_min = float(scene.get("n_min", 0.0))
    n_max = float(scene.get("n_max", 250.0))
    e_min = float(scene.get("e_min", 0.0))
    e_max = float(scene.get("e_max", 250.0))
    gx, gy = gcs_ne_from_scene(scene)
    base_loss = float(comm.get("base_loss", 0.15))
    enable_decay = bool(comm.get("enable_distance_decay", False))
    d0 = float(comm.get("distance_d0_m", 0.0))
    d1 = float(max(float(comm.get("distance_d1_m", 250.0)), d0 + 1e-6))
    lmin = float(comm.get("distance_loss_min", 0.05))
    lmax = float(comm.get("distance_loss_max", 0.60))
    bh_extra = float(comm.get("blackhole_extra_loss", 0.5))
    blackholes = blackholes_for_comm(scene, comm)

    xs = np.linspace(n_min, n_max, grid_n, dtype=np.float64)
    ys = np.linspace(e_min, e_max, grid_n, dtype=np.float64)
    x, y = np.meshgrid(xs, ys, indexing="xy")

    if enable_decay:
        dist = np.hypot(x - gx, y - gy)
        t = (dist - d0) / (d1 - d0)
        t = np.clip(t, 0.0, 1.0)
        loss = (1.0 - t) * lmin + t * lmax
        loss = np.clip(loss, 0.0, 1.0)
    else:
        loss = np.full_like(x, base_loss, dtype=np.float64)

    for cx, cy, r in blackholes:
        inside = (x - cx) ** 2 + (y - cy) ** 2 < r**2
        loss = np.where(inside, np.minimum(1.0, loss + bh_extra), loss)

    bounds = (n_min, n_max, e_min, e_max)
    return loss, bounds


def imshow_comm_field(
    ax: Any,
    scene: Dict[str, Any],
    comm: Dict[str, Any],
    *,
    grid_n: int = 200,
    alpha: float = 0.55,
    metric: str = "quality",
) -> Tuple[Any, str]:
    """
    Draw bilinear heatmap on ``ax``. ``metric`` is ``quality`` (1−loss, RdYlGn) or ``loss`` (inferno).
    Sets x/y limits to scene bounds. Returns ``(mappable, colorbar_label)``.
    """
    loss, (n_min, n_max, e_min, e_max) = expected_loss_grid(scene, comm, grid_n=grid_n)
    m = str(metric or "quality").strip().lower()
    if m == "loss":
        z = loss
        cmap, vmin, vmax = "inferno", 0.0, 1.0
        label = "Comm degradation (loss_p), deterministic"
    else:
        z = 1.0 - loss
        cmap, vmin, vmax = "RdYlGn", 0.0, 1.0
        label = "Expected link quality (1 − loss_p), deterministic"

    im = ax.imshow(
        z,
        extent=(n_min, n_max, e_min, e_max),
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=float(alpha),
        interpolation="bilinear",
        zorder=0,
    )
    ax.set_xlim(n_min, n_max)
    ax.set_ylim(e_min, e_max)
    return im, label
