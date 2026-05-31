"""Deterministic link loss at a map position (planning / masks, no jitter)."""

from __future__ import annotations

import math

from uavlab.paper1.sim.env import Paper1Env

Point2D = tuple[float, float]


def quality_from_loss(loss_p: float) -> float:
    return float(min(1.0, max(0.0, 1.0 - float(loss_p))))


def loss_proxy_at(env: Paper1Env, p: Point2D) -> float:
    """Deterministic loss at ``p`` (distance decay, base loss, blackholes); no jitter."""

    cfg = env.cfg
    gx, gy = float(cfg.gcs_ne[0]), float(cfg.gcs_ne[1])
    x, y = float(p[0]), float(p[1])
    d = float(math.hypot(x - gx, y - gy))

    if bool(getattr(cfg, "enable_distance_decay", False)):
        d0 = float(getattr(cfg, "distance_d0_m"))
        d1 = float(getattr(cfg, "distance_d1_m"))
        lmin = float(getattr(cfg, "distance_loss_min"))
        lmax = float(getattr(cfg, "distance_loss_max"))
        if d <= d0:
            loss = lmin
        elif d >= d1:
            loss = lmax
        else:
            t = (d - d0) / max(1e-9, (d1 - d0))
            loss = lmin + (lmax - lmin) * float(t)
    else:
        loss = float(getattr(cfg, "base_loss"))

    if bool(getattr(cfg, "use_scene_blackholes", False)) and bool(env.in_blackhole((x, y))):
        loss = float(loss) + float(getattr(cfg, "blackhole_extra_loss"))

    return float(min(max(loss, 0.0), 0.99))
