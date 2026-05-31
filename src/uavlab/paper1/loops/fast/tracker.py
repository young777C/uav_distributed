from __future__ import annotations

import math
from typing import Optional, Tuple

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.sim.env import Paper1Env

Point2D = Tuple[float, float]


def _dist(a: Point2D, b: Point2D) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def track_to_waypoint(
    *,
    env: Paper1Env,
    waypoint: Point2D,
    contract: Paper1ContractConfig,
    dt: float,
) -> Tuple[Point2D, Optional[Point2D]]:
    """
    Returns ``(target_ne, vel_ne_cmd)``. If PID selected, ``vel_ne_cmd`` is set (clipped to
    ``v_xy_max`` for urgent tracking); else ``None`` and ``step_fast`` uses nominal pursuit
    (average ``v_xy_cruise``, approach slowdown, hover in POI cover).
    """

    ft = dict(contract.fast_tracker or {})
    kind = str(ft.get("type", "pure_pursuit")).strip().lower()
    px, py = float(env.pos_ne[0]), float(env.pos_ne[1])
    wx, wy = float(waypoint[0]), float(waypoint[1])
    vmax = float(env.cfg.v_xy_max)

    if kind == "pid":
        pid = dict(ft.get("pid_gains") or {})
        kp = float(pid.get("kp", 1.0))
        ki = float(pid.get("ki", 0.0))
        kd = float(pid.get("kd", 0.0))
        ex, ey = wx - px, wy - py
        vx = kp * ex - kd * float(env.vel_ne[0])
        vy = kp * ey - kd * float(env.vel_ne[1])
        _ = ki  # reserved
        spd = float(math.hypot(vx, vy))
        if spd > vmax and spd > 1e-9:
            s = vmax / spd
            vx, vy = vx * s, vy * s
        return (waypoint, (vx, vy))

    # pure pursuit: lookahead along pos -> waypoint (not capped at 8 m; scales with cruise speed).
    vc = float(env.cfg.v_xy_cruise)
    d = float(_dist(env.pos_ne, waypoint))
    if d < 1e-6:
        return (waypoint, None)
    steps = float(ft.get("lookahead_steps", 25.0))
    ld_cap = float(ft.get("lookahead_max_m", 0.0))
    ld = max(2.0, vc * float(max(dt, 1e-9)) * max(1.0, steps))
    if ld_cap > 0.0:
        ld = min(ld, ld_cap)
    ld = min(d, ld)
    ux, uy = (wx - px) / d, (wy - py) / d
    target = (px + ux * ld, py + uy * ld)
    return (target, None)
