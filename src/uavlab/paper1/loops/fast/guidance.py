from __future__ import annotations

import math
from typing import List, Tuple

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.comm.link_proxy import loss_proxy_at, quality_from_loss
from uavlab.paper1.sim.env import Paper1Env
from uavlab.scene.geometry import nearest_nofly_dist_dir

Point2D = Tuple[float, float]


def _dist(a: Point2D, b: Point2D) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _terminal_homing_radius_m(*, env: Paper1Env, fg: dict, candidate_radius_m: float) -> float:
    """Near-goal zone: pursue POI center directly (see ``terminal_homing_radius_m`` in config)."""

    raw = fg.get("terminal_homing_radius_m")
    if raw is not None:
        v = float(raw)
        if v > 0.0:
            return v
    visit_r = float(env.cfg.visit_radius_m)
    return float(max(2.0 * visit_r, 1.5 * float(candidate_radius_m)))


def _ring_candidates(*, gx: float, gy: float, rad: float, n_c: int) -> List[Point2D]:
    cand: List[Point2D] = [(gx, gy)]
    if rad <= 1e-9 or n_c <= 1:
        return cand
    for k in range(max(0, n_c - 1)):
        ang = 2.0 * math.pi * float(k) / float(max(1, n_c - 1))
        cand.append((gx + rad * math.cos(ang), gy + rad * math.sin(ang)))
    return cand


def _best_feasible_on_ring(
    *,
    env: Paper1Env,
    goal_ne: Point2D,
    rad: float,
    n_c: int,
    enable_obstacle_filter: bool,
) -> Point2D:
    gx, gy = float(goal_ne[0]), float(goal_ne[1])
    best = (gx, gy)
    best_d = 1e30
    for q in _ring_candidates(gx=gx, gy=gy, rad=rad, n_c=n_c):
        if enable_obstacle_filter and env.in_nofly(q):
            continue
        d = _dist(q, (gx, gy))
        if d < best_d:
            best_d = d
            best = q
    return best


def pick_waypoint(
    *,
    env: Paper1Env,
    goal_ne: Point2D,
    contract: Paper1ContractConfig,
) -> Point2D:
    """
    Local reference toward ``goal_ne`` with optional micro-offset candidates.

    - **Terminal homing** (idea 1): inside ``terminal_homing_radius_m``, steer to the POI center
      when feasible (nofly-safe), else the closest feasible point on the candidate ring.
    - **Approach cost** (idea 2): prefer candidates *closer to the goal*, not closer to the
      current position (avoids stalling on a fixed-radius ring outside the cover disk).
    """

    gx, gy = float(goal_ne[0]), float(goal_ne[1])
    px, py = float(env.pos_ne[0]), float(env.pos_ne[1])
    fg = dict(contract.fast_guidance or {})
    if not bool(fg.get("enable_obstacle_filter", True)) and not bool(fg.get("enable_smooth_cost", True)):
        return (gx, gy)

    n_c = int(max(1, fg.get("n_candidates", 8)))
    rad = float(fg.get("candidate_radius_m", 6.0))
    w_d = float(fg.get("w_dist", 1.0))
    w_l = float(fg.get("w_link", 2.0))
    w_s = float(fg.get("w_smooth", 0.25))
    w_goal = float(fg.get("w_goal", 0.5))
    link_mode = str(fg.get("link_cost_mode", "current")).strip().lower()
    enable_obstacle = bool(fg.get("enable_obstacle_filter", True))
    enable_smooth = bool(fg.get("enable_smooth_cost", True))

    d_to_goal = _dist((px, py), (gx, gy))
    terminal_r = _terminal_homing_radius_m(env=env, fg=fg, candidate_radius_m=rad)

    if d_to_goal <= terminal_r:
        if not (enable_obstacle and env.in_nofly((gx, gy))):
            return (gx, gy)
        return _best_feasible_on_ring(
            env=env,
            goal_ne=goal_ne,
            rad=rad,
            n_c=n_c,
            enable_obstacle_filter=enable_obstacle,
        )

    link = env.observe_link_state()
    loss_here = float(link.loss_p)

    def loss_proxy_at(q: Point2D) -> float:
        if link_mode != "candidate_proxy":
            return loss_here
        x, y = float(q[0]), float(q[1])
        gx2, gy2 = float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1])
        d = float(math.hypot(x - gx2, y - gy2))
        if bool(env.cfg.enable_distance_decay):
            d0, d1 = float(env.cfg.distance_d0_m), float(env.cfg.distance_d1_m)
            lmin, lmax = float(env.cfg.distance_loss_min), float(env.cfg.distance_loss_max)
            if d <= d0:
                loss = lmin
            elif d >= d1:
                loss = lmax
            else:
                t = (d - d0) / max(1e-9, (d1 - d0))
                loss = lmin + (lmax - lmin) * float(t)
        else:
            loss = float(env.cfg.base_loss)
        if env.in_blackhole((x, y)):
            loss = float(loss) + float(env.cfg.blackhole_extra_loss)
        return float(min(max(loss, 0.0), 0.99))

    cand = _ring_candidates(gx=gx, gy=gy, rad=rad, n_c=n_c)
    prev = env.pos_ne
    best = (gx, gy)
    best_c = 1e30
    for q in cand:
        if enable_obstacle and env.in_nofly(q):
            continue
        d_approach = _dist(q, (gx, gy))
        lp = loss_proxy_at(q)
        sm = _dist(q, prev) if enable_smooth else 0.0
        scale = 1.0 + d_to_goal / max(terminal_r, 1e-6)
        c = w_d * d_approach + w_goal * d_approach * scale + w_l * lp + w_s * sm
        if c < best_c:
            best_c = c
            best = q
    return best


def pick_nofly_escape_target(
    *,
    env: Paper1Env,
    contract: Paper1ContractConfig,
    mission_goal_ne: Point2D,
) -> Point2D:
    """
    ``S_safe`` macro-target: step outward from the nearest no-fly boundary.

    Not GCS homing — return flight is reserved for ``S_back`` (energy critical).
    """

    px, py = float(env.pos_ne[0]), float(env.pos_ne[1])
    signed, dir_n, dir_e = nearest_nofly_dist_dir(px, py, env.scene_geom)
    if float(signed) >= 0.0:
        return (float(mission_goal_ne[0]), float(mission_goal_ne[1]))

    # Inside no-fly: ``nearest_nofly_dist_dir`` points from boundary toward the UAV (inward).
    # Escape uses the opposite direction (outward across the nearest boundary).
    dir_n, dir_e = -float(dir_n), -float(dir_e)

    fg = dict(contract.fast_guidance or {})
    base_step = float(fg.get("nofly_escape_step_m", 0.0))
    if base_step <= 0.0:
        rad = float(fg.get("candidate_radius_m", 6.0))
        base_step = float(max(12.0, 2.5 * rad))
    max_step = float(fg.get("nofly_escape_max_step_m", base_step * 4.0))

    if abs(dir_n) < 1e-9 and abs(dir_e) < 1e-9:
        mx, my = float(mission_goal_ne[0]), float(mission_goal_ne[1])
        dx, dy = mx - px, my - py
        norm = float(math.hypot(dx, dy))
        if norm > 1e-6:
            dir_n, dir_e = dx / norm, dy / norm
        else:
            dir_n, dir_e = 1.0, 0.0

    step = base_step
    tx, ty = px + dir_n * step, py + dir_e * step
    while step <= max_step + 1e-9 and env.in_nofly((tx, ty)):
        step += base_step
        tx, ty = px + dir_n * step, py + dir_e * step
    return (float(tx), float(ty))


def pick_link_recovery_waypoint(
    *,
    env: Paper1Env,
    contract: Paper1ContractConfig,
    params: FastLoopParams,
    fallback_ne: Point2D,
) -> Point2D:
    """
    Mobile ``Srec``: steer toward the nearest feasible point with strong link quality.

    Candidates: ring around the UAV and samples toward GCS. Prefer the closest point
    whose path quality ``q = 1 - loss`` meets the FSM recover threshold; if none qualify,
    pick the highest-q feasible point (tie-break by distance).
    """

    px, py = float(env.pos_ne[0]), float(env.pos_ne[1])
    gx, gy = float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1])
    fg = dict(contract.fast_guidance or {})
    enable_obstacle = bool(fg.get("enable_obstacle_filter", True))
    rad = float(max(fg.get("candidate_radius_m", 6.0) * 2.0, 12.0))
    n_c = int(max(8, fg.get("n_candidates", 8)))
    q_min = quality_from_loss(float(params.link_loss_recover))

    cand: List[Point2D] = [(px, py)]
    cand.extend(_ring_candidates(gx=px, gy=py, rad=rad, n_c=n_c))
    for k in range(1, 9):
        t = float(k) / 8.0
        cand.append((px + (gx - px) * t, py + (gy - py) * t))

    best_any: Point2D = fallback_ne
    best_any_q = -1.0
    best_any_d = 1e30
    best_qual: Point2D = fallback_ne
    best_qual_d = 1e30

    for q in cand:
        if enable_obstacle and env.in_nofly(q):
            continue
        qual = quality_from_loss(loss_proxy_at(env, q))
        d = _dist(q, (px, py))
        if qual > best_any_q or (abs(qual - best_any_q) < 1e-9 and d < best_any_d):
            best_any_q = qual
            best_any_d = d
            best_any = q
        if qual + 1e-9 >= q_min and d < best_qual_d:
            best_qual_d = d
            best_qual = q

    if best_qual_d < 1e29:
        return best_qual
    if best_any_q >= 0.0:
        return best_any
    return fallback_ne
