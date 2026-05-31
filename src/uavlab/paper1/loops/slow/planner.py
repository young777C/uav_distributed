"""
Paper1 GCS slow-loop **planner** primitives (candidate window + masks + greedy picks).

Implements the §3.4.5-style pipeline up to the MILP boundary:

1. Prefetch nearest POIs among uncovered.
2. Hard masks (control-path + energy + optional data-return for CDSL).
3. Soft scores (comm quality + data-return suitability R_ret).
4. Optional degrade ladder when the feasible set is empty (struct axis).

``solve_short_horizon_tour`` lives in ``optimizer_ortools.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from uavlab.paper1.comm.link_proxy import loss_proxy_at, quality_from_loss
from uavlab.paper1.comm.dual_link import Paper1DualLinkThresholds
from uavlab.paper1.comm.thresholds import (
    control_link_ok,
    data_path_feasible_at,
    link_proxy_at,
)
from uavlab.paper1.contracts.struct_profile import StructAxisProfile
from uavlab.paper1.sim.energy_model import energy_need_from_env
from uavlab.paper1.sim.env import Paper1Env

Point2D = Tuple[float, float]

# Loss-exposure path thresholds (清单 §2.2).
_LOSS_EXPOSURE_LOSS_P = 0.50
_LOSS_EXPOSURE_DELAY_S = 5.0


@dataclass(frozen=True)
class SlowLoopParams:
    """Weights / thresholds for MILP objective and per-POI masks."""

    lambda_q: float = 1.0
    mu_dist: float = 0.01
    beta_ret: float = 1.0
    t_obs_s: float = 10.0
    t_safe_s: float = 0.0
    dual_link: Paper1DualLinkThresholds = Paper1DualLinkThresholds()
    energy_plan_margin_frac: float = 0.0


@dataclass(frozen=True)
class SlowWindow:
    """
    Candidate window ``W_t`` plus precomputed matrices for CP-SAT.

    Node 0 = current UAV position; nodes 1..n = ``poi_ids`` in order.
    """

    poi_ids: List[int]
    poi_pos: List[Point2D]
    d_ij: List[List[float]]
    e_fly_ij: List[List[float]]
    e_hover_i: List[float]
    q_poi_i: List[float]
    q_path_min_i: List[float]
    r_ret_i: List[float]
    a_cmd_i: List[float]
    r_loss_i: List[float]
    m_energy_i: List[bool]
    m_quality_i: List[bool]
    m_i: List[bool]
    degrade_level: int = 0


def _dist(a: Point2D, b: Point2D) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _task_done_poi_ids(env: Paper1Env) -> set[int]:
    return set(env.effective)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, float(x))))


def _path_samples_along_segment(a: Point2D, b: Point2D, samples: int) -> List[Point2D]:
    n = int(max(2, samples))
    pts: List[Point2D] = []
    for k in range(n):
        t = float(k) / float(n - 1)
        pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return pts


def _path_control_stats(
    *,
    env: Paper1Env,
    poi_ne: Point2D,
    samples: int,
    th: Paper1DualLinkThresholds,
) -> tuple[float, float]:
    """Return (A_i^cmd, R_i^loss) along UAV→POI segment."""

    a = (float(env.pos_ne[0]), float(env.pos_ne[1]))
    pts = _path_samples_along_segment(a, poi_ne, samples)
    if not pts:
        return 1.0, 0.0
    ok_n = 0
    loss_n = 0
    for p in pts:
        link = link_proxy_at(env, p, jitter=False)
        if control_link_ok(link, th):
            ok_n += 1
        if (
            float(link.loss_p) > _LOSS_EXPOSURE_LOSS_P
            or float(link.delay_s) > _LOSS_EXPOSURE_DELAY_S
        ):
            loss_n += 1
    inv = 1.0 / float(len(pts))
    return float(ok_n) * inv, float(loss_n) * inv


def _q_path_min_sampled(*, env: Paper1Env, poi_ne: Point2D, samples: int) -> float:
    a = (float(env.pos_ne[0]), float(env.pos_ne[1]))
    qmin = 1.0
    for p in _path_samples_along_segment(a, poi_ne, samples):
        qmin = min(qmin, quality_from_loss(loss_proxy_at(env, p)))
    return float(qmin)


def _data_return_score(
    *,
    env: Paper1Env,
    poi_ne: Point2D,
    key_bits: float,
    th: Paper1DualLinkThresholds,
) -> float:
    """Continuous R_i^ret in [0, 1] (清单 §2.4)."""

    link = link_proxy_at(env, poi_ne, jitter=False)
    rho = float(link.loss_p)
    rho_max = float(th.data_max_loss_p)
    t_tx = float(link.delay_s) + float(key_bits) / max(float(link.bandwidth_bps), 1.0)
    t_max = float(th.data_max_return_time_s)
    j = float(getattr(link, "jitter_s", 0.0))
    j_max = float(th.data_max_jitter_s)

    w_rho, w_t, w_j = 0.4, 0.4, 0.2
    s_rho = _clip01(1.0 - rho / max(rho_max, 1e-9))
    s_t = _clip01(1.0 - t_tx / max(t_max, 1e-9))
    s_j = _clip01(1.0 - j / max(j_max, 1e-9)) if j_max > 0.0 else 1.0
    return float(w_rho * s_rho + w_t * s_t + w_j * s_j)


def _energy_need_to_complete_and_return(
    *,
    env: Paper1Env,
    poi_ne: Point2D,
    params: SlowLoopParams,
) -> float:
    t_hov = float(params.t_obs_s + params.t_safe_s)
    if t_hov <= 0.0:
        t_hov = float(env.cfg.poi_dwell_s)
    need = energy_need_from_env(env, poi_ne=poi_ne, t_hov_s=t_hov)
    plan_margin = float(params.energy_plan_margin_frac)
    if plan_margin > 0.0:
        need += plan_margin
    return float(need)


def _empty_slow_window(*, degrade_level: int = 0) -> SlowWindow:
    return SlowWindow(
        poi_ids=[],
        poi_pos=[],
        d_ij=[[0.0]],
        e_fly_ij=[[0.0]],
        e_hover_i=[0.0],
        q_poi_i=[0.0],
        q_path_min_i=[1.0],
        r_ret_i=[0.0],
        a_cmd_i=[1.0],
        r_loss_i=[0.0],
        m_energy_i=[True],
        m_quality_i=[True],
        m_i=[True],
        degrade_level=int(degrade_level),
    )


def _slow_window_for_poi_sequence(
    *,
    env: Paper1Env,
    params: SlowLoopParams,
    poi_ids: List[int],
    profile: StructAxisProfile,
    comm_objective: bool,
    legacy_comm_path_quality_mask: bool,
    apply_energy_per_node_mask: bool,
    path_samples: int,
    feasible_nodes_only: bool,
    degrade_level: int,
) -> SlowWindow:
    if not poi_ids:
        return _empty_slow_window(degrade_level=degrade_level)

    prof = profile.relaxed_for_level(degrade_level) if profile.use_struct_comm_profile else profile
    poi_pos = [(float(env.pois[int(pid)].pos_ne[0]), float(env.pois[int(pid)].pos_ne[1])) for pid in poi_ids]
    nodes: List[Point2D] = [(float(env.pos_ne[0]), float(env.pos_ne[1]))] + list(poi_pos)
    n = len(poi_ids)
    th = params.dual_link
    key_bits = float(env.cfg.key_bits_per_poi)

    d_ij: List[List[float]] = [[0.0 for _ in range(1 + n)] for _ in range(1 + n)]
    e_fly_ij: List[List[float]] = [[0.0 for _ in range(1 + n)] for _ in range(1 + n)]
    for i in range(1 + n):
        for j in range(1 + n):
            if i == j:
                continue
            d = _dist(nodes[i], nodes[j])
            d_ij[i][j] = float(d)
            e_fly_ij[i][j] = float(env.cfg.energy_per_meter) * float(d)

    e_hover_i: List[float] = [0.0 for _ in range(1 + n)]
    for i in range(1, 1 + n):
        e_hover_i[i] = float(env.cfg.energy_hover_per_s) * float(params.t_obs_s + params.t_safe_s)

    q_poi_i: List[float] = [0.0 for _ in range(1 + n)]
    q_path_min_i: List[float] = [1.0 for _ in range(1 + n)]
    r_ret_i: List[float] = [0.0 for _ in range(1 + n)]
    a_cmd_i: List[float] = [1.0 for _ in range(1 + n)]
    r_loss_i: List[float] = [0.0 for _ in range(1 + n)]

    use_struct = bool(prof.use_struct_comm_profile)
    for i in range(1, 1 + n):
        r_ret_i[i] = _data_return_score(env=env, poi_ne=nodes[i], key_bits=key_bits, th=th)
        if use_struct:
            a_cmd_i[i], r_loss_i[i] = _path_control_stats(
                env=env, poi_ne=nodes[i], samples=int(path_samples), th=th
            )
            q_path_min_i[i] = float(a_cmd_i[i])
        else:
            q_path_min_i[i] = _q_path_min_sampled(env=env, poi_ne=nodes[i], samples=int(path_samples))

    if bool(comm_objective):
        for i in range(1, 1 + n):
            q_poi_i[i] = quality_from_loss(loss_proxy_at(env, nodes[i]))

    if bool(feasible_nodes_only):
        m_energy_i = [True for _ in range(1 + n)]
        m_quality_i = [True for _ in range(1 + n)]
        m_i = [True for _ in range(1 + n)]
    else:
        m_energy_i = [True for _ in range(1 + n)]
        if bool(apply_energy_per_node_mask):
            for i in range(1, 1 + n):
                need = _energy_need_to_complete_and_return(env=env, poi_ne=nodes[i], params=params)
                m_energy_i[i] = bool(float(env.remaining_energy) >= float(need))

        m_quality_i = [True for _ in range(1 + n)]
        if use_struct and bool(prof.use_control_path_hard_mask):
            for i in range(1, 1 + n):
                ok_ctrl = bool(
                    float(a_cmd_i[i]) >= float(prof.control_cmd_min_ratio)
                    and float(r_loss_i[i]) <= float(prof.control_loss_exposure_max)
                )
                m_quality_i[i] = ok_ctrl
                if bool(prof.use_data_return_hard_mask):
                    m_quality_i[i] = bool(
                        m_quality_i[i]
                        and data_path_feasible_at(env, nodes[i], key_bits, th)
                    )
        elif bool(legacy_comm_path_quality_mask):
            for i in range(1, 1 + n):
                ok_path = bool(float(q_path_min_i[i]) >= float(th.data_min_path_quality))
                ok_dest = data_path_feasible_at(env, nodes[i], key_bits, th)
                m_quality_i[i] = bool(ok_path and ok_dest)

        m_i = [bool(m_energy_i[i] and m_quality_i[i]) for i in range(1 + n)]

    return SlowWindow(
        poi_ids=list(poi_ids),
        poi_pos=poi_pos,
        d_ij=d_ij,
        e_fly_ij=e_fly_ij,
        e_hover_i=e_hover_i,
        q_poi_i=q_poi_i,
        q_path_min_i=q_path_min_i,
        r_ret_i=r_ret_i,
        a_cmd_i=a_cmd_i,
        r_loss_i=r_loss_i,
        m_energy_i=m_energy_i,
        m_quality_i=m_quality_i,
        m_i=m_i,
        degrade_level=int(degrade_level),
    )


def build_candidate_window(
    *,
    env: Paper1Env,
    params: SlowLoopParams,
    profile: StructAxisProfile,
    window_k: int,
    prefetch_k_multiplier: int,
    comm_objective: bool,
    comm_path_quality_mask: bool,
    apply_energy_per_node_mask: bool,
    path_samples: int,
    degrade_level: int = 0,
) -> SlowWindow:
    done = _task_done_poi_ids(env)

    cand: List[tuple[float, int]] = []
    for poi in env.pois:
        pid = int(poi.poi_id)
        if pid in done:
            continue
        cand.append((_dist(env.pos_ne, poi.pos_ne), pid))
    cand.sort(key=lambda x: x[0])

    k_base = int(max(0, window_k))
    mult = int(max(1, prefetch_k_multiplier))
    prefetch_n = min(mult * k_base, len(cand)) if k_base > 0 else 0
    prefetch_ids = [pid for _, pid in cand[:prefetch_n]]

    if not prefetch_ids:
        return _empty_slow_window(degrade_level=degrade_level)

    pre = _slow_window_for_poi_sequence(
        env=env,
        params=params,
        poi_ids=prefetch_ids,
        profile=profile,
        comm_objective=bool(comm_objective),
        legacy_comm_path_quality_mask=bool(comm_path_quality_mask),
        apply_energy_per_node_mask=bool(apply_energy_per_node_mask),
        path_samples=int(path_samples),
        feasible_nodes_only=False,
        degrade_level=int(degrade_level),
    )
    feasible_ids = [pre.poi_ids[i] for i in range(len(pre.poi_ids)) if pre.m_i[i + 1]]
    if not feasible_ids:
        return _empty_slow_window(degrade_level=degrade_level)

    return _slow_window_for_poi_sequence(
        env=env,
        params=params,
        poi_ids=feasible_ids,
        profile=profile,
        comm_objective=bool(comm_objective),
        legacy_comm_path_quality_mask=bool(comm_path_quality_mask),
        apply_energy_per_node_mask=bool(apply_energy_per_node_mask),
        path_samples=int(path_samples),
        feasible_nodes_only=True,
        degrade_level=int(degrade_level),
    )


def build_candidate_window_with_degrade(
    *,
    env: Paper1Env,
    params: SlowLoopParams,
    profile: StructAxisProfile,
    window_k: int,
    prefetch_k_multiplier: int,
    comm_objective: bool,
    comm_path_quality_mask: bool,
    apply_energy_per_node_mask: bool,
    path_samples: int,
) -> SlowWindow:
    """Try degrade levels 0..max until the candidate window is non-empty."""

    max_lv = int(max(0, profile.candidate_degrade_max_level))
    if not profile.use_struct_comm_profile:
        max_lv = 0
    for lv in range(0, max_lv + 1):
        win = build_candidate_window(
            env=env,
            params=params,
            profile=profile,
            window_k=window_k,
            prefetch_k_multiplier=prefetch_k_multiplier,
            comm_objective=comm_objective,
            comm_path_quality_mask=comm_path_quality_mask,
            apply_energy_per_node_mask=apply_energy_per_node_mask,
            path_samples=path_samples,
            degrade_level=lv,
        )
        if win.poi_ids:
            return win
    return _empty_slow_window(degrade_level=max_lv)


def nearest_energy_feasible_poi(
    *,
    env: Paper1Env,
    params: SlowLoopParams,
) -> Optional[int]:
    """Level-3 fallback: nearest uncovered POI with physical energy feasibility only."""

    best_id: Optional[int] = None
    best_d = 1e30
    done = _task_done_poi_ids(env)
    phys = SlowLoopParams(
        lambda_q=params.lambda_q,
        mu_dist=params.mu_dist,
        beta_ret=params.beta_ret,
        t_obs_s=params.t_obs_s,
        t_safe_s=params.t_safe_s,
        dual_link=params.dual_link,
        energy_plan_margin_frac=0.0,
    )
    for poi in env.pois:
        pid = int(poi.poi_id)
        if pid in done:
            continue
        need = _energy_need_to_complete_and_return(env=env, poi_ne=poi.pos_ne, params=phys)
        if float(env.remaining_energy) < float(need):
            continue
        d = _dist(env.pos_ne, poi.pos_ne)
        if d < best_d:
            best_d = d
            best_id = pid
    return best_id


def pick_next_nearest_in_comm_quality_window(
    *,
    env: Paper1Env,
    params: SlowLoopParams,
    profile: StructAxisProfile,
    window_k: int,
    prefetch_k_multiplier: int,
    comm_objective: bool,
    comm_path_quality_mask: bool,
    apply_energy_per_node_mask: bool,
    path_samples: int,
) -> Optional[int]:
    win = build_candidate_window_with_degrade(
        env=env,
        params=params,
        profile=profile,
        window_k=int(window_k),
        prefetch_k_multiplier=int(prefetch_k_multiplier),
        comm_objective=bool(comm_objective),
        comm_path_quality_mask=bool(comm_path_quality_mask),
        apply_energy_per_node_mask=bool(apply_energy_per_node_mask),
        path_samples=int(path_samples),
    )
    if not win.poi_ids:
        return nearest_energy_feasible_poi(env=env, params=params)
    best_id: Optional[int] = None
    best_d = 1e30
    pos = env.pos_ne
    for i, pid in enumerate(win.poi_ids):
        d = _dist(pos, win.poi_pos[i])
        if d < best_d:
            best_d = d
            best_id = int(pid)
    return best_id


def pick_next_poi_open_loop_sequence(
    *,
    env: Paper1Env,
    params: SlowLoopParams,
) -> Optional[int]:
    return nearest_energy_feasible_poi(env=env, params=params)
