"""
Paper 2 §4.1 / §5.2–5.3: Effective completion probability model.

Implements:
- Eq. (2):  T̂^tx(C) = τ + B^key / (b(1-ℓ) + ε) ... estimated return time
- Eq. (15): P̂^eff_{i,G}(t) = clip(P̂^cov_{i,G}(t) · P̂^ret|cov_{i,G}(t) · η^calib, 0, 1)
- Eq. (16): P̂^cov_{i,G}(t) coverage success probability
- Eq. (17): P̂^ret|cov_{i,G}(t)  return conditional probability
- Eq. (18): q̂^ret_i(t) return link suitability
- Eq. (22): P̂^eff_{g,U}(a, t) UAV action-conditional probability
- Eq. (23): P̂^cov_{g,U}(a, t) UAV coverage success
- Eq. (24): P̂^ret|cov_{g,U}(a, t) UAV return conditional
- Eq. (9):  D_pe(t) = |P̂^eff_{g,G}(t) - P̂^eff_{g,U}(t)| prediction-execution discrepancy

All functions are PURE (no side effects, no state mutation).
"""

from __future__ import annotations

import math
from typing import Optional

# ——— constants —————————————————————————————————————————————
_EPS = 1e-12


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, float(x))))


def _geo_mean_4(a: float, b: float, c: float, d: float) -> float:
    """Geometric mean of four clipped values."""
    return float((_clip01(a) * _clip01(b) * _clip01(c) * _clip01(d)) ** 0.25)


def _soft_geo_mean_4(
    a: float, b: float, c: float, d: float, *, alpha: float = 0.25
) -> float:
    """Soft geometric mean: (1-α) · geo_mean + α · arith_mean.

    Preserves multi-factor sensitivity (via geo-mean) while preventing
    any single factor from being a hard veto (via arith-mean component).
    ``alpha=0`` → pure geometric mean, ``alpha=1`` → pure arithmetic mean.
    """
    ca, cb, cc, cd = _clip01(a), _clip01(b), _clip01(c), _clip01(d)
    gm = float((ca * cb * cc * cd) ** 0.25)
    am = float((ca + cb + cc + cd) / 4.0)
    return float((1.0 - float(alpha)) * gm + float(alpha) * am)


def _geo_mean_3(a: float, b: float, c: float) -> float:
    return float((_clip01(a) * _clip01(b) * _clip01(c)) ** (1.0 / 3.0))


# ——— Eq. (2): estimated transmission time —————————————————


def estimated_tx_time(
    *,
    key_bits: float,
    loss_p: float,
    delay_s: float,
    bandwidth_bps: float,
) -> float:
    """Eq. (2): T̂^tx_i(C) = τ + B^key / (b(1-ℓ) + ε)."""
    b_eff = float(bandwidth_bps) * max(0.0, 1.0 - float(loss_p))
    return float(delay_s) + float(key_bits) / (b_eff + _EPS)


# ——— Eq. (15)–(18): GCS-side probability estimation ————————


def gcs_coverage_success_prob(
    *,
    norm_dist: float,          # d̄_i(t) ∈ [0,1]
    energy_margin: float,      # m̄^E_i(t) ∈ [0,1]
    path_ctrl_quality: float,  # q̄^path_{i,min}(t) ∈ [0,1]
    path_risk: float,          # R̄^safe_i(t) ∈ [0,1]
    soft_alpha: float = 0.15,  # soft blend: 0=hard geo-mean, 1=arith-mean
) -> float:
    r"""Eq. (16): P̂^cov_{i,G}(t).

    Uses soft geo-mean (α=0.15 by default) so that a single bad factor
    (e.g. path risk) reduces but does not zero out coverage probability.
    """
    return _soft_geo_mean_4(
        1.0 - _clip01(norm_dist),
        _clip01(energy_margin),
        _clip01(path_ctrl_quality),
        1.0 - _clip01(path_risk),
        alpha=float(soft_alpha),
    )


def gcs_return_conditional_prob(
    *,
    link_reliability_margin: float,  # m̄^ℓ_i(t)
    delay_margin: float,            # m̄^T_i(t)
    bandwidth_margin: float,        # b̄^ret_i(t)
    backlog_margin: float,          # m̄^D_t
    min_prob: float = 0.08,         # floor to avoid total exclusion
    soft_alpha: float = 0.30,       # soft blend for return factors
) -> float:
    r"""Eq. (17): P̂^ret|cov_{i,G}(t).

    Uses soft geo-mean (α=0.30 by default) so that poor link conditions
    reduce but do not zero-out the return probability.  The ``min_prob``
    floor (default 0.08) provides a safety net when ALL return factors
    are very bad — this prevents the planner from freezing in heavily
    degraded communication environments.
    """
    raw = _soft_geo_mean_4(
        link_reliability_margin, delay_margin, bandwidth_margin, backlog_margin,
        alpha=float(soft_alpha),
    )
    return float(max(float(min_prob), raw))


def gcs_return_link_suitability(
    *,
    link_reliability_margin: float,
    delay_margin: float,
    bandwidth_margin: float,
) -> float:
    r"""Eq. (18): q̂^ret_i(t) ∈ [0,1]."""
    return _geo_mean_3(link_reliability_margin, delay_margin, bandwidth_margin)


def gcs_completion_probability(
    *,
    coverage_prob: float,
    return_cond_prob: float,
    calibration_factor: float = 1.0,
    min_prob: float = 0.04,          # global floor for exploration
) -> float:
    r"""Eq. (15): P̂^eff_{i,G}(t) = P̂^cov · P̂^ret (product semantics)."""
    raw = float(coverage_prob) * float(return_cond_prob) * float(calibration_factor)
    return _clip01(max(float(min_prob), raw))


# ——— Helper margins (module-level, no @staticmethod) ————————


def _norm_dist(estimated_uav_pos_ne, poi_pos_ne, d_max: float) -> float:
    d = math.hypot(
        estimated_uav_pos_ne[0] - poi_pos_ne[0],
        estimated_uav_pos_ne[1] - poi_pos_ne[1],
    )
    return _clip01(d / max(d_max, _EPS))


def _energy_margin(
    remaining_energy: float,
    fly_energy: float,
    obs_energy: float,
    home_energy: float,
) -> float:
    num = float(remaining_energy) - (float(fly_energy) + float(obs_energy) + float(home_energy))
    return _clip01(num / (float(remaining_energy) + _EPS))


def _link_reliability_margin(loss_p: float, loss_max: float) -> float:
    return _clip01(1.0 - float(loss_p) / max(float(loss_max), _EPS))


def _delay_margin(tx_time: float, t_max: float) -> float:
    return _clip01(1.0 - float(tx_time) / max(float(t_max), _EPS))


def _bandwidth_margin(bandwidth_bps: float, b_max: float) -> float:
    return _clip01(float(bandwidth_bps) / max(float(b_max), _EPS))


def _backlog_margin(backlog_bits: float, backlog_max: float) -> float:
    return _clip01(1.0 - float(backlog_bits) / max(float(backlog_max), _EPS))


def _path_risk(risk: float, risk_max: float) -> float:
    return _clip01(float(risk) / max(float(risk_max), _EPS))


def gcs_completion_probability_from_raw(
    *,
    estimated_uav_pos_ne: tuple[float, float],
    poi_pos_ne: tuple[float, float],
    remaining_energy: float,
    fly_energy_estimate: float,
    obs_energy_estimate: float,
    home_energy_estimate: float,
    path_ctrl_quality: float,
    path_risk_estimate: float,
    poi_loss_p: float,
    poi_delay_s: float,
    poi_bandwidth_bps: float,
    key_bits: float,
    backlog_bits: float,
    loss_max: float,
    tx_time_max: float,
    bandwidth_max: float,
    backlog_max: float,
    risk_max: float,
    dist_max: float,
    calibration_factor: float = 1.0,
) -> float:
    r"""
    Convenience: compute P̂^eff_{i,G}(t) from raw estimates in one call.

    All ``*_estimate`` values come from GCS-side global estimated state S^G_t.
    """
    # Margins
    nd = _norm_dist(estimated_uav_pos_ne, poi_pos_ne, dist_max)
    em = _energy_margin(remaining_energy, fly_energy_estimate, obs_energy_estimate, home_energy_estimate)
    pr = _path_risk(path_risk_estimate, risk_max)

    # Coverage success
    pcov = gcs_coverage_success_prob(
        norm_dist=nd, energy_margin=em, path_ctrl_quality=path_ctrl_quality, path_risk=pr,
    )

    # Return conditional
    tx_time = estimated_tx_time(key_bits=key_bits, loss_p=poi_loss_p, delay_s=poi_delay_s, bandwidth_bps=poi_bandwidth_bps)
    lm = _link_reliability_margin(poi_loss_p, loss_max)
    dm = _delay_margin(tx_time, tx_time_max)
    bm = _bandwidth_margin(poi_bandwidth_bps, bandwidth_max)
    bkm = _backlog_margin(backlog_bits, backlog_max)

    pret = gcs_return_conditional_prob(
        link_reliability_margin=lm, delay_margin=dm, bandwidth_margin=bm, backlog_margin=bkm,
    )

    return gcs_completion_probability(coverage_prob=pcov, return_cond_prob=pret, calibration_factor=calibration_factor)


# ——— Eq. (22)–(24): UAV-side action-conditional probability ———


def uav_coverage_success_prob(
    *,
    goal_already_covered: bool,          # c_g(t)
    obs_suitability: float,              # m̄^obs_g(a,t)
    energy_margin: float,                # m̄^E_U(a,t)
    ctrl_quality: float,                 # q̄^ctrl_U(a,t)
    safety_margin: float,                # m̄^S_U(a,t)
    soft_alpha: float = 0.10,
) -> float:
    r"""Eq. (23): P̂^cov_{g,U}(a, t).

    If goal already covered (c_g=1), returns 1.0.
    Uses soft geo-mean (α=0.10) for consistency with GCS-side.
    """
    if bool(goal_already_covered):
        return 1.0
    return _soft_geo_mean_4(obs_suitability, energy_margin, ctrl_quality, safety_margin, alpha=float(soft_alpha))


def uav_return_conditional_prob(
    *,
    goal_already_returned: bool,         # r_g(t)
    link_reliability_margin: float,      # m̄^ℓ_U(a,t)
    delay_margin: float,                 # m̄^T_U(a,t)
    bandwidth_margin: float,             # b̄_U(a,t)
    backlog_margin: float,               # m̄^D_U(a,t)
    soft_alpha: float = 0.20,
) -> float:
    r"""Eq. (24): P̂^ret|cov_{g,U}(a, t).

    If goal already returned (r_g=1), returns 1.0.
    Uses soft geo-mean (α=0.20) for consistency with GCS-side.
    """
    if bool(goal_already_returned):
        return 1.0
    return _soft_geo_mean_4(link_reliability_margin, delay_margin, bandwidth_margin, backlog_margin, alpha=float(soft_alpha))


def uav_action_conditional_probability(
    *,
    goal_already_covered: bool,
    goal_already_returned: bool,
    obs_suitability: float,
    energy_margin: float,
    ctrl_quality: float,
    safety_margin: float,
    link_reliability_margin: float,
    delay_margin: float,
    bandwidth_margin: float,
    backlog_margin: float,
) -> float:
    r"""Eq. (22): P̂^eff_{g,U}(a, t) = P̂^cov · P̂^ret (product semantics)."""
    pcov = uav_coverage_success_prob(
        goal_already_covered=goal_already_covered,
        obs_suitability=obs_suitability,
        energy_margin=energy_margin,
        ctrl_quality=ctrl_quality,
        safety_margin=safety_margin,
    )
    pret = uav_return_conditional_prob(
        goal_already_returned=goal_already_returned,
        link_reliability_margin=link_reliability_margin,
        delay_margin=delay_margin,
        bandwidth_margin=bandwidth_margin,
        backlog_margin=backlog_margin,
    )
    return float(pcov * pret)


def uav_action_conditional_probability_from_raw(
    *,
    goal_already_covered: bool,
    goal_already_returned: bool,
    # Predicted state after action a
    predicted_pos_ne: tuple[float, float],
    goal_pos_ne: tuple[float, float],
    predicted_remaining_energy: float,
    predicted_loss_p: float,
    predicted_delay_s: float,
    predicted_bandwidth_bps: float,
    predicted_backlog_bits: float,
    predicted_risk: float,
    # Current state for margin computation
    current_energy: float,
    home_energy_need: float,
    safe_energy_margin: float,
    # Thresholds
    obs_radius: float,
    loss_max: float,
    tx_time_max: float,
    bandwidth_max: float,
    backlog_max: float,
    risk_max: float,
    uav_dist_max: float,
    key_bits: float,
) -> float:
    r"""Convenience: UAV action-conditional probability from raw predictions."""
    # Observation suitability
    d_to_goal = math.hypot(
        predicted_pos_ne[0] - goal_pos_ne[0],
        predicted_pos_ne[1] - goal_pos_ne[1],
    )
    if d_to_goal <= obs_radius:
        obs_suit = 1.0
    else:
        obs_suit = _clip01(1.0 - (d_to_goal - obs_radius) / max(uav_dist_max, _EPS))

    # Energy margin
    energy_margin = _clip01(
        (float(predicted_remaining_energy) - float(home_energy_need) - float(safe_energy_margin))
        / (float(current_energy) + _EPS)
    )

    # Control link quality (placeholder — depends on link model)
    ctrl_quality = _clip01(1.0 - float(predicted_loss_p))

    # Safety margin
    safety_margin = _clip01(1.0 - float(predicted_risk) / max(float(risk_max), _EPS))

    # Link margins
    tx_time = estimated_tx_time(
        key_bits=key_bits, loss_p=predicted_loss_p, delay_s=predicted_delay_s, bandwidth_bps=predicted_bandwidth_bps,
    )
    lm = _clip01(1.0 - float(predicted_loss_p) / max(float(loss_max), _EPS))
    dm = _clip01(1.0 - tx_time / max(float(tx_time_max), _EPS))
    bm = _clip01(float(predicted_bandwidth_bps) / max(float(bandwidth_max), _EPS))
    bkm = _clip01(1.0 - float(predicted_backlog_bits) / max(float(backlog_max), _EPS))

    return uav_action_conditional_probability(
        goal_already_covered=goal_already_covered,
        goal_already_returned=goal_already_returned,
        obs_suitability=obs_suit,
        energy_margin=energy_margin,
        ctrl_quality=ctrl_quality,
        safety_margin=safety_margin,
        link_reliability_margin=lm,
        delay_margin=dm,
        bandwidth_margin=bm,
        backlog_margin=bkm,
    )


# ——— Eq. (9): prediction-execution discrepancy —————————————


def prediction_execution_discrepancy(
    *,
    gcs_estimate: float,
    uav_evaluation: float,
) -> float:
    r"""Eq. (9): D_pe(t) = |P̂^eff_{g,G}(t) - P̂^eff_{g,U}(t)|."""
    return abs(float(gcs_estimate) - float(uav_evaluation))
