from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import math
import random


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


@dataclass(frozen=True)
class LinkState:
    """
    Paper-1 Eq. (1): c_t = (ℓ_t, τ_t, b_t).

    - ℓ_t: loss probability in [0,1]
    - τ_t: delay (seconds)
    - b_t: bandwidth (bps)
    """

    loss_p: float
    delay_s: float
    bandwidth_bps: float
    jitter_s: float = 0.0


@dataclass(frozen=True)
class LinkStateParams:
    """
    Parameters for Paper-1 Eq. (5)–(7).

    This is a direct, reusable implementation of the paper model; it is intentionally
    decoupled from the simulator's `CommChannel`, which may include additional
    engineering knobs.
    """

    d_max_m: float = 250.0

    # ℓ_t = clip(ℓ_min + a_ℓ \bar d^{α_ℓ} + Δℓ_blk M_blk(p_t) + ξ_{ℓ,t}, 0, ℓ_max)
    loss_min: float = 0.05
    loss_max: float = 0.60
    a_loss: float = 0.55
    alpha_loss: float = 1.0
    delta_loss_blk: float = 0.85
    noise_loss_sigma: float = 0.0

    # τ_t = clip(τ_min + a_τ \bar d^{α_τ} + Δτ_blk M_blk(p_t) + ξ_{τ,t}, τ_min, τ_max)
    delay_min_s: float = 0.05
    delay_max_s: float = 0.50
    a_delay: float = 0.30
    alpha_delay: float = 1.0
    delta_delay_blk: float = 0.20
    noise_delay_sigma: float = 0.0

    # b_t = clip(b_max − a_b \bar d^{α_b} − Δb_blk M_blk(p_t) + ξ_{b,t}, b_min, b_max)
    bw_min_bps: float = 1_000.0
    bw_max_bps: float = 1_000_000.0
    a_bw: float = 900_000.0
    alpha_bw: float = 1.0
    delta_bw_blk: float = 400_000.0
    noise_bw_sigma: float = 0.0


def compute_link_state(
    *,
    uav_ne: Tuple[float, float],
    gcs_ne: Tuple[float, float],
    in_blackhole: bool | Callable[[Tuple[float, float]], bool] = False,
    params: Optional[LinkStateParams] = None,
    rng: Optional[random.Random] = None,
) -> LinkState:
    """
    Compute link state using Paper-1 Eq. (1)–(7).
    """

    p = params or LinkStateParams()
    r = rng or random.Random()

    ux, uy = float(uav_ne[0]), float(uav_ne[1])
    gx, gy = float(gcs_ne[0]), float(gcs_ne[1])

    d = float(math.hypot(ux - gx, uy - gy))
    d_bar = float(min(d / max(p.d_max_m, 1e-9), 1.0))

    if callable(in_blackhole):
        m_blk = 1.0 if bool(in_blackhole((ux, uy))) else 0.0
    else:
        m_blk = 1.0 if bool(in_blackhole) else 0.0

    xi_l = r.gauss(0.0, p.noise_loss_sigma) if p.noise_loss_sigma > 0 else 0.0
    xi_t = r.gauss(0.0, p.noise_delay_sigma) if p.noise_delay_sigma > 0 else 0.0
    xi_b = r.gauss(0.0, p.noise_bw_sigma) if p.noise_bw_sigma > 0 else 0.0

    loss_p = p.loss_min + p.a_loss * (d_bar ** p.alpha_loss) + p.delta_loss_blk * m_blk + xi_l
    loss_p = _clip(loss_p, 0.0, p.loss_max)

    delay_s = p.delay_min_s + p.a_delay * (d_bar ** p.alpha_delay) + p.delta_delay_blk * m_blk + xi_t
    delay_s = _clip(delay_s, p.delay_min_s, p.delay_max_s)

    bw = p.bw_max_bps - p.a_bw * (d_bar ** p.alpha_bw) - p.delta_bw_blk * m_blk + xi_b
    bw = _clip(bw, p.bw_min_bps, p.bw_max_bps)

    return LinkState(loss_p=float(loss_p), delay_s=float(delay_s), bandwidth_bps=float(bw))

