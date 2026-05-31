from __future__ import annotations

from dataclasses import dataclass

from uavlab.related_models.comm_link_state import LinkState


@dataclass(frozen=True)
class ReturnDecisionParams:
    """
    Paper-1 Eq. (12): r_i = 1( ℓ_t ≤ ℓ_max^ret  ∧  T_i^ret ≤ T_max^ret )
    """

    max_loss_p_for_return: float = 0.60  # ℓ_max^ret
    max_return_time_s: float = 5.0  # T_max^ret
    eps_s: float = 1e-6  # ε in Eq. (10)


def effective_bandwidth_bps(link: LinkState) -> float:
    """
    Paper-1 Eq. (9): b_t^eff = b_t (1 - ℓ_t).
    """

    return float(link.bandwidth_bps) * float(max(0.0, 1.0 - float(link.loss_p)))


def key_return_time_s(*, link: LinkState, key_bits: float, params: ReturnDecisionParams | None = None) -> float:
    """
    Paper-1 Eq. (10): T_i^ret = τ_t + B_i^key / (b_t^eff + ε).
    """

    p = params or ReturnDecisionParams()
    beff = effective_bandwidth_bps(link)
    return float(link.delay_s) + float(key_bits) / float(beff + p.eps_s)


def return_success(*, link: LinkState, key_bits: float, params: ReturnDecisionParams | None = None) -> bool:
    """
    Paper-1 Eq. (12): binary return feasibility.
    """

    p = params or ReturnDecisionParams()
    t_ret = key_return_time_s(link=link, key_bits=key_bits, params=p)
    return bool((float(link.loss_p) <= float(p.max_loss_p_for_return)) and (t_ret <= float(p.max_return_time_s)))

