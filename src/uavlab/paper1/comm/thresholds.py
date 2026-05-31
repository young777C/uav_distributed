"""
Dual-link feasibility helpers (UAV-GCS §9, ``UAV_GCS_通信阈值设置说明.md``).
"""

from __future__ import annotations

from uavlab.paper1.comm.dual_link import Paper1DualLinkThresholds, dual_link_thresholds_from_comm
from uavlab.paper1.comm.link_proxy import loss_proxy_at
from uavlab.paper1.sim.env import Paper1Env
from uavlab.related_models.comm_link_state import LinkState
from uavlab.related_models.key_data_return import key_return_time_s, return_success

Point2D = tuple[float, float]

__all__ = [
    "Paper1DualLinkThresholds",
    "dual_link_thresholds_from_comm",
    "control_link_ok",
    "data_return_ok",
    "data_return_time_s",
    "data_path_feasible_at",
    "link_proxy_at",
]


def link_proxy_at(env: Paper1Env, p: Point2D, *, jitter: bool = False) -> LinkState:
    """Deterministic (or sampled) link at ``p`` for planning / feasibility checks."""

    if jitter:
        return env.observe_link_state()

    loss = float(loss_proxy_at(env, p))
    bw = 1_000_000.0 * float(max(0.0, 1.0 - loss))
    return LinkState(
        loss_p=loss,
        delay_s=float(env.cfg.delay_mean_s),
        bandwidth_bps=bw,
    )


def control_link_ok(link: LinkState, th: Paper1DualLinkThresholds) -> bool:
    """§9.1: command / telemetry link acceptable at this instant."""

    return bool(
        float(link.loss_p) <= float(th.control_max_loss_p)
        and float(link.delay_s) <= float(th.control_max_delay_s)
    )


def data_return_time_s(
    link: LinkState,
    key_bits: float,
    th: Paper1DualLinkThresholds,
) -> float:
    return float(key_return_time_s(link=link, key_bits=float(key_bits), params=th.to_return_params()))


def data_return_ok(
    link: LinkState,
    key_bits: float,
    th: Paper1DualLinkThresholds,
) -> bool:
    """§9.2 + Eq. (12): task-data return feasible (loss, delay, transfer time)."""

    return return_success(link=link, key_bits=float(key_bits), params=th.to_return_params())


def data_path_feasible_at(
    env: Paper1Env,
    p: Point2D,
    key_bits: float,
    th: Paper1DualLinkThresholds,
) -> bool:
    """Slow-loop comm mask: can key data from POI at ``p`` be returned within data-link limits?"""

    link = link_proxy_at(env, p, jitter=False)
    return data_return_ok(link, key_bits, th)
