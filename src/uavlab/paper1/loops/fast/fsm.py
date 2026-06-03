from __future__ import annotations

from typing import Optional

from uavlab.paper1.comm.dual_link import Paper1DualLinkThresholds
from uavlab.paper1.contracts.contract_types import FastObservation
from uavlab.paper1.contracts.return_policy import ReturnPolicyConfig
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.sim.energy_model import energy_return_need_from_env
from uavlab.paper1.sim.env import Paper1Env
from uavlab.paper1.types import FastCommState
from uavlab.related_models.comm_link_state import LinkState
from uavlab.related_models.key_data_return import effective_bandwidth_bps


def expected_tx_gain_bits(
    *,
    link_loss_p: float,
    link_bandwidth_bps: float,
    link_delay_s: float,
    dt_s: float,
    dual_link: Paper1DualLinkThresholds,
) -> float:
    """Expected bits transferable this step (``b_eff * dt``)."""

    link = LinkState(
        loss_p=float(link_loss_p),
        bandwidth_bps=float(link_bandwidth_bps),
        delay_s=float(link_delay_s),
    )
    if float(link.loss_p) > float(dual_link.data_max_loss_p):
        return 0.0
    b_eff = float(effective_bandwidth_bps(link))
    return float(max(0.0, b_eff * float(dt_s)))


def select_comm_mode(
    *,
    env: Paper1Env,
    obs: FastObservation,
    params: FastLoopParams,
    goal_id: Optional[int],
    use_comm_in_fast: bool,
    use_energy_in_fast: bool,
    mode_switching_allowed: bool,
    dt_s: float,
    return_policy: ReturnPolicyConfig,
    dual_link: Paper1DualLinkThresholds,
    link_bandwidth_bps: float,
    link_delay_s: float,
    return_phase: bool = False,
) -> FastCommState:
    """
    Priority: ``S_safe`` > ``S_back`` (energy only) > data ``S_rec``/``S_tx`` > ``S_ins``.

    Transit (goal not yet covered): default ``S_ins``; do not enter ``S_rec`` on brief link dips.
    Post-coverage / return-phase pending-return: ``S_tx`` / ``S_rec`` when ``use_comm_in_fast``.
    ``S_tx`` only if ``expected_tx_gain >= min_tx_gain_bits`` (P1).
    Control-link weakness does **not** force ``S_back`` (v3 P0-a).
    """

    cur = obs.comm_mode
    if params.enable_safety_mode and env.in_nofly(env.pos_ne):
        return FastCommState.SAFE

    if not mode_switching_allowed or not params.enable_fsm:
        return cur if isinstance(cur, FastCommState) else FastCommState(str(cur))

    if params.enable_back_mode and use_energy_in_fast:
        need = energy_return_need_from_env(env)
        if float(env.remaining_energy) <= float(need):
            return FastCommState.BACK

    backlog = float(obs.backlog_bits)
    min_gain = float(return_policy.min_tx_gain_bits)

    def _pending_return_mode() -> FastCommState:
        if backlog <= 0.0:
            return FastCommState.INS
        if not use_comm_in_fast:
            return FastCommState.TX
        if not params.enable_recovery_mode:
            return FastCommState.TX
        gain = expected_tx_gain_bits(
            link_loss_p=float(obs.link_loss_p),
            link_bandwidth_bps=float(link_bandwidth_bps),
            link_delay_s=float(link_delay_s),
            dt_s=float(dt_s),
            dual_link=dual_link,
        )
        link_ok = float(obs.link_loss_p) <= float(params.link_loss_recover)
        if link_ok and gain >= min_gain:
            return FastCommState.TX
        if not link_ok:
            return FastCommState.REC
        return FastCommState.INS

    if return_phase or goal_id is None:
        return _pending_return_mode()

    gid = int(goal_id)
    if gid not in env.covered:
        if use_comm_in_fast and bool(return_policy.enable_backlog_gates) and backlog > 0.0:
            pending_count = 0
            q = getattr(env, "return_queue", None)
            if q is not None:
                pending_count = int(q.pending_count)
            phase = return_policy.mission_phase(backlog, pending_count=pending_count)
            if phase in ("balance", "return"):
                return _pending_return_mode()
        return FastCommState.INS

    if not use_comm_in_fast:
        return FastCommState.TX if backlog > 0.0 else FastCommState.INS

    return _pending_return_mode()
