from __future__ import annotations

from typing import Optional

from uavlab.paper1.contracts.contract_types import FastObservation
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.sim.energy_model import energy_return_need_from_env
from uavlab.paper1.sim.env import Paper1Env
from uavlab.paper1.types import FastCommState


def select_comm_mode(
    *,
    env: Paper1Env,
    obs: FastObservation,
    params: FastLoopParams,
    goal_id: Optional[int],
    use_comm_in_fast: bool,
    use_energy_in_fast: bool,
    mode_switching_allowed: bool,
) -> FastCommState:
    """
    Priority: ``S_safe`` > ``S_back`` (energy only) > data ``S_rec``/``S_tx`` > ``S_ins``.

    Transit (goal not yet covered): default ``S_ins``; do not enter ``S_rec`` on brief link dips.
    Post-coverage pending-return: ``S_tx`` / ``S_rec`` when ``use_comm_in_fast``.
    Control-link weakness does **not** force ``S_back`` (v3 P0-a).
    """

    cur = obs.comm_mode
    if not mode_switching_allowed or not params.enable_fsm:
        return cur if isinstance(cur, FastCommState) else FastCommState(str(cur))

    if params.enable_safety_mode and env.in_nofly(env.pos_ne):
        return FastCommState.SAFE

    if params.enable_back_mode and use_energy_in_fast:
        need = energy_return_need_from_env(env)
        if float(env.remaining_energy) <= float(need):
            return FastCommState.BACK

    if goal_id is None:
        return FastCommState.INS

    gid = int(goal_id)
    if gid not in env.covered:
        return FastCommState.INS

    if not use_comm_in_fast:
        return FastCommState.TX if float(obs.backlog_bits) > 0 else FastCommState.INS

    if float(obs.backlog_bits) <= 0.0:
        return FastCommState.INS

    if not params.enable_recovery_mode:
        return FastCommState.TX

    if float(obs.link_loss_p) <= float(params.link_loss_recover):
        return FastCommState.TX
    return FastCommState.REC
