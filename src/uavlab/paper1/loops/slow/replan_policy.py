"""Replan decision with cooldown and BACK-mode link suppress (v3 P0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.loops.slow.event_classifier import ClassifiedEvents


@dataclass
class ReplanDecision:
    should_replan: bool
    periodic_only: bool
    reason: str
    latch_warning: bool = False


def _cooldown_steps(contract: Paper1ContractConfig, step_hz: float) -> int:
    trig = dict(contract.slow_loop_triggers or {})
    cooldown_s = float(trig.get("replan_cooldown_s", 15.0))
    hz = max(float(step_hz), 1e-6)
    return max(0, int(round(cooldown_s * hz)))


def decide_replan(
    *,
    contract: Paper1ContractConfig,
    classified: ClassifiedEvents,
    steps: int,
    last_replan_step: int,
    step_hz: float,
    periodic_tick: bool,
    pending_warning: bool,
) -> ReplanDecision:
    """
  Decide whether slow loop should replan this tick.

  - Critical (safety/energy): immediate replan, subject to cooldown.
  - Warning (link/control/stuck): latch pending_warning; replan on next periodic tick.
  - Goal edge: always replan (bypass cooldown).
  - Bootstrap (steps==0): always replan.
    """

    if steps == 0:
        return ReplanDecision(True, False, "bootstrap")

    trig_pol = str(
        dict(contract.slow_loop_triggers or {}).get("replan_trigger_policy", "hybrid")
    ).strip().lower()
    cooldown = _cooldown_steps(contract, step_hz)
    since_last = steps - int(last_replan_step)
    in_cooldown = cooldown > 0 and since_last < cooldown and last_replan_step >= 0

    goal_edge = bool(classified.goal_edge)
    critical = bool(classified.has_critical_interrupt)
    warning_now = bool(classified.has_warning)
    pending = bool(pending_warning or warning_now)

    if in_cooldown and critical and not goal_edge:
        critical = False
        pending = True
        latch = True
    else:
        latch = bool(warning_now)

    immediate = goal_edge or critical

    if trig_pol == "periodic":
        should = bool(periodic_tick)
        reason = "periodic" if should else "hold"
    elif trig_pol == "event":
        should = bool(immediate or (periodic_tick and pending))
        if goal_edge:
            reason = "goal_edge"
        elif critical:
            reason = "critical"
        elif periodic_tick and pending:
            reason = "warning_on_periodic"
        else:
            reason = "hold"
    else:
        # hybrid
        should = bool(periodic_tick or immediate or (periodic_tick and pending))
        if goal_edge:
            reason = "goal_edge"
        elif critical:
            reason = "critical"
        elif periodic_tick and pending:
            reason = "warning_on_periodic"
        elif periodic_tick:
            reason = "periodic"
        else:
            reason = "hold"

    periodic_only = bool(should and periodic_tick and not immediate)
    return ReplanDecision(
        should_replan=should,
        periodic_only=periodic_only,
        reason=reason,
        latch_warning=bool(latch),
    )
