"""
Paper 2 §6.3 Table 3: Periodic-Dual-RHP baseline.

Fixed-period dual-loop strategy:
- GCS: rolling planning using paper1 CP-SAT optimizer (periodic replan only).
- UAV: local response (FSM enabled), but ONLY periodic feedback to GCS (no event triggers).

NO prediction-execution discrepancy evaluation.
NO VoI-driven feedback.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import SlowObservation, SlowPlan
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.loops.slow.slow_loop import SlowLoop as Paper1SlowLoop
from uavlab.paper1.sim.env import Paper1Env


@dataclass
class PeriodicDualRHPSlowLoop:
    """Periodic-Dual-RHP: Paper1's WCDL with periodic-only fast→slow feedback.

    - GCS: uses CP-SAT, periodic replanning only
    - UAV: FSM enabled for local response
    - Fast→slow: periodic feedback only (no event triggers, no backlog/mode reporting)
    """

    env: Paper1Env
    params: SlowLoopParams
    contract: Paper1ContractConfig
    slow_interval_steps: int

    _p1: Paper1SlowLoop = field(init=False)

    def __post_init__(self):
        modified = copy.deepcopy(self.contract)
        # Force periodic-only replan
        if hasattr(modified, 'slow_loop_triggers') and modified.slow_loop_triggers is not None:
            triggers = dict(modified.slow_loop_triggers)
            triggers["replan_trigger_policy"] = "periodic"
            # Disable all event triggers
            triggers["event_triggers"] = {
                "on_safety_event": False,
                "on_energy_low": False,
                "on_link_drop": False,
                "on_control_link_lost": False,
                "on_goal_spatial_complete": False,
                "on_goal_effective_complete": False,
            }
            object.__setattr__(modified, 'slow_loop_triggers', triggers)
        # Disable fast→slow feedback
        if hasattr(modified, 'fast_to_slow'):
            f2s = {"send_completion": False, "send_link_stats": "none",
                    "send_backlog": False, "send_mode": False, "send_safety_events": False}
            object.__setattr__(modified, 'fast_to_slow', f2s)
        self._p1 = Paper1SlowLoop(
            env=self.env,
            params=self.params,
            contract=modified,
            slow_interval_steps=int(self.slow_interval_steps),
        )

    def reset(self) -> None:
        self._p1.reset()

    def step(self, obs: SlowObservation) -> SlowPlan:
        return self._p1.step(obs)
