"""
Paper 2 §6.3 Table 3: Centralized-RHP baseline.

Centralized rolling horizon planning: GCS selects targets based on fixed-period
state estimation using paper1's CP-SAT optimizer. UAV only executes targets
(track-only, no FSM, no local autonomy, no fast→slow feedback).

This is essentially Paper1's CDSL (centralized dominant single-loop) with
the base paper1 objective (coverage + comm + distance, no probability model).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import SlowObservation, SlowPlan
from uavlab.paper1.loops.fast.fast_loop import FastLoop as Paper1FastLoop
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.loops.slow.slow_loop import SlowLoop as Paper1SlowLoop
from uavlab.paper1.sim.env import Paper1Env


@dataclass
class CentralizedRHPSlowLoop:
    """Centralized-RHP: Paper1's CDSL with periodic replanning only.

    - Uses CP-SAT optimizer (paper1's coverage + comm quality + distance objective)
    - NO event-triggered replanning (periodic only)
    - NO fast→slow feedback
    """

    env: Paper1Env
    params: SlowLoopParams
    contract: Paper1ContractConfig
    slow_interval_steps: int

    _p1: Paper1SlowLoop = field(init=False)

    def __post_init__(self):
        # Force periodic-only replan by modifying contract
        modified = copy.deepcopy(self.contract)
        if hasattr(modified, 'slow_loop_triggers') and modified.slow_loop_triggers is not None:
            triggers = dict(modified.slow_loop_triggers)
            triggers["replan_trigger_policy"] = "periodic"
            object.__setattr__(modified, 'slow_loop_triggers', triggers)
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
