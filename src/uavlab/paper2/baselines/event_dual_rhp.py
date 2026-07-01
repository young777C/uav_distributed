"""
Paper 2 §6.3 Table 3: Event-Dual-RHP baseline.

Rule-event-triggered dual-loop strategy:
- GCS: rolling planning using paper1 CP-SAT optimizer.
- UAV: FSM enabled for local response.
- Fast→slow: event-triggered feedback (link drop, low battery, coverage complete,
  safety events) — but NO prediction-execution discrepancy evaluation.
  All events are treated equally (no VoI prioritization).

This is essentially Paper1's FDLC with the base CP-SAT objective.
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
class EventDualRHPSlowLoop:
    """Event-Dual-RHP: Paper1's FDLC with full event feedback but NO VoI/probability model.

    - GCS: uses CP-SAT (coverage + comm + distance objective, NOT probability)
    - UAV: full FSM + event feedback
    - All events trigger feedback equally (no VoI prioritization)
    - No prediction-execution discrepancy computation
    """

    env: Paper1Env
    params: SlowLoopParams
    contract: Paper1ContractConfig
    slow_interval_steps: int

    _p1: Paper1SlowLoop = field(init=False)

    def __post_init__(self):
        # Use Paper1's FDLC config as-is — it already has full event feedback
        self._p1 = Paper1SlowLoop(
            env=self.env,
            params=self.params,
            contract=self.contract,
            slow_interval_steps=int(self.slow_interval_steps),
        )

    def reset(self) -> None:
        self._p1.reset()

    def step(self, obs: SlowObservation) -> SlowPlan:
        return self._p1.step(obs)
