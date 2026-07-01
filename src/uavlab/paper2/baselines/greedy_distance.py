"""
Paper 2 §6.3 Table 3: Greedy-Distance baseline.

Nearest-neighbor greedy strategy: selects the next POI by Euclidean distance only.
No communication quality awareness, no completion probability, no distributed feedback.

UAV: pure tracking (no FSM, no local autonomy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from uavlab.paper1.contracts.contract_types import SlowObservation, SlowPlan
from uavlab.paper1.loops.fast.fast_loop import FastLoop as Paper1FastLoop
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.sim.env import Paper1Env


@dataclass
class GreedyDistanceSlowLoop:
    """Greedy-Distance: nearest uncovered POI by Euclidean distance."""

    env: Paper1Env
    params: SlowLoopParams
    contract: object  # unused; kept for API compatibility
    slow_interval_steps: int

    _goal_id: Optional[int] = None
    _goal_ne: tuple[float, float] = (0.0, 0.0)

    def reset(self) -> None:
        self._goal_id = None
        self._goal_ne = (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))

    def _nearest_uncovered(self) -> Optional[int]:
        best_id: Optional[int] = None
        best_d = 1e30
        done = set(self.env.effective)
        for poi in self.env.pois:
            pid = int(poi.poi_id)
            if pid in done:
                continue
            d = math.hypot(
                self.env.pos_ne[0] - poi.pos_ne[0],
                self.env.pos_ne[1] - poi.pos_ne[1],
            )
            if d < best_d:
                best_d = d
                best_id = pid
        return best_id

    def step(self, obs: SlowObservation) -> SlowPlan:
        # Replan every step (pure greedy)
        gid = self._nearest_uncovered()
        self._goal_id = gid
        if gid is not None:
            p = self.env.pois[int(gid)].pos_ne
            self._goal_ne = (float(p[0]), float(p[1]))
        else:
            self._goal_ne = (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
        return SlowPlan(goal_id=self._goal_id, goal_ne=self._goal_ne)
