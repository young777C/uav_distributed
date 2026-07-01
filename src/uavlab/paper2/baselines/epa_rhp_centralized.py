"""
Paper 2 §6.3 Table 3: EPA-RHP-Centralized baseline.

Effective-completion-probability-driven centralized rolling planning:
- GCS: uses the probability model (P̂^eff) for target selection (like Paper2SlowLoop)
- UAV: executes targets only (NO local autonomy, NO FSM mode switching)
- NO distributed feedback mechanism
- NO prediction-execution discrepancy evaluation

Isolates the contribution of the probability model from the distributed collaboration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import SlowObservation, SlowPlan
from uavlab.paper1.loops.slow.planner import SlowLoopParams
from uavlab.paper1.loops.slow.slow_loop import SlowLoop as Paper1SlowLoop
from uavlab.paper1.sim.env import Paper1Env

from uavlab.paper2.loops.slow_loop_prob import (
    _compute_poi_probability,
    _beam_search_sequence,
    ProbCandidateScore,
)


@dataclass
class EPACentralizedSlowLoop:
    """EPA-RHP-Centralized: probability-driven centralized planning.

    Uses P̂^eff_{i,G}(t) for target selection (same as Paper2SlowLoop),
    but UAV has NO local autonomy (pure track-to-goal, no FSM).
    No distributed feedback or VoI computation.

    Isolates the benefit of probability modeling from distributed collaboration.
    """

    env: Paper1Env
    params: SlowLoopParams
    contract: Paper1ContractConfig
    slow_interval_steps: int

    _goal_id: Optional[int] = None
    _goal_ne: Tuple[float, float] = (0.0, 0.0)
    _last_replan_step: int = -10**9
    _replan_count: int = 0

    def reset(self) -> None:
        self._goal_id = None
        self._goal_ne = (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
        self._last_replan_step = -10**9
        self._replan_count = 0

    @property
    def replan_count(self) -> int:
        return int(self._replan_count)

    def _build_candidates(self) -> List[int]:
        """Build candidate POI list using distance-only nearest prefetch."""
        done = set(self.env.effective)
        cand: List[Tuple[float, int]] = []
        for poi in self.env.pois:
            pid = int(poi.poi_id)
            if pid in done:
                continue
            d = math.hypot(
                self.env.pos_ne[0] - poi.pos_ne[0],
                self.env.pos_ne[1] - poi.pos_ne[1],
            )
            cand.append((d, pid))
        cand.sort(key=lambda x: x[0])
        # Take up to 2x horizon candidates
        H = 4
        prefetch_n = min(2 * H, len(cand))
        return [pid for _, pid in cand[:prefetch_n]]

    def _solve(self) -> Tuple[Optional[int], List[int]]:
        candidate_ids = self._build_candidates()
        if not candidate_ids:
            return None, []

        uav_pos = (float(self.env.pos_ne[0]), float(self.env.pos_ne[1]))
        est_energy = float(self.env.remaining_energy)
        backlog = float(getattr(self.env, 'backlog_bits', 0.0))

        scores: List[ProbCandidateScore] = []
        for pid in candidate_ids:
            sc = _compute_poi_probability(
                env=self.env,
                poi_id=int(pid),
                params=self.params,
                contract=self.contract,
                estimated_uav_pos=uav_pos,
                estimated_energy=est_energy,
                estimated_backlog_bits=backlog,
            )
            if sc is not None:
                scores.append(sc)

        if not scores:
            return None, []

        map_extent = float(self.env.cfg.n_max) - float(self.env.cfg.n_min)
        seq = _beam_search_sequence(candidates=scores, horizon_h=4, max_dist=map_extent)
        if not seq:
            return None, []
        return seq[0], seq

    def step(self, obs: SlowObservation) -> SlowPlan:
        steps = int(obs.step)
        # Periodic replan only (centralized)
        if steps - self._last_replan_step >= int(self.slow_interval_steps) or self._goal_id is None:
            self._replan_count += 1
            self._last_replan_step = steps
            new_gid, new_seq = self._solve()
            self._goal_id = new_gid
            if new_gid is not None:
                p = self.env.pois[int(new_gid)].pos_ne
                self._goal_ne = (float(p[0]), float(p[1]))
            else:
                self._goal_ne = (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))

        return SlowPlan(goal_id=self._goal_id, goal_ne=self._goal_ne)
