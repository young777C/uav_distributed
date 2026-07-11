"""
RHC-Inspection: rolling-horizon inspection baseline.

Key properties (see ``两个外部对比基线_Codex实施说明.md`` §5):
- Fixed-periodic replanning only (no event-driven triggers).
- Energy constraint (return-home feasibility + tour budget).
- Path connectivity + sub-tour elimination (shared MILP structure).
- **No** communication-quality reward in the objective.
- **No** return acknowledgement, backlog, FSM mode, or event feedback.
- Metrics are still computed by the environment but never fed back to the planner.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

from uavlab.paper1.baselines.common import BaselinePlannerInput, BaselinePlannerOutput
from uavlab.paper1.loops.slow.planner import (
    SlowWindow,
    _dist,
    nearest_energy_feasible_poi,
)
from uavlab.paper1.comm.dual_link import Paper1DualLinkThresholds


@dataclass(frozen=True)
class _RHCOptResult:
    sequence_poi_ids: List[int]
    objective_value: float
    status: str


class RHCInspectionPlanner:
    """
    Rolling-horizon inspection baseline.

    At each fixed planning cycle:
      1. Build the POI candidate window (nearest uncovered, energy-filtered).
      2. Solve a short-horizon ATSP maximising POI count - distance cost.
      3. Execute the first POI in the optimal sequence.
      4. Replan only at the next fixed interval.
    """

    def __init__(self) -> None:
        self._last_window: Optional[SlowWindow] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, inp: BaselinePlannerInput) -> BaselinePlannerOutput:
        """Run one planning cycle.  Returns the next target POI (or ``None`` = GCS)."""
        env = inp._env
        if env is None:
            return BaselinePlannerOutput(
                target_poi_id=None,
                planned_sequence=[],
                objective_value=None,
                solve_time_s=0.0,
                status="error",
                diagnostics={"reason": "no_env"},
            )

        t0 = time.perf_counter()

        # -- build candidate window (energy-filtered only, no comm) ----------
        win = self._build_window(env, inp)
        self._last_window = win

        if not win.poi_ids:
            target = self._fallback_target(env, inp)
            elapsed = time.perf_counter() - t0
            return BaselinePlannerOutput(
                target_poi_id=target,
                planned_sequence=[target] if target is not None else [],
                objective_value=None,
                solve_time_s=elapsed,
                status="fallback",
                diagnostics={"fallback_reason": "empty_window"},
            )

        # -- solve short-horizon tour (visit reward + distance cost) ---------
        horizon_h = int(max(1, inp.horizon_h))
        budget = float(max(0.0, float(env.remaining_energy) - float(inp.energy_safe_margin)))

        result = self._solve_rhc_tour(
            window=win,
            horizon_h=horizon_h,
            mu_dist=float(inp.distance_weight) if inp.distance_weight > 0 else 1.0,
            energy_budget=budget,
            time_limit_s=float(inp.solver_time_limit_s),
        )

        elapsed = time.perf_counter() - t0
        seq = list(result.sequence_poi_ids)
        target = int(seq[0]) if seq else None

        return BaselinePlannerOutput(
            target_poi_id=target,
            planned_sequence=seq,
            objective_value=float(result.objective_value) if result.status != "empty_window" else None,
            solve_time_s=elapsed,
            status=str(result.status).lower(),
            diagnostics={
                "candidate_count": len(win.poi_ids),
                "horizon_h": horizon_h,
                "energy_budget": budget,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_window(env, inp: BaselinePlannerInput) -> SlowWindow:
        """Build energy-filtered candidate window (no communication masking)."""
        params = _baseline_slow_params(inp)
        profile = _baseline_struct_profile()
        from uavlab.paper1.loops.slow.planner import build_candidate_window

        return build_candidate_window(
            env=env,
            params=params,
            profile=profile,
            window_k=int(inp.window_k),
            prefetch_k_multiplier=int(inp.prefetch_k_multiplier),
            comm_objective=False,
            comm_path_quality_mask=False,
            apply_energy_per_node_mask=True,
            path_samples=int(inp.path_samples),
            degrade_level=0,
        )

    @staticmethod
    def _fallback_target(env, inp: BaselinePlannerInput) -> Optional[int]:
        """Nearest energy-feasible POI, else nearest uncovered."""
        params = _baseline_slow_params(inp)
        pid = nearest_energy_feasible_poi(env=env, params=params)
        if pid is not None:
            return int(pid)
        # Final fallback: nearest uncovered POI (distance only)
        best_id, best_d = None, 1e30
        for poi in env.pois:
            p = int(poi.poi_id)
            if p in env.covered:
                continue
            d = _dist(env.pos_ne, poi.pos_ne)
            if d < best_d:
                best_d, best_id = d, p
        return best_id

    # ------------------------------------------------------------------
    # RHC CP-SAT solver:  min  mu * sum(d_ij * x_ij)  -  sum(visit_i)
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_rhc_tour(
        *,
        window: SlowWindow,
        horizon_h: int,
        mu_dist: float,
        energy_budget: float,
        time_limit_s: float = 2.0,
    ) -> _RHCOptResult:
        or_cp = RHCInspectionPlanner._require_ortools()

        n_pois = len(window.poi_ids)
        n_nodes = 1 + n_pois
        if n_nodes <= 1:
            return _RHCOptResult(sequence_poi_ids=[], objective_value=0.0, status="empty_window")

        H = int(horizon_h)
        visit_cap = n_pois if H <= 0 else min(n_pois, H)
        SCALE = 1000

        def cint(x: float) -> int:
            return int(round(float(x) * SCALE))

        model = or_cp.CpModel()
        x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(n_nodes)] for i in range(n_nodes)]

        for i in range(n_nodes):
            model.Add(sum(x[i][j] for j in range(n_nodes)) == 1)
        for j in range(n_nodes):
            model.Add(sum(x[i][j] for i in range(n_nodes)) == 1)
        arcs = [(i, j, x[i][j]) for i in range(n_nodes) for j in range(n_nodes)]
        model.AddCircuit(arcs)
        model.Add(x[0][0] == 0)

        visit = []
        for i in range(1, n_nodes):
            vi = model.NewBoolVar(f"v_{i}")
            model.Add(vi == 1 - x[i][i])
            visit.append(vi)
        model.Add(sum(visit) <= visit_cap)

        # Energy budget constraint.
        fly_terms = [
            cint(float(window.e_fly_ij[i][j])) * x[i][j]
            for i in range(n_nodes) for j in range(n_nodes) if i != j
        ]
        hover_terms = [
            cint(float(window.e_hover_i[i])) * visit[i - 1]
            for i in range(1, n_nodes)
        ]
        model.Add(sum(fly_terms) + sum(hover_terms) <= cint(max(0.0, float(energy_budget))))

        # Objective:  mu * distance  -  visit reward.
        arc_cost = [
            cint(float(mu_dist) * float(window.d_ij[i][j])) * x[i][j]
            for i in range(n_nodes) for j in range(n_nodes) if i != j
        ]
        visit_reward = [cint(-1.0) * visit[i - 1] for i in range(1, n_nodes)]
        model.Minimize(sum(arc_cost) + sum(visit_reward))

        solver = or_cp.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_s)
        status = solver.Solve(model)
        st = solver.StatusName(status)

        if status not in (or_cp.OPTIMAL, or_cp.FEASIBLE):
            return _RHCOptResult(sequence_poi_ids=[], objective_value=float("inf"), status=st)

        seq: List[int] = []
        cur = 0
        for _guard in range(n_nodes + 5):
            nxt = None
            for j in range(n_nodes):
                if int(solver.Value(x[cur][j])) == 1:
                    nxt = j
                    break
            if nxt is None or nxt == 0:
                break
            if 1 <= nxt < n_nodes:
                seq.append(int(window.poi_ids[nxt - 1]))
            cur = nxt
            if cur == 0:
                break

        obj_val = float(solver.ObjectiveValue()) / float(SCALE)
        return _RHCOptResult(sequence_poi_ids=seq, objective_value=obj_val, status=st)

    @staticmethod
    def _require_ortools():
        try:
            from ortools.sat.python import cp_model  # type: ignore
            return cp_model
        except Exception as e:
            raise RuntimeError(
                "OR-Tools is required for RHC tour optimisation. "
                "Install via `pip install ortools`."
            ) from e


# ------------------------------------------------------------------
# Module-level helpers (shared with CBCP)
# ------------------------------------------------------------------

def _baseline_slow_params(inp: BaselinePlannerInput):
    from uavlab.paper1.loops.slow.planner import SlowLoopParams
    return SlowLoopParams(
        t_obs_s=float(inp.poi_dwell_s),
        t_safe_s=0.0,
        lambda_q=0.0,
        mu_dist=1.0,
        beta_ret=0.0,
        dual_link=Paper1DualLinkThresholds(
            control_max_loss_p=float(inp.control_max_loss_p),
            data_max_loss_p=float(inp.data_max_loss_p),
            data_max_return_time_s=float(inp.data_max_return_time_s),
        ),
        energy_plan_margin_frac=0.0,
    )


def _baseline_struct_profile():
    from uavlab.paper1.contracts.struct_profile import StructAxisProfile
    return StructAxisProfile(
        use_struct_comm_profile=False,
        control_cmd_min_ratio=0.0,
        control_loss_exposure_max=1.0,
        use_data_return_hard_mask=False,
        comm_ret_objective_weight=0.0,
        energy_plan_margin_frac=0.0,
        candidate_degrade_max_level=0,
        use_control_path_hard_mask=False,
    )
