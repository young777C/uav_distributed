"""
CBCP: Connectivity–Battery Constrained Planning baseline.

Key properties (see ``两个外部对比基线_Codex实施说明.md`` §6):
- Fixed-periodic replanning only (no event-driven triggers).
- Objective: weighted sum of distance + comm risk + energy cost (single-step).
- Control-link hard constraint (minimum path quality).
- Energy constraint (return-home feasibility + tour budget).
- **No** return acknowledgement, backlog, FSM mode, or event feedback.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

from uavlab.paper1.baselines.common import BaselinePlannerInput, BaselinePlannerOutput
from uavlab.paper1.loops.slow.planner import (
    SlowWindow,
    _dist,
    build_candidate_window,
    nearest_energy_feasible_poi,
)
from uavlab.paper1.comm.dual_link import Paper1DualLinkThresholds


class CBCPPlanner:
    """
    Connectivity–Battery Constrained Planning baseline.

    At each fixed planning cycle:
      1. Build the POI candidate window (communication + energy aware).
      2. For each candidate, compute a weighted score of:
         - normalised distance
         - communication risk  (1 − min path quality)
         - normalised energy cost
      3. Select the POI with the lowest combined cost that passes energy + control-link constraints.
      4. Execute the selected POI.
      5. Replan only at the next fixed interval.
    """

    def __init__(self) -> None:
        self._last_window: Optional[SlowWindow] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, inp: BaselinePlannerInput) -> BaselinePlannerOutput:
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

        # -- build candidate window (comm + energy aware) --------------------
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

        # -- score candidates and pick the best -----------------------------
        ω_d = float(inp.distance_weight)
        ω_c = float(inp.communication_weight)
        ω_e = float(inp.energy_weight)
        total_w = ω_d + ω_c + ω_e
        if total_w > 0.0:
            ω_d /= total_w
            ω_c /= total_w
            ω_e /= total_w

        best_idx: Optional[int] = None
        best_score = float("inf")
        all_scores: list[float] = []

        # Compute max distance for normalisation (prefer route-distance from node 0).
        max_d = 1.0
        if len(win.d_ij) > 1 and len(win.d_ij[0]) > 1:
            max_d = float(max(win.d_ij[0][1:]) or 1.0)
        max_e = 1e-9
        if len(win.e_fly_ij) > 1 and len(win.e_fly_ij[0]) > 1:
            max_e = float(max(win.e_fly_ij[0][1:]) or 1e-9)

        for i, pid in enumerate(win.poi_ids):
            j = i + 1  # SlowWindow node index (0 = UAV, 1..n = POIs)

            # Skip if energy-infeasible (per-node mask).
            if j < len(win.m_energy_i) and not bool(win.m_energy_i[j]):
                continue
            # Skip if quality-infeasible (control-link mask).
            if j < len(win.m_quality_i) and not bool(win.m_quality_i[j]):
                continue

            d_norm = float(win.d_ij[0][j]) / max_d if max_d > 0 else 0.0
            e_norm = float(win.e_fly_ij[0][j]) / max_e if max_e > 1e-9 else 0.0
            comm_risk = float(max(0.0, 1.0 - float(win.q_path_min_i[j])))

            score = float(ω_d * d_norm + ω_c * comm_risk + ω_e * e_norm)
            all_scores.append(float(score))

            if score < best_score:
                best_score = float(score)
                best_idx = int(i)

        elapsed = time.perf_counter() - t0

        if best_idx is None or best_idx >= len(win.poi_ids):
            # Fallback: no feasible POI in window
            fallback = self._fallback_target(env, inp)
            return BaselinePlannerOutput(
                target_poi_id=fallback,
                planned_sequence=[fallback] if fallback is not None else [],
                objective_value=None,
                solve_time_s=elapsed,
                status="fallback",
                diagnostics={"fallback_reason": "all_infeasible"},
            )

        target_id = int(win.poi_ids[best_idx])
        return BaselinePlannerOutput(
            target_poi_id=target_id,
            planned_sequence=[target_id],
            objective_value=float(best_score),
            solve_time_s=elapsed,
            status="optimal",
            diagnostics={
                "candidate_count": len(win.poi_ids),
                "best_score": float(best_score),
                "weights": {"dist": ω_d, "comm": ω_c, "energy": ω_e},
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_window(env, inp: BaselinePlannerInput) -> SlowWindow:
        """Build candidate window with communication and energy awareness."""
        from uavlab.paper1.loops.slow.planner import SlowLoopParams

        params = SlowLoopParams(
            t_obs_s=float(inp.poi_dwell_s),
            t_safe_s=0.0,
            dual_link=Paper1DualLinkThresholds(
                control_max_loss_p=float(inp.control_max_loss_p),
                data_max_loss_p=float(inp.data_max_loss_p),
                data_max_return_time_s=float(inp.data_max_return_time_s),
            ),
            energy_plan_margin_frac=0.0,
        )
        profile = CBCPPlanner._baseline_struct_profile()

        return build_candidate_window(
            env=env,
            params=params,
            profile=profile,
            window_k=int(inp.window_k),
            prefetch_k_multiplier=int(inp.prefetch_k_multiplier),
            comm_objective=True,          # compute q_poi_i for soft objective
            comm_path_quality_mask=True,  # apply hard control-link mask
            apply_energy_per_node_mask=True,
            path_samples=int(inp.path_samples),
            degrade_level=0,
        )

    @staticmethod
    def _baseline_struct_profile():
        from uavlab.paper1.contracts.struct_profile import StructAxisProfile
        # Aligned with FDLC struct_profile (control_cmd_min_ratio=0.70,
        # control_loss_exposure_max=0.10, energy_plan_margin_frac=0.08,
        # candidate_degrade_max_level=3) for fair comparison.
        return StructAxisProfile(
            use_struct_comm_profile=False,
            control_cmd_min_ratio=0.70,
            control_loss_exposure_max=0.10,
            use_data_return_hard_mask=False,
            comm_ret_objective_weight=0.0,
            energy_plan_margin_frac=0.08,
            candidate_degrade_max_level=3,
            use_control_path_hard_mask=True,
        )

    @staticmethod
    def _fallback_target(env, inp: BaselinePlannerInput) -> Optional[int]:
        from uavlab.paper1.loops.slow.planner import SlowLoopParams

        params = SlowLoopParams(
            t_obs_s=float(inp.poi_dwell_s),
            t_safe_s=0.0,
            dual_link=Paper1DualLinkThresholds(
                control_max_loss_p=float(inp.control_max_loss_p),
                data_max_loss_p=float(inp.data_max_loss_p),
                data_max_return_time_s=float(inp.data_max_return_time_s),
            ),
            energy_plan_margin_frac=0.0,
        )
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
