"""Tests for RHC-Inspection baseline — information isolation and basic correctness."""

from __future__ import annotations

import pytest

from uavlab.paper1.baselines.rhc_inspection import RHCInspectionPlanner


class TestRHCInspectionPlanner:
    """Sanity checks on the planner class itself (no OR-Tools needed)."""

    def test_create(self):
        p = RHCInspectionPlanner()
        assert p is not None
        assert p._last_window is None

    def test_plan_no_env(self):
        p = RHCInspectionPlanner()
        from uavlab.paper1.baselines.common import BaselinePlannerInput

        inp = BaselinePlannerInput(
            current_position=(0, 0),
            gcs_position=(125, 125),
            uncovered_poi_ids=[],
            remaining_energy=0.5,
            simulation_step=0,
            step_hz=5,
            poi_dwell_s=5,
            energy_per_meter=6e-5,
            energy_hover_per_s=3e-5,
            energy_safe_margin=0.2,
            _env=None,
        )
        out = p.plan(inp)
        assert out.status == "error"
        assert out.target_poi_id is None
