"""
Shared interfaces for external baseline planners (RHC-Inspection / CBCP).

``BaselinePlannerInput`` wraps the per-replan-cycle state; ``BaselinePlannerOutput``
carries the planning result.  Both dataclasses are plain data — no env coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Point2D = Tuple[float, float]


@dataclass
class BaselinePlannerInput:
    """Snapshot passed to the baseline planner at each planning cycle."""

    current_position: Point2D
    gcs_position: Point2D
    uncovered_poi_ids: List[int]
    remaining_energy: float
    simulation_step: int
    step_hz: int
    poi_dwell_s: float

    # Energy-model coefficients (normalised).
    energy_per_meter: float
    energy_hover_per_s: float
    energy_safe_margin: float

    # Comm dual-link thresholds (control / data).
    control_max_loss_p: float = 0.05
    data_max_loss_p: float = 0.20
    data_max_return_time_s: float = 15.0

    # Candidate-window knobs.
    window_k: int = 16
    prefetch_k_multiplier: int = 4
    horizon_h: int = 6
    path_samples: int = 9

    # CBCP weights (ignored by RHC).
    distance_weight: float = 0.4
    communication_weight: float = 0.4
    energy_weight: float = 0.2

    # Communication map (callable: position → loss_p).
    # Set by the runner before each planning call.
    _link_proxy_fn: Any = None

    # Environment reference for computing path-level comm/energy during window building.
    # Used by the planner internally; never stored across planning cycles.
    _env: Any = None

    # Solver time limit (seconds).
    solver_time_limit_s: float = 2.0


@dataclass
class BaselinePlannerOutput:
    """Result of a single planning cycle."""

    target_poi_id: Optional[int]
    planned_sequence: List[int]
    objective_value: Optional[float]
    solve_time_s: float
    status: str  # "optimal" | "feasible" | "infeasible" | "empty_window" | "error"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
