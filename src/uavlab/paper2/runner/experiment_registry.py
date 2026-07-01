"""
Paper 2 experiment registry — maps baseline names to (slow_loop_cls, fast_loop_cls).

Used by the paper2 runner to dispatch experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type

# Slow loop classes
from uavlab.paper1.loops.slow.slow_loop import SlowLoop as Paper1SlowLoop
from uavlab.paper2.loops.slow_loop_prob import Paper2SlowLoop
from uavlab.paper2.baselines.greedy_distance import GreedyDistanceSlowLoop
from uavlab.paper2.baselines.centralized_rhp import CentralizedRHPSlowLoop
from uavlab.paper2.baselines.periodic_dual_rhp import PeriodicDualRHPSlowLoop
from uavlab.paper2.baselines.event_dual_rhp import EventDualRHPSlowLoop
from uavlab.paper2.baselines.epa_rhp_centralized import EPACentralizedSlowLoop

# Fast loop classes
from uavlab.paper1.loops.fast.fast_loop import FastLoop as Paper1FastLoop
from uavlab.paper2.loops.fast_loop_voi import Paper2FastLoop


@dataclass(frozen=True)
class ExperimentRecipe:
    """Describes one experiment: which slow/fast loops, feedback policy, and notes."""

    key: str
    description: str
    slow_loop_cls: Type
    fast_loop_cls: Type
    use_voi_feedback: bool = False
    use_prob_model: bool = False
    use_uav_local_autonomy: bool = False
    paper_ref: str = ""
    note: str = ""


# Registry of all baselines
EXPERIMENT_REGISTRY: Dict[str, ExperimentRecipe] = {
    "greedy_distance": ExperimentRecipe(
        key="greedy_distance",
        description="Greedy-Distance: nearest POI by distance only",
        slow_loop_cls=GreedyDistanceSlowLoop,
        fast_loop_cls=Paper1FastLoop,
        use_prob_model=False,
        use_uav_local_autonomy=False,
        paper_ref="Paper 2 §6.3 Table 3",
        note="No comm/energy awareness, no probability, no distributed feedback.",
    ),
    "centralized_rhp": ExperimentRecipe(
        key="centralized_rhp",
        description="Centralized-RHP: CP-SAT periodic rolling, no UAV autonomy",
        slow_loop_cls=CentralizedRHPSlowLoop,
        fast_loop_cls=Paper1FastLoop,
        use_prob_model=False,
        use_uav_local_autonomy=False,
        paper_ref="Paper 2 §6.3 Table 3",
        note="GCS: CP-SAT (coverage+comm+distance). UAV: track-only, no FSM. Periodic replan only.",
    ),
    "periodic_dual_rhp": ExperimentRecipe(
        key="periodic_dual_rhp",
        description="Periodic-Dual-RHP: CP-SAT + periodic feedback only",
        slow_loop_cls=PeriodicDualRHPSlowLoop,
        fast_loop_cls=Paper1FastLoop,
        use_prob_model=False,
        use_uav_local_autonomy=True,
        paper_ref="Paper 2 §6.3 Table 3",
        note="GCS: CP-SAT, periodic replan. UAV: FSM. Fast→slow: periodic only, no event triggers.",
    ),
    "event_dual_rhp": ExperimentRecipe(
        key="event_dual_rhp",
        description="Event-Dual-RHP: CP-SAT + event feedback (no VoI)",
        slow_loop_cls=EventDualRHPSlowLoop,
        fast_loop_cls=Paper1FastLoop,
        use_prob_model=False,
        use_uav_local_autonomy=True,
        paper_ref="Paper 2 §6.3 Table 3",
        note="GCS: CP-SAT with event-triggered replan (paper1 FDLC). No VoI/probability model.",
    ),
    "epa_rhp_centralized": ExperimentRecipe(
        key="epa_rhp_centralized",
        description="EPA-RHP-Centralized: probability-driven centralized",
        slow_loop_cls=EPACentralizedSlowLoop,
        fast_loop_cls=Paper1FastLoop,
        use_prob_model=True,
        use_uav_local_autonomy=False,
        paper_ref="Paper 2 §6.3 Table 3",
        note="GCS: beam-search over P̂^eff. UAV: track-only, no FSM, no feedback.",
    ),
    "d_epa_rhp": ExperimentRecipe(
        key="d_epa_rhp",
        description="D-EPA-RHP: proposed distributed probability-driven with VoI",
        slow_loop_cls=Paper2SlowLoop,
        fast_loop_cls=Paper2FastLoop,
        use_voi_feedback=True,
        use_prob_model=True,
        use_uav_local_autonomy=True,
        paper_ref="Paper 2 §5 (proposed method)",
        note="Full method: GCS beam-search P̂^eff + UAV action-conditional prob + VoI feedback.",
    ),
}


def get_recipe(key: str) -> Optional[ExperimentRecipe]:
    """Look up experiment recipe by key (case-insensitive)."""
    return EXPERIMENT_REGISTRY.get(str(key).strip().lower())


def list_experiments() -> list[str]:
    """Return sorted list of available experiment keys."""
    return sorted(EXPERIMENT_REGISTRY.keys())
