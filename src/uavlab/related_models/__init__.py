"""
Paper-1 related mathematical models (standalone, reusable).

This package centralizes the paper's core abstractions so the simulator, planners,
and evaluators can share a single source of truth.
"""

from uavlab.related_models.comm_link_state import LinkState, LinkStateParams, compute_link_state
from uavlab.related_models.energy_budget import (
    EnergyBudgetParams,
    fly_energy,
    hover_energy,
    sequence_energy,
)
from uavlab.related_models.key_data_return import (
    ReturnDecisionParams,
    effective_bandwidth_bps,
    key_return_time_s,
    return_success,
)
from uavlab.related_models.task_completion import (
    effective_completion,
    is_covered,
)

__all__ = [
    "LinkState",
    "LinkStateParams",
    "compute_link_state",
    "ReturnDecisionParams",
    "effective_bandwidth_bps",
    "key_return_time_s",
    "return_success",
    "is_covered",
    "effective_completion",
    "EnergyBudgetParams",
    "fly_energy",
    "hover_energy",
    "sequence_energy",
]

