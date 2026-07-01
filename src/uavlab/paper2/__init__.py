"""
Paper 2: 基于信息价值反馈的GCS-UAV分布式自适应滚动规划算法

Effective-completion-probability-driven GCS-UAV heterogeneous collaborative
decision-making with VoI feedback.

This package extends Paper 1 (FDLC architecture) with:
1. Effective completion probability model (§4.1)
2. GCS-UAV heterogeneous state model (§4.2)
3. Prediction-execution discrepancy driven VoI feedback (§5.4)
4. Adaptive horizon/threshold/period scheduling (§5.4)
5. New baselines: Greedy-Distance, Centralized-RHP, Periodic-Dual-RHP,
   Event-Dual-RHP, EPA-RHP-Centralized

Paper 1 types and core infrastructure (env, comm, energy) are reused.
"""

from uavlab.paper2.contracts.prob_model import (
    gcs_completion_probability,
    uav_action_conditional_probability,
    prediction_execution_discrepancy,
)
from uavlab.paper2.contracts.feedback_policy import (
    VoIFeedbackPolicy,
    FeedbackDecision,
)
from uavlab.paper2.contracts.adaptive_schedule import (
    AdaptiveSchedule,
    ScheduleState,
)

__all__ = [
    "gcs_completion_probability",
    "uav_action_conditional_probability",
    "prediction_execution_discrepancy",
    "VoIFeedbackPolicy",
    "FeedbackDecision",
    "AdaptiveSchedule",
    "ScheduleState",
]
