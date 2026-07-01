"""
Paper 2 §5.4: Value-of-Information (VoI) feedback policy.

Implements Eq. (28)–(30): prediction-execution discrepancy driven event feedback.

Key concepts:
- **Prediction-execution discrepancy** D_pe(t): difference between GCS's estimated
  completion probability and UAV's local evaluation of the current goal.
- **Feedback threshold** θ_t: D_pe(t) > θ_t triggers feedback.
- **Critical events** E^crit_t: always trigger feedback regardless of threshold.
- **State update operator** B(·): merges UAV feedback into GCS global estimated state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class CriticalEventType(str, Enum):
    """E^crit_t: critical events that always trigger feedback (§5.4 Eq. 28)."""

    COVERAGE_COMPLETE = "coverage_complete"         # c_g(t) transition 0→1
    DATA_RETURN_COMPLETE = "data_return_complete"   # r_g(t) transition 0→1
    CONTROL_LINK_DROP = "control_link_drop"         # q^ctrl < q^ctrl_min
    LOW_ENERGY = "low_energy"                       # E_t ≤ E_home + E^safe + margin
    LOCAL_SAFETY_RISK = "local_safety_risk"         # nofly / boundary risk
    EXPLICIT_REPLAN_REQUEST = "explicit_replan_request"  # UAV requests replan
    LINK_RECOVERY_OPPORTUNITY = "link_recovery_opportunity"  # link improved significantly


@dataclass(frozen=True)
class FeedbackMessage:
    """m^fb_t: minimal feedback message from UAV to GCS (§5.4).

    Does NOT contain full business data — only state/link/task summaries.
    """

    step: int
    # UAV state summary
    uav_pos_ne: tuple[float, float]
    uav_energy: float
    uav_mode: str
    # Link summary
    link_loss_p: float
    link_delay_s: float
    link_bandwidth_bps: float
    # Backlog summary
    backlog_bits: float
    pending_poi_count: int
    # Task events
    critical_events: List[CriticalEventType]
    # Discrepancy
    prediction_execution_discrepancy: float
    gcs_estimate: float
    uav_evaluation: float
    # Replan request flag
    request_replan: bool = False


@dataclass
class FeedbackDecision:
    """Output of the feedback policy for one slow-loop step."""

    send_feedback: bool               # send_t = 1 ?
    discrepancy: float                # D_pe(t)
    critical_events: List[CriticalEventType]
    trigger_reason: str               # "discrepancy", "critical_event", "both", "none"
    request_replan: bool = False


@dataclass
class VoIFeedbackPolicy:
    """VoI-driven event feedback policy (§5.4).

    Controls:
    - Whether to send feedback based on D_pe(t) > θ_t or E^crit_t.
    - Whether the feedback should trigger GCS replanning.

    Does NOT handle the feedback message transmission itself
    (that is the runner's responsibility).
    """

    # Threshold parameters
    feedback_threshold: float = 0.15           # θ_t: base discrepancy threshold
    min_feedback_interval_steps: int = 5       # cooldown: min steps between feedbacks

    # Allowed critical events (subset filter — empty = all allowed)
    enabled_critical_events: Optional[Set[CriticalEventType]] = None

    # State
    _last_feedback_step: int = -10**9
    _feedback_count: int = 0
    _replan_count: int = 0

    def reset(self) -> None:
        self._last_feedback_step = -10**9
        self._feedback_count = 0
        self._replan_count = 0

    @property
    def feedback_count(self) -> int:
        return int(self._feedback_count)

    @property
    def replan_count(self) -> int:
        return int(self._replan_count)

    def set_threshold(self, threshold: float) -> None:
        """Update θ_t dynamically (called by adaptive scheduler)."""
        self.feedback_threshold = max(0.0, float(threshold))

    def evaluate(
        self,
        *,
        step: int,
        gcs_probability_estimate: float,     # P̂^eff_{g,G}(t)
        uav_probability_evaluation: float,   # P̂^eff_{g,U}(t)
        current_critical_events: List[CriticalEventType],
    ) -> FeedbackDecision:
        """Eq. (28): send_t = 1(D_pe(t) > θ_t ∨ E^crit_t = 1).

        Returns a FeedbackDecision summarising the evaluation.
        """
        # Compute discrepancy
        dpe = abs(float(gcs_probability_estimate) - float(uav_probability_evaluation))

        # Filter critical events
        if self.enabled_critical_events is not None:
            filtered = [e for e in current_critical_events if e in self.enabled_critical_events]
        else:
            filtered = list(current_critical_events)

        # Cooldown check
        in_cooldown = (int(step) - int(self._last_feedback_step)) < int(self.min_feedback_interval_steps)

        # Decision logic
        has_discrepancy = dpe > float(self.feedback_threshold)
        has_critical = len(filtered) > 0

        if in_cooldown and not has_critical:
            # Cooldown suppresses discrepancy-triggered feedback only
            should_send = False
            reason = "cooldown"
        elif has_critical:
            should_send = True
            reason = "critical_event"
            if has_discrepancy:
                reason = "both"
        elif has_discrepancy:
            should_send = True
            reason = "discrepancy"
        else:
            should_send = False
            reason = "none"

        if should_send:
            self._last_feedback_step = int(step)
            self._feedback_count += 1

        # Replan recommendation
        request_replan = bool(has_critical) or (has_discrepancy and dpe > float(self.feedback_threshold) * 2.0)

        return FeedbackDecision(
            send_feedback=should_send,
            discrepancy=dpe,
            critical_events=filtered,
            trigger_reason=reason,
            request_replan=request_replan,
        )


# ——— State update operator B(·) ————————————————————————


def apply_feedback_state_update(
    *,
    gcs_state: Dict[str, Any],
    feedback: FeedbackMessage,
) -> Dict[str, Any]:
    """Eq. (29): S^{G+}_t = B(S^G_t, m^fb_t).

    Updates GCS-side global estimated state based on UAV feedback.
    Returns a new state dict (does not mutate input).
    """
    updated = dict(gcs_state)

    # Update UAV position estimate
    if "estimated_uav_pos" in updated:
        updated["estimated_uav_pos"] = (
            float(feedback.uav_pos_ne[0]),
            float(feedback.uav_pos_ne[1]),
        )

    # Update energy estimate
    if "estimated_energy" in updated:
        updated["estimated_energy"] = float(feedback.uav_energy)

    # Update link estimate from UAV observation
    if "estimated_link" in updated:
        updated["estimated_link"] = {
            "loss_p": float(feedback.link_loss_p),
            "delay_s": float(feedback.link_delay_s),
            "bandwidth_bps": float(feedback.link_bandwidth_bps),
        }

    # Update backlog
    if "estimated_backlog_bits" in updated:
        updated["estimated_backlog_bits"] = float(feedback.backlog_bits)

    # Update task state based on critical events
    task_state = dict(updated.get("task_state", {}))
    for event in feedback.critical_events:
        if event == CriticalEventType.COVERAGE_COMPLETE:
            task_state["last_coverage_complete_step"] = int(feedback.step)
        elif event == CriticalEventType.DATA_RETURN_COMPLETE:
            task_state["last_return_complete_step"] = int(feedback.step)
        elif event == CriticalEventType.LOW_ENERGY:
            task_state["low_energy_detected"] = True
    updated["task_state"] = task_state

    # Append to history
    history: List[Dict[str, Any]] = list(updated.get("feedback_history", []))
    history.append({
        "step": int(feedback.step),
        "discrepancy": float(feedback.prediction_execution_discrepancy),
        "events": [e.value for e in feedback.critical_events],
        "replan_requested": bool(feedback.request_replan),
    })
    updated["feedback_history"] = history

    return updated
