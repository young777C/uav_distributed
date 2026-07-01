"""
Paper 2 §5.4: Adaptive horizon/threshold/period scheduling.

Implements Eq. (31):
  (H_t, θ_t, T_{s,t}) = Ψ(Ê_t, R_Q(t), N_event(t), L_comm(t))

and Table 1 rules for online adjustment of:
- H_t: GCS planning horizon (window length)
- θ_t: VoI feedback threshold
- T_{s,t}: slow-loop replan period
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScheduleState:
    """Current adaptive schedule values."""

    horizon_h: int = 4           # H_t: planning horizon
    feedback_threshold: float = 0.15   # θ_t: VoI threshold
    slow_interval_steps: int = 40      # T_{s,t}: slow-loop period (in steps)


@dataclass
class AdaptiveSchedule:
    """Eq. (31): Ψ(·) — adaptive scheduler for horizon/threshold/period.

    Uses rule-based adjustments (Table 1) based on recent statistics.
    All adjustments are incremental (bounded random walk), not optimal control.
    """

    # Bounds for each adjustable quantity
    horizon_min: int = 2
    horizon_max: int = 12
    threshold_min: float = 0.05
    threshold_max: float = 0.40
    interval_min: int = 10
    interval_max: int = 200

    # Adjustment step sizes
    horizon_step: int = 1
    threshold_step: float = 0.02
    interval_step: int = 5

    # Windows for statistics (in steps)
    stat_window_steps: int = 200

    # Thresholds for decision
    energy_high_ratio: float = 0.50        # Ê_t / E_max > this → high
    energy_low_ratio: float = 0.25         # Ê_t / E_max < this → low
    comm_bad_loss_p: float = 0.30          # Avg loss_p > this → degraded
    event_high_rate: float = 0.05          # Events/step > this → frequent
    comm_load_high_rate: float = 0.10      # Feedback steps / total > this → high load

    # State
    state: ScheduleState = field(default_factory=ScheduleState)

    # Recent statistics (updated externally by runner)
    _recent_energy_ratio: float = 1.0
    _recent_avg_loss_p: float = 0.0
    _recent_event_count: int = 0
    _recent_total_steps: int = 1
    _recent_feedback_steps: int = 0
    _consecutive_high_discrepancy: int = 0
    _consecutive_low_discrepancy: int = 0

    def reset(self) -> None:
        self.state = ScheduleState()
        self._recent_energy_ratio = 1.0
        self._recent_avg_loss_p = 0.0
        self._recent_event_count = 0
        self._recent_total_steps = 1
        self._recent_feedback_steps = 0
        self._consecutive_high_discrepancy = 0
        self._consecutive_low_discrepancy = 0

    def update_statistics(
        self,
        *,
        energy_ratio: float,        # Ê_t / E_max
        avg_loss_p: float,          # R_Q(t) — sliding window avg loss
        event_count: int,           # N_event(t) in recent window
        total_steps: int,           # total steps in recent window
        feedback_steps: int,        # steps with feedback in recent window
        avg_discrepancy: float,     # average |D_pe| in recent window
    ) -> None:
        """Feed recent statistics for scheduling."""
        self._recent_energy_ratio = max(0.0, min(1.0, float(energy_ratio)))
        self._recent_avg_loss_p = max(0.0, min(1.0, float(avg_loss_p)))
        self._recent_event_count = max(0, int(event_count))
        self._recent_total_steps = max(1, int(total_steps))
        self._recent_feedback_steps = max(0, int(feedback_steps))

        # Track consecutive discrepancy trends
        if avg_discrepancy > self.state.feedback_threshold * 1.5:
            self._consecutive_high_discrepancy += 1
            self._consecutive_low_discrepancy = 0
        elif avg_discrepancy < self.state.feedback_threshold * 0.5:
            self._consecutive_high_discrepancy = 0
            self._consecutive_low_discrepancy += 1
        else:
            self._consecutive_high_discrepancy = max(0, self._consecutive_high_discrepancy - 1)
            self._consecutive_low_discrepancy = max(0, self._consecutive_low_discrepancy - 1)

    def step(self) -> ScheduleState:
        """Eq. (31): Compute new (H_t, θ_t, T_{s,t}) based on recent statistics (Table 1)."""
        s = ScheduleState(
            horizon_h=int(self.state.horizon_h),
            feedback_threshold=float(self.state.feedback_threshold),
            slow_interval_steps=int(self.state.slow_interval_steps),
        )

        # ——— H_t: planning horizon ———
        # Increase: energy high, comm stable, few events
        # Decrease: energy tight, comm degraded, frequent events
        if (self._recent_energy_ratio > self.energy_high_ratio
                and self._recent_avg_loss_p < self.comm_bad_loss_p
                and self._recent_event_count / max(1, self._recent_total_steps) < self.event_high_rate):
            s.horizon_h = min(self.horizon_max, s.horizon_h + self.horizon_step)
        elif (self._recent_energy_ratio < self.energy_low_ratio
              or self._recent_avg_loss_p > self.comm_bad_loss_p
              or self._recent_event_count / max(1, self._recent_total_steps) > self.event_high_rate * 2):
            s.horizon_h = max(self.horizon_min, s.horizon_h - self.horizon_step)

        # ——— θ_t: feedback threshold ———
        # Increase: feedback load high, link quality poor (suppress low-value feedback)
        # Decrease: D_pe persistently high or critical events frequent (lower sensitivity)
        feedback_rate = self._recent_feedback_steps / max(1, self._recent_total_steps)
        if feedback_rate > self.comm_load_high_rate and self._recent_avg_loss_p > self.comm_bad_loss_p:
            s.feedback_threshold = min(self.threshold_max, s.feedback_threshold + self.threshold_step)
        elif self._consecutive_high_discrepancy >= 3:
            s.feedback_threshold = max(self.threshold_min, s.feedback_threshold - self.threshold_step)
        elif self._consecutive_low_discrepancy >= 5 and feedback_rate < self.comm_load_high_rate * 0.5:
            s.feedback_threshold = min(self.threshold_max, s.feedback_threshold + self.threshold_step)

        # ——— T_{s,t}: slow-loop period ———
        # Increase: state stable, recent feedback load high (reduce replan freq)
        # Decrease: goal failure, critical events frequent, D_pe increasing
        if feedback_rate > self.comm_load_high_rate and self._consecutive_low_discrepancy >= 3:
            s.slow_interval_steps = min(self.interval_max, s.slow_interval_steps + self.interval_step)
        elif (self._consecutive_high_discrepancy >= 3
              or self._recent_event_count / max(1, self._recent_total_steps) > self.event_high_rate):
            s.slow_interval_steps = max(self.interval_min, s.slow_interval_steps - self.interval_step)

        self.state = s
        return s
