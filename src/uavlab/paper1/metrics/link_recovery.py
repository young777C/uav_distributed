"""
Link degradation recovery latency (Paper §4.2.4, metric 4).

Records time from a link-mutation event until stable mission advancement resumes.
See ``docs/paper1_metrics_operationalization.md`` for thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.types import FastCommState


def link_recovery_recorder_from_contract(
    contract: Paper1ContractConfig,
    *,
    dt: float,
    stable_steps_required: int = 5,
) -> "LinkRecoveryRecorder":
    sl = dict(contract.slow_loop_triggers or {})
    fl = dict(contract.fast_loop or {})
    thr = dict(fl.get("fsm_thresholds") or {})
    return LinkRecoveryRecorder(
        link_drop_loss_p=float(
            sl.get("link_drop_loss_p", getattr(contract.dual_link, "data_weak_loss_p", 0.50))
        ),
        link_loss_recover=float(
            thr.get("link_loss_recover", getattr(contract.dual_link, "data_max_loss_p", 0.20))
        ),
        dt=float(dt),
        stable_steps_required=int(stable_steps_required),
    )


@dataclass
class LinkRecoveryRecorder:
    """
    Per-episode recorder for link-degradation recovery latency.

    Event start: ``loss_p`` crosses upward through ``link_drop_loss_p`` (was below, now at/above).
    Fast-loop response: first step with ``comm_mode`` in {Srec, Ssafe, Stx} after event start.
    Recovery: ``loss_p <= link_loss_recover`` and ``comm_mode == Sins`` for ``stable_steps_required`` steps.
    Latency: wall time from event start to recovery confirmation (seconds).
    """

    link_drop_loss_p: float = 0.50
    link_loss_recover: float = 0.20
    dt: float = 0.05
    stable_steps_required: int = 5

    _latencies_s: List[float] = field(default_factory=list)
    _in_event: bool = False
    _t_degrade_s: Optional[float] = None
    _t_response_s: Optional[float] = None
    _stable_count: int = 0
    _prev_loss: float = 0.0
    _prev_above_drop: bool = False

    def reset(self) -> None:
        self._latencies_s.clear()
        self._in_event = False
        self._t_degrade_s = None
        self._t_response_s = None
        self._stable_count = 0
        self._prev_loss = 0.0
        self._prev_above_drop = False

    def observe(
        self,
        *,
        step: int,
        loss_p: float,
        comm_mode: FastCommState,
    ) -> None:
        t_s = float(step) * float(self.dt)
        loss = float(loss_p)
        above_drop = bool(loss >= float(self.link_drop_loss_p))

        if (not self._in_event) and above_drop and (not self._prev_above_drop):
            self._in_event = True
            self._t_degrade_s = t_s
            self._t_response_s = None
            self._stable_count = 0

        if self._in_event:
            mode = comm_mode if isinstance(comm_mode, FastCommState) else FastCommState(str(comm_mode))
            if self._t_response_s is None and mode in (FastCommState.REC, FastCommState.SAFE, FastCommState.TX):
                self._t_response_s = t_s

            stable = bool(loss <= float(self.link_loss_recover)) and mode == FastCommState.INS
            if stable:
                self._stable_count += 1
            else:
                self._stable_count = 0

            if self._stable_count >= int(max(1, self.stable_steps_required)) and self._t_degrade_s is not None:
                self._latencies_s.append(float(t_s - self._t_degrade_s))
                self._in_event = False
                self._t_degrade_s = None
                self._t_response_s = None
                self._stable_count = 0

        self._prev_loss = loss
        self._prev_above_drop = above_drop

    def latencies_s(self) -> List[float]:
        return list(self._latencies_s)
