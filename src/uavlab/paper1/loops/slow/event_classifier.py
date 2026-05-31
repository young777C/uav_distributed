"""Info / Warning / Critical event classification for slow-loop replan (v3 P0)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import FastToSlowPacket, SlowObservation


class EventLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


def _level_rank(lv: EventLevel) -> int:
    return {EventLevel.INFO: 0, EventLevel.WARNING: 1, EventLevel.CRITICAL: 2}[lv]


def max_event_level(a: EventLevel, b: EventLevel) -> EventLevel:
    return a if _level_rank(a) >= _level_rank(b) else b


def _parse_event_level(raw: Any, default: EventLevel) -> EventLevel:
    s = str(raw or "").strip().lower()
    if s in ("critical", "crit"):
        return EventLevel.CRITICAL
    if s in ("warning", "warn"):
        return EventLevel.WARNING
    if s in ("info", "none", "off"):
        return EventLevel.INFO
    return default


def default_event_levels() -> Dict[str, str]:
    return {
        "on_safety_event": "critical",
        "on_energy_low": "critical",
        "on_link_drop": "warning",
        "on_control_link_lost": "warning",
        "on_post_cover_upload_stuck": "warning",
    }


@dataclass(frozen=True)
class ClassifiedEvents:
    max_level: EventLevel
    goal_edge: bool
    reasons: Tuple[str, ...]
    has_critical_interrupt: bool
    has_warning: bool

    @property
    def needs_immediate_replan(self) -> bool:
        return bool(self.has_critical_interrupt)

    @property
    def needs_warning_replan(self) -> bool:
        return bool(self.has_warning)


def _in_back_mode(pkt: Optional[FastToSlowPacket]) -> bool:
    if pkt is None or pkt.mode is None:
        return False
    m = pkt.mode
    s = str(getattr(m, "value", m)).strip().lower()
    return s in ("sback", "back")


def classify_slow_events(
    *,
    contract: Paper1ContractConfig,
    obs: SlowObservation,
    ev_flags: Dict[str, Any],
    post_cover_upload_stuck: bool,
    goal_edge: bool = False,
) -> ClassifiedEvents:
    """
    Map fast→slow packet + local stuck flags to Info / Warning / Critical.

    Link/control events are **Warning** by default (never Critical interrupt).
    Safety/energy remain Critical when enabled.
  Goal-edge detection is handled by ``SlowLoop`` (rising-edge on completion).
    """

    if str(contract.coupling_mode or "").strip().lower() == "no_replan":
        return ClassifiedEvents(EventLevel.INFO, False, (), False, False)
    if obs.fast_to_slow is None:
        return ClassifiedEvents(EventLevel.INFO, False, (), False, False)

    trig = dict(contract.slow_loop_triggers or {})
    levels_raw = trig.get("event_levels") if isinstance(trig.get("event_levels"), dict) else {}
    levels = {**default_event_levels(), **{str(k): str(v) for k, v in dict(levels_raw).items()}}

    suppress_in_back = bool(trig.get("suppress_link_interrupt_in_back", True))
    in_back = _in_back_mode(obs.fast_to_slow)
    ev_on = bool(contract.enable_event_feedback)

    max_lv = EventLevel.INFO
    reasons: List[str] = []
    has_critical = False
    has_warning = False

    pkt = obs.fast_to_slow

    def _apply(name: str, active: bool, trigger_key: str) -> None:
        nonlocal max_lv, has_critical, has_warning
        if not active or not ev_on:
            return
        if not bool(ev_flags.get(trigger_key, True)):
            return
        if in_back and suppress_in_back and trigger_key in (
            "on_link_drop",
            "on_control_link_lost",
        ):
            reasons.append(f"{name}:suppressed_in_back")
            return
        lv = _parse_event_level(levels.get(trigger_key), EventLevel.WARNING)
        max_lv = max_event_level(max_lv, lv)
        reasons.append(f"{name}:{lv.value}")
        if lv == EventLevel.CRITICAL:
            has_critical = True
        elif lv == EventLevel.WARNING:
            has_warning = True

    s = pkt.safety
    if s is not None:
        _apply("safety", bool(s.collision or s.oob), "on_safety_event")
        _apply("energy_low", bool(getattr(s, "energy_low", False)), "on_energy_low")
        _apply(
            "control_link_lost",
            bool(getattr(s, "control_link_lost", False)),
            "on_control_link_lost",
        )

    link = pkt.link
    if link is not None:
        thr = float(trig.get("link_drop_loss_p", contract.dual_link.data_weak_loss_p))
        _apply("link_drop", float(link.loss_p) >= thr, "on_link_drop")

    if post_cover_upload_stuck:
        _apply("post_cover_stuck", True, "on_post_cover_upload_stuck")

    if goal_edge:
        reasons.append("goal_edge")

    return ClassifiedEvents(
        max_level=max_lv,
        goal_edge=bool(goal_edge),
        reasons=tuple(reasons),
        has_critical_interrupt=bool(has_critical),
        has_warning=bool(has_warning),
    )
