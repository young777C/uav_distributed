from __future__ import annotations

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import (
    CompletionStatus,
    FastToSlowPacket,
    LinkStats,
    SafetyEvents,
    SlowObservation,
)
from uavlab.paper1.loops.fast.fsm import select_comm_mode
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.slow.event_classifier import EventLevel, classify_slow_events
from uavlab.paper1.loops.slow.replan_policy import decide_replan
from uavlab.paper1.types import FastCommState


def _minimal_contract(**slow_overrides) -> Paper1ContractConfig:
    sl = {
        "replan_trigger_policy": "hybrid",
        "replan_cooldown_s": 15.0,
        "suppress_link_interrupt_in_back": True,
        "event_triggers": {
            "on_control_link_lost": True,
            "on_link_drop": True,
            "on_safety_event": True,
            "on_energy_low": True,
        },
        **slow_overrides,
    }
    return Paper1ContractConfig.from_cfg(
        {
            "paper1_loops": {
                "coupling_mode": "full_coupling",
                "semantics": {"enable_event_feedback": True, "structure": "fdlc"},
                "slow_loop": sl,
            }
        },
        waypoint_delta_max_m=5.0,
    )


def _env_stub(*, covered=None):
    cov = set(covered or [])

    class _Env:
        comm_mode = FastCommState.INS
        covered = cov
        remaining_energy = 1.0
        pos_ne = (0.0, 0.0)
        cfg = type("C", (), {"gcs_ne": (0.0, 0.0)})()

        def in_nofly(self, _pos):
            return False

    return _Env()


def test_fsm_no_back_on_weak_control_link():
    """P0-a: sustained control loss must not force S_back."""
    obs = type(
        "O",
        (),
        {"comm_mode": FastCommState.INS, "backlog_bits": 0.0, "link_loss_p": 0.35},
    )()
    params = FastLoopParams(enable_back_mode=True, enable_recovery_mode=True)
    mode = select_comm_mode(
        env=_env_stub(),
        obs=obs,
        params=params,
        goal_id=3,
        use_comm_in_fast=True,
        use_energy_in_fast=False,
        mode_switching_allowed=True,
    )
    assert mode != FastCommState.BACK


def test_event_classifier_control_link_is_warning_not_critical():
    contract = _minimal_contract()
    pkt = FastToSlowPacket(
        step=10,
        goal_id=1,
        goal_ne=(0.0, 0.0),
        safety=SafetyEvents(
            collision=False,
            oob=False,
            energy_low=False,
            energy_depleted=False,
            returned_home=False,
            control_link_lost=True,
            control_link_lost_duration_s=4.5,
        ),
    )
    classified = classify_slow_events(
        contract=contract,
        obs=SlowObservation(step=10, fast_to_slow=pkt),
        ev_flags=dict(contract.slow_loop_triggers.get("event_triggers") or {}),
        post_cover_upload_stuck=False,
    )
    assert classified.max_level == EventLevel.WARNING
    assert classified.has_critical_interrupt is False
    assert classified.has_warning is True


def test_event_classifier_suppresses_link_in_back_mode():
    contract = _minimal_contract()
    pkt = FastToSlowPacket(
        step=10,
        goal_id=None,
        goal_ne=(0.0, 0.0),
        mode=FastCommState.BACK,
        safety=SafetyEvents(
            collision=False,
            oob=False,
            energy_low=False,
            energy_depleted=False,
            returned_home=False,
            control_link_lost=True,
        ),
        link=LinkStats(loss_p=0.9, delay_s=0.0, bandwidth_bps=1e6),
    )
    classified = classify_slow_events(
        contract=contract,
        obs=SlowObservation(step=10, fast_to_slow=pkt),
        ev_flags=dict(contract.slow_loop_triggers.get("event_triggers") or {}),
        post_cover_upload_stuck=False,
    )
    assert classified.has_warning is False
    assert "suppressed_in_back" in "".join(classified.reasons)


def test_replan_cooldown_blocks_critical_not_goal_edge():
    contract = _minimal_contract(replan_cooldown_s=15.0)
    from uavlab.paper1.loops.slow.event_classifier import ClassifiedEvents

    critical = ClassifiedEvents(
        max_level=EventLevel.CRITICAL,
        goal_edge=False,
        reasons=("energy_low:critical",),
        has_critical_interrupt=True,
        has_warning=False,
    )
    d_block = decide_replan(
        contract=contract,
        classified=critical,
        steps=100,
        last_replan_step=95,
        step_hz=1.0,
        periodic_tick=False,
        pending_warning=False,
    )
    assert d_block.should_replan is False

    goal = ClassifiedEvents(
        max_level=EventLevel.INFO,
        goal_edge=True,
        reasons=("goal_edge",),
        has_critical_interrupt=False,
        has_warning=False,
    )
    d_goal = decide_replan(
        contract=contract,
        classified=goal,
        steps=100,
        last_replan_step=95,
        step_hz=1.0,
        periodic_tick=False,
        pending_warning=False,
    )
    assert d_goal.should_replan is True


def test_replan_warning_defers_to_periodic():
    contract = _minimal_contract(replan_trigger_policy="event")
    from uavlab.paper1.loops.slow.event_classifier import ClassifiedEvents

    warning = ClassifiedEvents(
        max_level=EventLevel.WARNING,
        goal_edge=False,
        reasons=("control_link_lost:warning",),
        has_critical_interrupt=False,
        has_warning=True,
    )
    d_now = decide_replan(
        contract=contract,
        classified=warning,
        steps=50,
        last_replan_step=0,
        step_hz=1.0,
        periodic_tick=False,
        pending_warning=True,
    )
    assert d_now.should_replan is False

    d_periodic = decide_replan(
        contract=contract,
        classified=warning,
        steps=60,
        last_replan_step=0,
        step_hz=1.0,
        periodic_tick=True,
        pending_warning=True,
    )
    assert d_periodic.should_replan is True
