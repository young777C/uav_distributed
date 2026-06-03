"""Unit tests for fair key-return scheduling across structure variants."""

from __future__ import annotations

from uavlab.paper1.runner.return_scheduling import (
    fixed_upload_tx_gate,
    goal_spatial_complete,
    should_attempt_key_return,
)
from uavlab.paper1.types import FastCommState


class _EnvStub:
    def __init__(self, covered: set[int]) -> None:
        self.covered = covered


def test_goal_spatial_complete() -> None:
    env = _EnvStub({7, 8})
    assert goal_spatial_complete(env=env, goal_id=7)
    assert not goal_spatial_complete(env=env, goal_id=9)
    assert not goal_spatial_complete(env=env, goal_id=None)


def test_fixed_upload_tx_gate_period() -> None:
    assert fixed_upload_tx_gate(step=0, fixed_send_ratio=0.2)
    assert not fixed_upload_tx_gate(step=1, fixed_send_ratio=0.2)
    assert fixed_upload_tx_gate(step=5, fixed_send_ratio=0.2)
    assert fixed_upload_tx_gate(step=0, fixed_send_ratio=1.0)
    assert not fixed_upload_tx_gate(step=0, fixed_send_ratio=0.0)


def test_spatial_complete_attempts_return_in_sins_with_fast_mode_switch() -> None:
    """WCDL/FDLC: covered + backlog must try return even in Sins (fair struct comparison)."""
    assert should_attempt_key_return(
        backlog_bits=100.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=True,
        spatial_complete=True,
        fast_upload_mode="policy",
    )


def test_fixed_mode_gates_spatial_complete() -> None:
    assert should_attempt_key_return(
        backlog_bits=100.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=False,
        spatial_complete=True,
        fast_upload_mode="fixed",
        fixed_send_ratio=0.2,
        step=0,
    )
    assert not should_attempt_key_return(
        backlog_bits=100.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=False,
        spatial_complete=True,
        fast_upload_mode="fixed",
        fixed_send_ratio=0.2,
        step=1,
    )


def test_fixed_mode_return_phase_ungated() -> None:
    for step in (0, 1, 2, 3):
        assert should_attempt_key_return(
            backlog_bits=50.0,
            comm_mode=FastCommState.INS,
            enable_fast_mode_switch=False,
            spatial_complete=False,
            return_phase=True,
            fast_upload_mode="fixed",
            fixed_send_ratio=0.2,
            step=step,
        )


def test_no_backlog_never_attempts() -> None:
    assert not should_attempt_key_return(
        backlog_bits=0.0,
        comm_mode=FastCommState.TX,
        enable_fast_mode_switch=True,
        spatial_complete=True,
    )


def test_return_phase_attempts_with_backlog() -> None:
    assert should_attempt_key_return(
        backlog_bits=50.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=True,
        spatial_complete=False,
        return_phase=True,
    )
    assert not should_attempt_key_return(
        backlog_bits=0.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=True,
        spatial_complete=False,
        return_phase=True,
    )


def test_before_spatial_only_stx_srec_or_cdsl_path() -> None:
    assert should_attempt_key_return(
        backlog_bits=1.0,
        comm_mode=FastCommState.TX,
        enable_fast_mode_switch=True,
        spatial_complete=False,
    )
    assert should_attempt_key_return(
        backlog_bits=1.0,
        comm_mode=FastCommState.REC,
        enable_fast_mode_switch=True,
        spatial_complete=False,
    )
    assert should_attempt_key_return(
        backlog_bits=1.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=False,
        spatial_complete=False,
        fast_upload_mode="policy",
    )
    assert should_attempt_key_return(
        backlog_bits=1.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=True,
        spatial_complete=False,
        fast_upload_mode="policy",
    )
    assert not should_attempt_key_return(
        backlog_bits=1.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=True,
        spatial_complete=False,
        fast_upload_mode="fixed",
    )


def test_policy_ins_transit_can_be_disabled() -> None:
    assert not should_attempt_key_return(
        backlog_bits=1.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=True,
        spatial_complete=False,
        fast_upload_mode="policy",
        return_during_ins_transit=False,
    )


def test_fixed_mode_gates_cdsl_path() -> None:
    assert should_attempt_key_return(
        backlog_bits=1.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=False,
        spatial_complete=False,
        fast_upload_mode="fixed",
        fixed_send_ratio=0.2,
        step=0,
    )
    assert not should_attempt_key_return(
        backlog_bits=1.0,
        comm_mode=FastCommState.INS,
        enable_fast_mode_switch=False,
        spatial_complete=False,
        fast_upload_mode="fixed",
        fixed_send_ratio=0.2,
        step=1,
    )
