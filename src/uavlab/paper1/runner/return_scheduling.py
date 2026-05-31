from __future__ import annotations

from typing import Any, Optional

from uavlab.paper1.types import FastCommState


def goal_spatial_complete(*, env: Any, goal_id: Optional[int]) -> bool:
    """True when the slow-loop goal POI has met on-site coverage (``env.covered``)."""

    if goal_id is None:
        return False
    covered = getattr(env, "covered", None)
    if covered is None:
        return False
    return int(goal_id) in covered


def fixed_upload_tx_gate(*, step: int, fixed_send_ratio: float) -> bool:
    """
    Deterministic duty cycle for ``fast_upload_mode=fixed``.

    ``fixed_send_ratio=0.2`` → attempt upload on every 5th step (``step % 5 == 0``).
    """

    ratio = float(max(0.0, min(1.0, fixed_send_ratio)))
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    period = max(1, int(round(1.0 / ratio)))
    return int(step) % period == 0


def should_attempt_key_return(
    *,
    backlog_bits: float,
    comm_mode: Any,
    enable_fast_mode_switch: bool,
    spatial_complete: bool = False,
    return_phase: bool = False,
    fast_upload_mode: str = "policy",
    fixed_send_ratio: float = 0.5,
    step: int = 0,
) -> bool:
    """
    When to call ``env.progress_key_return`` in the main loop.

    **Policy mode (struct fair comparison):** once the current goal is spatially complete and
    key data is pending (``backlog_bits > 0``), attempt return **regardless of**
    fast FSM mode (``Sins`` / ``Srec`` / ``Stx``).

    **Fixed mode (coupling axis):** gate attempts by ``fixed_send_ratio`` duty cycle except
    during return phase (``goal_id is None``), where backlog must still be cleared.

    - ``return_phase``: try whenever backlog is pending while homing to GCS.
    - ``spatial_complete`` + policy: always try.
    - ``spatial_complete`` + fixed: duty-gated.
    - Not yet covered: ``Stx`` / ``Srec``, or CDSL path (``enable_fast_mode_switch`` false);
      fixed mode duty-gates the CDSL path as well.
    """

    if float(backlog_bits) <= 0.0:
        return False
    if bool(return_phase):
        return True

    mode_fixed = str(fast_upload_mode).strip().lower() == "fixed"

    if bool(spatial_complete):
        if mode_fixed:
            return fixed_upload_tx_gate(step=int(step), fixed_send_ratio=fixed_send_ratio)
        return True

    if isinstance(comm_mode, FastCommState):
        mode = str(comm_mode.value)
    else:
        mode = str(comm_mode)
    if mode in ("Stx", "Srec"):
        return True
    if not bool(enable_fast_mode_switch):
        if mode_fixed:
            return fixed_upload_tx_gate(step=int(step), fixed_send_ratio=fixed_send_ratio)
        return True
    return False
