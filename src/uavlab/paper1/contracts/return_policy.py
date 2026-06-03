"""Backlog-aware return policy (struct axis P1 / P1.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _normalize_struct_key(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    legacy = {
        "b1_centralized_single_loop": "cdsl",
        "b3_decouple_dual_loop": "wcdl",
        "full_architecture": "fdlc",
    }
    return legacy.get(s, s)


def _default_upload_stuck_s(chunk: float) -> float:
    """Wall-clock stall before upload-stuck recovery (scales with per-POI transfer time)."""

    if chunk <= 1_500_000.0:
        return 60.0
    if chunk <= 5_000_000.0:
        return 30.0
    return 45.0


def _default_hard_mult(chunk: float) -> float:
    if chunk <= 1_500_000.0:
        return 3.0
    if chunk <= 5_000_000.0:
        return 5.0
    return 6.0


@dataclass(frozen=True)
class ReturnPolicyConfig:
    """
    Thresholds derived from ``comm.data_chunk_bits`` unless overridden in YAML.

    - ``backlog_soft_bits`` / ``backlog_hard_bits``: mission phase gates (FDLC).
    - ``backlog_soft_poi_count`` / ``backlog_hard_poi_count``: optional POI-count gates.
    - ``min_tx_gain_bits``: minimum ``b_eff * dt`` to enter ``S_tx`` when comm-aware.
    - ``upload_stuck_s``: stall time before upload-stuck recovery (per-POI progress).
    """

    backlog_soft_bits: float
    backlog_hard_bits: float
    min_tx_gain_bits: float
    balance_beta_ret_mult: float = 1.5
    balance_mu_dist_mult: float = 1.25
    enable_backlog_gates: bool = False
    enable_upload_stuck_recovery: bool = False
    upload_stuck_s: float = 30.0
    backlog_soft_poi_count: int = 0
    backlog_hard_poi_count: int = 0
    upload_stuck_min_pending_count: int = 2
    upload_stuck_requires_soft_backlog: bool = True

    def mission_phase(self, backlog_bits: float, *, pending_count: int = 0) -> str:
        """``explore`` | ``balance`` | ``return``."""

        if not bool(self.enable_backlog_gates):
            return "explore"
        b = float(backlog_bits)
        pc = int(pending_count)

        hard = bool(b >= float(self.backlog_hard_bits))
        if int(self.backlog_hard_poi_count) > 0 and pc >= int(self.backlog_hard_poi_count):
            hard = True
        if hard:
            return "return"

        soft = bool(b >= float(self.backlog_soft_bits))
        if int(self.backlog_soft_poi_count) > 0 and pc >= int(self.backlog_soft_poi_count):
            soft = True
        if soft:
            return "balance"
        return "explore"


def return_policy_from_cfg(
    cfg: Mapping[str, Any],
    *,
    struct_key: str,
) -> ReturnPolicyConfig:
    loops = cfg.get("paper1_loops") if isinstance(cfg.get("paper1_loops"), dict) else {}
    comm = cfg.get("comm") if isinstance(cfg.get("comm"), dict) else {}
    rp_raw = loops.get("return_policy") if isinstance(loops.get("return_policy"), dict) else {}

    chunk = float(comm.get("data_chunk_bits", 16_000_000.0))
    soft_mult = float(rp_raw.get("backlog_soft_mult", 2.0))
    hard_mult = float(rp_raw.get("backlog_hard_mult", _default_hard_mult(chunk)))

    key = _normalize_struct_key(struct_key)
    enable_gates = bool(rp_raw.get("enable_backlog_gates", key == "fdlc"))
    enable_stuck_recovery = bool(
        rp_raw.get("enable_upload_stuck_recovery", key == "fdlc")
    )

    min_tx_raw = rp_raw.get("min_tx_gain_bits")
    if min_tx_raw is None:
        min_tx = max(1000.0, 0.001 * chunk)
    else:
        min_tx = float(min_tx_raw)

    soft_poi = int(rp_raw.get("backlog_soft_poi_count", 2 if key == "fdlc" else 0))
    hard_poi = int(rp_raw.get("backlog_hard_poi_count", 5 if key == "fdlc" else 0))
    stuck_min_pending = int(rp_raw.get("upload_stuck_min_pending_count", 2 if key == "fdlc" else 0))
    stuck_requires_soft = bool(rp_raw.get("upload_stuck_requires_soft_backlog", key == "fdlc"))

    upload_stuck_raw = rp_raw.get("upload_stuck_s")
    upload_stuck_s = (
        float(upload_stuck_raw) if upload_stuck_raw is not None else _default_upload_stuck_s(chunk)
    )

    return ReturnPolicyConfig(
        backlog_soft_bits=float(rp_raw.get("backlog_soft_bits", soft_mult * chunk)),
        backlog_hard_bits=float(rp_raw.get("backlog_hard_bits", hard_mult * chunk)),
        min_tx_gain_bits=min_tx,
        balance_beta_ret_mult=float(rp_raw.get("balance_beta_ret_mult", 1.5)),
        balance_mu_dist_mult=float(rp_raw.get("balance_mu_dist_mult", 1.25)),
        enable_backlog_gates=enable_gates,
        enable_upload_stuck_recovery=enable_stuck_recovery,
        upload_stuck_s=upload_stuck_s,
        backlog_soft_poi_count=soft_poi,
        backlog_hard_poi_count=hard_poi,
        upload_stuck_min_pending_count=stuck_min_pending,
        upload_stuck_requires_soft_backlog=stuck_requires_soft,
    )
