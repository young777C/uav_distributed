"""Dual-link threshold dataclass (no sim/env imports)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from uavlab.related_models.key_data_return import ReturnDecisionParams


@dataclass(frozen=True)
class Paper1DualLinkThresholds:
    """Recommended defaults from ``UAV_GCS_通信阈值设置说明.md`` §9."""

    control_max_loss_p: float = 0.05
    control_max_delay_s: float = 1.0
    control_max_jitter_s: float = 0.20

    data_max_loss_p: float = 0.20
    data_max_return_time_s: float = 10.0
    data_max_jitter_s: float = 1.0

    data_weak_loss_p: float = 0.50

    @property
    def data_min_path_quality(self) -> float:
        return float(max(0.0, 1.0 - self.data_max_loss_p))

    @property
    def control_min_path_quality(self) -> float:
        return float(max(0.0, 1.0 - self.control_max_loss_p))

    def to_return_params(self) -> ReturnDecisionParams:
        return ReturnDecisionParams(
            max_loss_p_for_return=float(self.data_max_loss_p),
            max_return_time_s=float(self.data_max_return_time_s),
        )


def dual_link_thresholds_from_comm(comm: Mapping[str, Any] | None) -> Paper1DualLinkThresholds:
    c = dict(comm or {})
    ctrl = dict(c.get("control_link") or {})
    data = dict(c.get("data_link") or {})

    def _f(block: Dict[str, Any], key: str, default: float) -> float:
        return float(block.get(key, c.get(key, default)))

    return Paper1DualLinkThresholds(
        control_max_loss_p=_f(ctrl, "max_loss_p", 0.05),
        control_max_delay_s=_f(ctrl, "max_delay_s", 1.0),
        control_max_jitter_s=_f(ctrl, "max_jitter_s", 0.20),
        data_max_loss_p=_f(data, "max_loss_p", 0.20),
        data_max_return_time_s=_f(data, "max_return_time_s", 10.0),
        data_max_jitter_s=_f(data, "max_jitter_s", 1.0),
        data_weak_loss_p=float(
            data.get("weak_loss_p", c.get("data_weak_loss_p", c.get("link_drop_loss_p", 0.50)))
        ),
    )
