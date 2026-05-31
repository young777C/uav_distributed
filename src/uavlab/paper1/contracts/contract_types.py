from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from uavlab.paper1.types import FastCommState

Point2D = Tuple[float, float]


# CouplingPacket 被设计成一个 "thin wrapper"（瘦包装器）结构，是协议中 fast→slow 反馈“包裹层”的最低限度实现。
# 这种写法的核心目的是将 fast→slow 反馈数据的载荷(payload)通过一个简单字典传递，保持协议层的灵活性和可扩展性：
#
# 1. **简洁抽象层**：在系统演化初期，CouplingPacket 只需支持字段透传，无需强制字段结构，因而使用 Dict[str, Any] 存储所有 fast→slow 通信负载，避免过早定型协议格式。
# 2. **便于扩展和兼容**：随着需求细化，可以无侵入地添加、调整 payload 内容，不影响主协议调用方式。未来如需结构化（引入固定 slot 或校验字段），可逐步进化为静态字段结构。
# 3. **灵活绕障和降级**：在“全耦合”（full_coupling）、“无反馈”（no_feedback）、“带事件/不带事件反馈”等多策略下，可以根据策略裁剪 payload，使协议支持多种降级模式。
# 4. **面向未来的演进**：为后续引入 completion、safety、link、backlog、mode 等 summary 字段做了铺垫，不需反复更改接口签名。
#
# 总结：当前的 CouplingPacket 结构是对 fast→slow 反馈数据的轻量级适配层设计，强调灵活、便于协议演化，同时兼顾代码实现的简洁性和未来兼容性。

@dataclass(frozen=True)
class CouplingPacket:
    """
    Minimal fast->slow packet placeholder (thin wrapper stage).

    In the thin-wrapper refactor we keep it lightweight and optional; later we can
    extend with completion/mode/safety/backlog/link summaries.
    """

    payload: Dict[str, Any]

@dataclass(frozen=True)
class LinkStats:
    loss_p: float
    delay_s: float
    bandwidth_bps: float


@dataclass(frozen=True)
class CompletionStatus:
    """
    Fast→slow completion for the **current slow goal** POI (``FastToSlowPacket.goal_id``).

    - ``spatial_complete``: spatial task done for this POI — entered cover region **and**
      dwell/sampling requirement met (Paper1Lite: membership in ``env.covered``). This is
      **independent** of whether key data has been returned over the link; slow-loop event
      replanning on ``spatial_complete`` can therefore overlap with Stx/Srec upload.
    - ``effective``: key return succeeded for this POI (``env.returned``; ``env.effective`` is a legacy mirror).

    ``covered`` is a backward-compatible alias for ``spatial_complete`` (same value).
    """

    spatial_complete: bool
    effective: bool

    @property
    def covered(self) -> bool:
        return bool(self.spatial_complete)


@dataclass(frozen=True)
class SafetyEvents:
    collision: bool
    oob: bool
    energy_low: bool
    energy_depleted: bool
    returned_home: bool
    control_link_lost: bool = False
    control_link_lost_duration_s: float = 0.0


@dataclass(frozen=True)
class FastToSlowPacket:
    """
    Minimal structured fast->slow packet (operationalized from ``paper1_loops.fast_to_slow``).

    Optional fields are controlled by contract.fast_to_slow toggles.
    """

    step: int
    goal_id: Optional[int]
    goal_ne: Point2D
    completion: Optional[CompletionStatus] = None
    link: Optional[LinkStats] = None
    backlog_bits: Optional[float] = None
    mode: Optional[FastCommState] = None
    safety: Optional[SafetyEvents] = None


@dataclass(frozen=True)
class SlowObservation:
    step: int
    fast_to_slow: Optional[FastToSlowPacket] = None


@dataclass(frozen=True)
class SlowPlan:
    goal_id: Optional[int]
    goal_ne: Point2D


@dataclass(frozen=True)
class FastObservation:
    step: int
    pos_ne: Point2D
    comm_mode: FastCommState
    backlog_bits: float
    link_loss_p: float


@dataclass(frozen=True)
class FastCommand:
    target_ne: Point2D
    next_comm_mode: FastCommState
    vel_ne_cmd: Optional[Point2D] = None
    # Slow-loop / mission goal for approach slowdown (not the short pure-pursuit lookahead).
    approach_goal_ne: Optional[Point2D] = None
