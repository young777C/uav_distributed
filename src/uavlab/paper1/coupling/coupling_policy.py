from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import (
    CompletionStatus,
    FastToSlowPacket,
    LinkStats,
    SafetyEvents,
    SlowPlan,
)
from uavlab.paper1.sim.energy_model import energy_return_need_from_env
from uavlab.paper1.sim.env import Paper1Env

from uavlab.paper1.contracts.contract_types import CouplingPacket


@dataclass(frozen=True)
class CouplingPolicy:
    """
    Builds ``FastToSlowPacket`` from ``contract.fast_to_slow`` and applies ``coupling_mode`` /
    ``enable_event_feedback`` stripping for the slow loop.
    """

    mode: str = "full_coupling"

    def build_fast_to_slow_packet(
        self,
        *,
        contract: Paper1ContractConfig,
        env: Paper1Env,
        step: int,
        link_loss_p: float,
        link_delay_s: float = 0.0,
        link_bandwidth_bps: float = 0.0,
        link_jitter_s: float = 0.0,
        control_link_lost: bool = False,
        control_link_lost_duration_s: float = 0.0,
        plan: Optional[SlowPlan] = None,
    ) -> FastToSlowPacket:
        """
        Build a structured fast→slow packet from ``fast_to_slow`` toggles, then apply:

        - ``coupling_mode == no_feedback``: drop completion/link/backlog/mode/safety (only ``step`` / goal).
        - ``enable_event_feedback == false`` (otherwise): keep ``completion``; drop link,
          backlog, mode, safety (task-only feedback).
        """

        f2s = dict(contract.fast_to_slow or {})
        goal_id = plan.goal_id if plan is not None else None
        goal_ne: Tuple[float, float] = (
            (float(plan.goal_ne[0]), float(plan.goal_ne[1])) if plan is not None else (float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1]))
        )

        completion = None
        if bool(f2s.get("send_completion", True)) and goal_id is not None:
            gid = int(goal_id)
            spatial = bool(gid in env.covered)
            returned = getattr(env, "returned", env.effective)
            completion = CompletionStatus(
                spatial_complete=spatial,
                effective=bool(gid in returned),
            )

        link = None
        if str(f2s.get("send_link_stats", "full")).strip().lower() != "none":
            link = LinkStats(loss_p=float(link_loss_p), delay_s=float(link_delay_s), bandwidth_bps=float(link_bandwidth_bps))

        backlog_bits = float(env.backlog_bits) if bool(f2s.get("send_backlog", True)) else None
        mode = env.comm_mode if bool(f2s.get("send_mode", True)) else None

        safety = None
        if bool(f2s.get("send_safety_events", True)):
            energy_low = float(env.remaining_energy) <= float(energy_return_need_from_env(env))
            safety = SafetyEvents(
                collision=bool(env.terminated_by_collision),
                oob=bool(env.terminated_by_oob),
                energy_low=bool(energy_low),
                energy_depleted=bool(env.terminated_by_energy_depleted),
                returned_home=bool(env.terminated_by_returned_home),
                control_link_lost=bool(control_link_lost),
                control_link_lost_duration_s=float(control_link_lost_duration_s),
            )

        pkt = FastToSlowPacket(
            step=int(step),
            goal_id=goal_id,
            goal_ne=goal_ne,
            completion=completion,
            link=link,
            backlog_bits=backlog_bits,
            mode=mode,
            safety=safety,
        )

        # ``periodic_goal`` / legacy ``no_feedback``: no fast→slow informational payload.
        cm = str(contract.coupling_mode or "").strip().lower()
        if cm in ("periodic_goal", "no_feedback"):
            return FastToSlowPacket(
                step=pkt.step,
                goal_id=pkt.goal_id,
                goal_ne=pkt.goal_ne,
                completion=None,
                link=None,
                backlog_bits=None,
                mode=None,
                safety=None,
            )

        # ``event_driven_goal`` / legacy ``no_fast_switching``: task + safety + link for interrupts; no mode/backlog.
        if cm in ("event_driven_goal", "no_fast_switching"):
            return FastToSlowPacket(
                step=pkt.step,
                goal_id=pkt.goal_id,
                goal_ne=pkt.goal_ne,
                completion=pkt.completion,
                link=pkt.link,
                backlog_bits=None,
                mode=None,
                safety=pkt.safety,
            )

        # ``enable_event_feedback`` false: still send task completion; strip link/mode/safety/backlog.
        if not bool(contract.enable_event_feedback):
            return FastToSlowPacket(
                step=pkt.step,
                goal_id=pkt.goal_id,
                goal_ne=pkt.goal_ne,
                completion=pkt.completion,
                link=None,
                backlog_bits=None,
                mode=None,
                safety=None,
            )

        return pkt

    def build_fast_to_slow(self, *, payload: Dict[str, object]) -> CouplingPacket:
        """
        Back-compat: keep the old placeholder payload wrapper.
        """
        return CouplingPacket(payload=dict(payload))

