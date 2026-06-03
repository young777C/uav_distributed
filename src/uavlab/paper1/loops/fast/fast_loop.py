from __future__ import annotations

import math
from dataclasses import dataclass, field

from uavlab.paper1.comm.thresholds import control_link_ok
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import FastCommand, FastObservation, SlowPlan
from uavlab.paper1.loops.fast.fsm import select_comm_mode
from uavlab.paper1.loops.fast.guidance import (
    pick_link_recovery_waypoint,
    pick_nofly_escape_target,
    pick_waypoint,
)
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.fast.tracker import track_to_waypoint
from uavlab.related_models.comm_link_state import LinkState
from uavlab.paper1.sim.env import Paper1Env
from uavlab.paper1.types import FastCommState


@dataclass
class FastLoop:
    env: Paper1Env
    params: FastLoopParams
    contract: Paper1ContractConfig
    _control_link_bad_s: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._control_link_bad_s = 0.0

    @property
    def control_link_lost_duration_s(self) -> float:
        return float(self._control_link_bad_s)

    @property
    def control_link_lost(self) -> bool:
        """Report threshold for slow-loop Warning (debounced, >= report_s)."""
        return bool(self._control_link_bad_s >= float(self._control_link_loss_report_s()))

    def _fsm_thresholds(self) -> dict:
        fl = dict(self.contract.fast_loop or {})
        return dict(fl.get("fsm_thresholds") or {})

    def _control_link_loss_report_s(self) -> float:
        thr = self._fsm_thresholds()
        return float(thr.get("control_link_loss_report_s", thr.get("control_link_loss_cmd_s", 4.0)))

    def step(self, obs: FastObservation, plan: SlowPlan, dt: float) -> FastCommand:
        gid = plan.goal_id
        gx, gy = float(plan.goal_ne[0]), float(plan.goal_ne[1])
        goal_ne = (gx, gy)

        link = LinkState(
            loss_p=float(obs.link_loss_p),
            delay_s=float(getattr(obs, "link_delay_s", self.env.cfg.delay_mean_s)),
            bandwidth_bps=float(getattr(obs, "link_bandwidth_bps", 1_000_000.0)),
        )
        ctrl_ok = control_link_ok(link, self.contract.dual_link)
        if ctrl_ok:
            self._control_link_bad_s = 0.0
        else:
            self._control_link_bad_s += float(dt)

        mode_sw = bool(self.contract.allow_mode_switching) and bool(self.contract.enable_fast_mode_switch)
        mode = select_comm_mode(
            env=self.env,
            obs=obs,
            params=self.params,
            goal_id=gid,
            use_comm_in_fast=bool(self.contract.use_comm_in_fast),
            use_energy_in_fast=bool(self.contract.use_energy_in_fast),
            mode_switching_allowed=mode_sw,
            dt_s=float(dt),
            return_policy=self.contract.return_policy,
            dual_link=self.contract.dual_link,
            link_bandwidth_bps=float(link.bandwidth_bps),
            link_delay_s=float(link.delay_s),
            return_phase=gid is None,
        )

        if mode == FastCommState.BACK:
            target = (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
        elif mode == FastCommState.SAFE:
            fb = goal_ne if gid is not None else (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
            target = pick_nofly_escape_target(
                env=self.env,
                contract=self.contract,
                mission_goal_ne=fb,
            )
        elif mode == FastCommState.REC and bool(self.params.enable_recovery_mode):
            fb = goal_ne if gid is not None else (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
            target = pick_link_recovery_waypoint(
                env=self.env,
                contract=self.contract,
                params=self.params,
                fallback_ne=fb,
            )
        else:
            target = goal_ne if gid is not None else (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))

        if bool(self.contract.allow_waypoint_delta) and mode in (
            FastCommState.INS,
            FastCommState.SAFE,
            FastCommState.REC,
        ):
            wp = pick_waypoint(env=self.env, goal_ne=target, contract=self.contract)
        else:
            wp = target

        dmax = float(self.contract.waypoint_delta_max_m)
        if bool(self.contract.allow_waypoint_delta) and dmax > 0.0:
            px, py = float(self.env.pos_ne[0]), float(self.env.pos_ne[1])
            gx2, gy2 = float(target[0]), float(target[1])
            vx, vy = gx2 - px, gy2 - py
            dist = float(math.hypot(vx, vy))
            if dist > 1e-6:
                ux, uy = vx / dist, vy / dist
                wx, wy = float(wp[0]) - px, float(wp[1]) - py
                cross = abs(wx * uy - wy * ux)
                if cross > dmax:
                    t = max(0.0, min(dist, (ux * wx + uy * wy)))
                    bx, by = px + ux * t, py + uy * t
                    nx, ny = wx - ux * t, wy - uy * t
                    nc = float(math.hypot(nx, ny))
                    if nc > 1e-9:
                        scale = dmax / nc
                        wp = (bx + nx * scale, by + ny * scale)

        approach_goal = goal_ne if gid is not None else (
            float(self.env.cfg.gcs_ne[0]),
            float(self.env.cfg.gcs_ne[1]),
        )
        t_ne, vel = track_to_waypoint(env=self.env, waypoint=wp, contract=self.contract, dt=float(dt))
        return FastCommand(
            target_ne=t_ne,
            next_comm_mode=mode,
            vel_ne_cmd=vel,
            approach_goal_ne=approach_goal,
        )
