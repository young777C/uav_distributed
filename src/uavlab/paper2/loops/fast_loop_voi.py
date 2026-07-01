"""
Paper 2 §5.3: UAV-side probability-driven fast action selection.

Overrides Paper 1 FastLoop:
- Replaces FSM-based mode selection with action-conditional probability
  evaluation (Eq. 22–24, 26–27).
- Each action a ∈ A_U is scored by:
    J_U(a) = P̂^eff_{g,U}(a) + λ_D·Δ_D(a) - λ_M·Δ_M(a)
- Safe action set A^safe_U is pre-filtered by energy, control link, and safety (Eq. 25).
- Reports prediction-execution discrepancy for VoI feedback.

Wraps Paper1 FastLoop for trajectory tracking and waypoint computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from uavlab.paper1.comm.dual_link import Paper1DualLinkThresholds
from uavlab.paper1.comm.thresholds import control_link_ok
from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import (
    FastCommand,
    FastObservation,
    SlowPlan,
)
from uavlab.paper1.loops.fast.fast_loop import FastLoop as Paper1FastLoop
from uavlab.paper1.loops.fast.guidance import (
    pick_link_recovery_waypoint,
    pick_nofly_escape_target,
    pick_waypoint,
)
from uavlab.paper1.loops.fast.policy import FastLoopParams
from uavlab.paper1.loops.fast.tracker import track_to_waypoint
from uavlab.paper1.sim.energy_model import energy_return_need_from_env
from uavlab.paper1.sim.env import Paper1Env
from uavlab.paper1.types import FastCommState

from uavlab.paper2.contracts.prob_model import (
    uav_action_conditional_probability,
    estimated_tx_time,
)


class UavAction(str, Enum):
    """UAV local action set A_U (§5.3)."""

    INSPECT = "inspect"         # 巡检推进 — proceed toward goal
    TRANSMIT = "transmit"       # 数据回传 — transmit key data
    RECOVER = "recover"         # 链路恢复 — improve link quality
    SAFE = "safe"               # 安全撤离 — exit nofly/risk zone
    RETURN = "return"           # 返航 — return to GCS


@dataclass
class ActionEvaluation:
    """Evaluation result for one candidate action."""

    action: UavAction
    prob: float                 # P̂^eff_{g,U}(a, t)
    utility: float              # J_U(a)
    feasibility_mask: int       # G_E · G_Q · G_S (Eq. 25)


@dataclass
class Paper2FastLoop:
    """Probability-driven fast loop (§5.3).

    Extends Paper1's FastLoop by replacing FSM with action-conditional
    probability evaluation.

    Key differences from Paper 1:
    - Actions are scored by P̂^eff + backlog improvement - mode switch cost
    - Safe action set is explicitly pre-filtered
    - Reports local probability evaluation for VoI feedback
    """

    env: Paper1Env
    params: FastLoopParams
    contract: Paper1ContractConfig

    # Paper1 fast loop (delegate for tracking)
    _p1: Paper1FastLoop = field(init=False)

    # Weights for local utility (Eq. 27)
    lambda_d: float = 1.0       # λ_D: backlog improvement weight
    lambda_m: float = 0.5       # λ_M: mode switch cost

    # Internal state
    _previous_mode: FastCommState = FastCommState.INS
    _last_action: UavAction = UavAction.INSPECT
    _last_probability: float = 0.0

    def __post_init__(self) -> None:
        self._p1 = Paper1FastLoop(
            env=self.env,
            params=self.params,
            contract=self.contract,
        )

    def reset(self) -> None:
        self._p1.reset()
        self._previous_mode = FastCommState.INS
        self._last_action = UavAction.INSPECT
        self._last_probability = 0.0

    @property
    def control_link_lost(self) -> bool:
        return self._p1.control_link_lost

    @property
    def control_link_lost_duration_s(self) -> float:
        return self._p1.control_link_lost_duration_s

    @property
    def local_probability(self) -> float:
        """P̂^eff_{g,U}(t) — current local probability evaluation."""
        return float(self._last_probability)

    # ——— Safety feasibility checks (Eq. 25) ———


    def _energy_feasible(self, predicted_energy: float) -> bool:
        """G_E: energy feasibility check."""
        home_need = float(energy_return_need_from_env(self.env))
        safe_margin = float(self.env.cfg.energy_safe_margin)
        return float(predicted_energy) >= home_need + safe_margin

    def _ctrl_link_feasible(self, predicted_loss_p: float) -> bool:
        """G_Q: control link feasibility check."""
        th = self.contract.dual_link
        from uavlab.related_models.comm_link_state import LinkState
        link = LinkState(
            loss_p=float(predicted_loss_p),
            delay_s=0.1,
            bandwidth_bps=1_000_000.0,
        )
        return bool(control_link_ok(link, th))

    def _safety_feasible(self, predicted_pos_ne: Tuple[float, float]) -> bool:
        """G_S: local safety check (nofly zones + adaptive map boundary margin).

        Margin scales with degradation severity: lower margin at higher degradation
        because link quality is already poor everywhere, so boundary exclusion
        costs more (lost coverage) than it saves (OOB prevention).
        """
        if bool(self.env.in_nofly(predicted_pos_ne)):
            return False
        # Adaptive margin: keep 100m minimum at all degradation levels.
        # At high degradation the UAV flies more aggressively near boundaries
        # (INS instead of SAFE), so margins must stay large to prevent OOB.
        margin = float(getattr(self.env.cfg, 'boundary_safety_margin_m', 100.0))
        n_min = float(self.env.cfg.n_min)
        n_max = float(self.env.cfg.n_max)
        px, py = float(predicted_pos_ne[0]), float(predicted_pos_ne[1])
        if px < n_min + margin or px > n_max - margin:
            return False
        if py < n_min + margin or py > n_max - margin:
            return False
        return True

    # ——— Action evaluation ———


    def _evaluate_action(
        self,
        *,
        action: UavAction,
        goal_id: Optional[int],
        goal_ne: Tuple[float, float],
        dt_s: float,
    ) -> ActionEvaluation:
        """Evaluate one action a ∈ A_U (Eq. 22–27)."""
        pos = (float(self.env.pos_ne[0]), float(self.env.pos_ne[1]))
        energy = float(self.env.remaining_energy)
        backlog = float(getattr(self.env, 'backlog_bits', 0.0))
        loss_p = float(getattr(self.env, 'link_loss_p', 0.0))
        delay_s = float(self.env.cfg.delay_mean_s)
        bw = 1_000_000.0
        key_bits = float(self.env.cfg.key_bits_per_poi)

        # Predict action outcome
        pred_pos = (pos[0], pos[1])
        pred_energy = energy
        pred_loss_p = loss_p
        pred_backlog = backlog

        # Use actual cruise speed from env config (default 11.0 m/s)
        speed = float(getattr(self.env.cfg, 'v_xy_cruise', 11.0))

        if action == UavAction.INSPECT and goal_id is not None:
            # Move toward goal
            dx = goal_ne[0] - pos[0]
            dy = goal_ne[1] - pos[1]
            d = math.hypot(dx, dy)
            if d > 1.0:
                step_dist = min(d, speed * dt_s)
                pred_pos = (pos[0] + dx / d * step_dist, pos[1] + dy / d * step_dist)
            pred_energy -= float(self.env.cfg.energy_per_meter) * speed * dt_s

        elif action == UavAction.TRANSMIT:
            # Hover and transmit — energy cost, backlog reduction
            b_eff = bw * (1.0 - loss_p)
            tx_bits = b_eff * dt_s
            pred_backlog = max(0.0, backlog - tx_bits)
            pred_energy -= float(self.env.cfg.energy_hover_per_s) * dt_s

        elif action == UavAction.RECOVER:
            # Move toward GCS (improving link)
            dx = float(self.env.cfg.gcs_ne[0]) - pos[0]
            dy = float(self.env.cfg.gcs_ne[1]) - pos[1]
            d = math.hypot(dx, dy)
            if d > 1.0:
                step_dist = min(d, speed * dt_s)
                pred_pos = (pos[0] + dx / d * step_dist, pos[1] + dy / d * step_dist)
            pred_energy -= float(self.env.cfg.energy_per_meter) * speed * dt_s
            # Use actual link proxy for accurate loss prediction at new position
            from uavlab.paper1.comm.thresholds import link_proxy_at
            pred_link = link_proxy_at(self.env, pred_pos)
            pred_loss_p = float(pred_link.loss_p)

        elif action == UavAction.SAFE:
            # Safe exit — move toward GCS (away from boundaries/threats)
            dx = float(self.env.cfg.gcs_ne[0]) - pos[0]
            dy = float(self.env.cfg.gcs_ne[1]) - pos[1]
            d = math.hypot(dx, dy)
            if d > 1.0:
                step_dist = min(d, speed * dt_s)
                pred_pos = (pos[0] + dx / d * step_dist, pos[1] + dy / d * step_dist)
            pred_energy -= float(self.env.cfg.energy_per_meter) * speed * dt_s
            # Predict improved link at safer position
            from uavlab.paper1.comm.thresholds import link_proxy_at
            pred_link = link_proxy_at(self.env, pred_pos)
            pred_loss_p = float(pred_link.loss_p)

        elif action == UavAction.RETURN:
            # Return to GCS
            dx = float(self.env.cfg.gcs_ne[0]) - pos[0]
            dy = float(self.env.cfg.gcs_ne[1]) - pos[1]
            d = math.hypot(dx, dy)
            if d > 1.0:
                step_dist = min(d, speed * dt_s)
                pred_pos = (pos[0] + dx / d * step_dist, pos[1] + dy / d * step_dist)
            pred_energy -= float(self.env.cfg.energy_per_meter) * speed * dt_s
            # Predict link improvement by moving closer to GCS
            from uavlab.paper1.comm.thresholds import link_proxy_at
            pred_link = link_proxy_at(self.env, pred_pos)
            pred_loss_p = float(pred_link.loss_p)

        # Feasibility masks (Eq. 25)
        g_e = 1.0 if self._energy_feasible(pred_energy) else 0.0
        g_q = 1.0 if self._ctrl_link_feasible(pred_loss_p) else 0.0
        g_s = 1.0 if self._safety_feasible(pred_pos) else 0.0
        feasibility_mask = int(g_e * g_q * g_s)

        # Action-conditional probability (Eq. 22–24)
        goal_already_covered = bool(goal_id is not None and goal_id in self.env.covered)
        goal_already_returned = bool(goal_id is not None and goal_id in self.env.returned)

        obs_radius = float(self.env.cfg.poi_cover_radius) if hasattr(self.env.cfg, 'poi_cover_radius') else 15.0
        uav_dist_max = 500.0
        home_need = float(energy_return_need_from_env(self.env))
        safe_margin = float(self.env.cfg.energy_safe_margin)

        th = self.contract.dual_link
        loss_max = float(th.data_max_loss_p)
        t_max = float(th.data_max_return_time_s)
        # bw_max: actual max bandwidth at GCS, scaled for headroom
        from uavlab.paper1.comm.thresholds import link_proxy_at
        gcs_link = link_proxy_at(self.env, (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1])))
        bw_max = float(gcs_link.bandwidth_bps) * 1.2
        backlog_max = float(self.env.cfg.key_bits_per_poi) * 25.0
        risk_max = 1.0

        # Compute margins
        d_to_goal = math.hypot(pred_pos[0] - goal_ne[0], pred_pos[1] - goal_ne[1])
        obs_suit = 1.0 if d_to_goal <= obs_radius else max(0.0, 1.0 - (d_to_goal - obs_radius) / uav_dist_max)
        energy_margin = max(0.0, (pred_energy - home_need - safe_margin) / (energy + 1e-12))
        ctrl_quality = max(0.0, 1.0 - pred_loss_p)
        safety_margin = 1.0  # simplified: already checked by G_S

        tx_time = estimated_tx_time(key_bits=key_bits, loss_p=pred_loss_p, delay_s=delay_s, bandwidth_bps=bw)
        lm = max(0.0, 1.0 - pred_loss_p / max(loss_max, 1e-12))
        dm = max(0.0, 1.0 - tx_time / max(t_max, 1e-12))
        bm = max(0.0, bw / max(bw_max, 1e-12))
        bkm = max(0.0, 1.0 - pred_backlog / max(backlog_max, 1e-12))

        prob = uav_action_conditional_probability(
            goal_already_covered=goal_already_covered,
            goal_already_returned=goal_already_returned,
            obs_suitability=obs_suit,
            energy_margin=energy_margin,
            ctrl_quality=ctrl_quality,
            safety_margin=safety_margin,
            link_reliability_margin=lm,
            delay_margin=dm,
            bandwidth_margin=bm,
            backlog_margin=bkm,
        )

        # Utility function (Eq. 27 + link-improvement bonus)
        delta_d = (backlog - pred_backlog) / max(backlog_max, 1e-12)
        mode_map = {
            UavAction.INSPECT: FastCommState.INS,
            UavAction.TRANSMIT: FastCommState.TX,
            UavAction.RECOVER: FastCommState.REC,
            UavAction.SAFE: FastCommState.SAFE,
            UavAction.RETURN: FastCommState.BACK,
        }
        new_mode = mode_map[action]
        delta_m = 1.0 if new_mode != self._previous_mode else 0.0

        # Link improvement bonus: when current loss exceeds the data-return threshold,
        # reward actions that reduce loss (move toward GCS). This prevents the UAV
        # from being stuck at a covered POI that has degraded upload link.
        lambda_l = 0.8
        link_improvement = 0.0
        th = self.contract.dual_link
        data_loss_max = float(th.data_max_loss_p)
        # Check return queue pressure for RECOVER bonus
        ret_q = getattr(self.env, 'return_queue', None)
        queue_pending = int(ret_q.pending_count) if ret_q is not None else 0
        queue_backed_up = queue_pending >= 5
        link_degraded = float(loss_p) > data_loss_max

        # RECOVER bonus: move toward GCS when link is bad (improve data throughput)
        if (link_degraded or queue_backed_up) and action in (UavAction.RECOVER, UavAction.RETURN):
            improvement = (float(loss_p) - float(pred_loss_p)) / max(float(loss_p), 1e-12)
            link_improvement = lambda_l * max(0.0, improvement)
            if queue_backed_up:
                link_improvement += 0.3

        # When at a covered POI with pending data: actively manage return.
        # - Link good → TRANSMIT (upload data)
        # - Link bad → RECOVER (move toward GCS to improve link)
        lambda_tx = 1.0
        covered_tx_bonus = 0.0
        covered_recover_bonus = 0.0
        if goal_already_covered and not goal_already_returned and float(backlog) > 0:
            if not link_degraded:
                # Link is good: prefer TRANSMIT to clear backlog
                if action == UavAction.TRANSMIT:
                    covered_tx_bonus = lambda_tx
            else:
                # Link is bad: prefer RECOVER to improve position
                if action == UavAction.RECOVER:
                    covered_recover_bonus = 0.6

        utility = (prob + float(self.lambda_d) * delta_d
                   - float(self.lambda_m) * delta_m
                   + link_improvement
                   + covered_tx_bonus
                   + covered_recover_bonus)

        return ActionEvaluation(
            action=action,
            prob=prob,
            utility=utility,
            feasibility_mask=feasibility_mask,
        )

    # ——— Action hysteresis ———
    _action_commit_step: int = 0          # step when current action was committed
    _committed_action: Optional[UavAction] = None
    MIN_ACTION_DURATION: int = 15         # minimum steps to hold an action


    # ——— Main step (V2: phase-based action restriction) ———


    def step(self, obs: FastObservation, plan: SlowPlan, dt: float) -> tuple[FastCommand, float]:
        """V2 fast-loop step with phase-based action restriction.

        Phase A (goal NOT covered):  {INSPECT, SAFE} — fly to goal, stay safe.
        Phase B (goal covered, data pending): {TRANSMIT, RECOVER} — return data.
        Safety/energy overrides always available.

        Action hysteresis prevents mode oscillation (min 15-step commitment).
        """
        gid = plan.goal_id
        gx, gy = float(plan.goal_ne[0]), float(plan.goal_ne[1])
        goal_ne = (gx, gy)
        current_step = int(obs.step)

        # ── Determine phase ──
        goal_covered_now = bool(gid is not None and gid in self.env.covered)
        goal_returned_now = bool(gid is not None and gid in self.env.returned)
        has_backlog = float(getattr(self.env, 'backlog_bits', 0.0)) > 0
        in_return_phase = gid is None

        # ── Safety / energy override checks ──
        near_boundary = not self._safety_feasible(
            (float(self.env.pos_ne[0]), float(self.env.pos_ne[1])))
        # Energy critical: only when truly cannot reach any more POIs.
        home_need = float(energy_return_need_from_env(self.env))
        energy_critical = float(self.env.remaining_energy) < home_need

        # ── Degradation-aware: at high degradation, prioritize coverage over safety ──
        loss_max_cfg = float(getattr(self.env.cfg, 'distance_loss_max', 0.40))
        high_degradation = loss_max_cfg >= 0.50  # High or Severe

        # ── Phase-based candidate action set (V4: degradation-aware) ──
        if in_return_phase:
            candidate_actions = [UavAction.RETURN]
        elif energy_critical:
            candidate_actions = [UavAction.RETURN, UavAction.RECOVER]
        elif near_boundary and not high_degradation:
            # Low/Medium: stay safe near boundaries
            candidate_actions = [UavAction.SAFE, UavAction.RECOVER]
        elif near_boundary and high_degradation:
            # High/Severe: accept boundary risk, keep flying to goal
            candidate_actions = [UavAction.INSPECT, UavAction.SAFE, UavAction.RECOVER]
        elif goal_covered_now and not goal_returned_now and has_backlog:
            # Phase B: at covered POI, need to return data.
            # At high degradation, limit RECOVER (link improvement is marginal).
            if high_degradation:
                candidate_actions = [UavAction.TRANSMIT]
            else:
                candidate_actions = [UavAction.TRANSMIT, UavAction.RECOVER]
        else:
            # Phase A: flying to uncovered goal
            candidate_actions = [UavAction.INSPECT, UavAction.SAFE]

        # ── Action hysteresis: hold committed action unless overridden ──
        if (self._committed_action is not None
                and self._committed_action in candidate_actions
                and current_step - self._action_commit_step < self.MIN_ACTION_DURATION
                and not near_boundary
                and not energy_critical):
            candidate_actions = [self._committed_action]

        # ── Evaluate candidate actions ──
        evaluations = []
        for act in candidate_actions:
            ev = self._evaluate_action(
                action=act, goal_id=gid, goal_ne=goal_ne, dt_s=float(dt),
            )
            evaluations.append(ev)

        # Filter feasible + always include SAFE/RETURN as fallback
        safe_evals = [e for e in evaluations if e.feasibility_mask >= 1]
        fallback_actions = {UavAction.SAFE, UavAction.RETURN}
        safe_or_fallback = list(safe_evals)
        seen = {e.action for e in safe_evals}
        for e in evaluations:
            if e.action in fallback_actions and e.action not in seen:
                safe_or_fallback.append(e)
                seen.add(e.action)

        if not safe_or_fallback:
            # All actions infeasible — delegate to Paper1 FSM
            p1_cmd = self._p1.step(obs=obs, plan=plan, dt=float(dt))
            self._last_probability = 0.0
            self._last_action = UavAction.SAFE
            self._previous_mode = p1_cmd.next_comm_mode
            self._committed_action = None
            return p1_cmd, 0.0

        # Select best action by utility
        best = max(safe_or_fallback, key=lambda e: e.utility)
        self._last_action = best.action
        self._last_probability = best.prob

        # Update hysteresis state
        if best.action != self._committed_action:
            self._committed_action = best.action
            self._action_commit_step = current_step

        # Map selected action to FastCommand
        mode_map = {
            UavAction.INSPECT: FastCommState.INS,
            UavAction.TRANSMIT: FastCommState.TX,
            UavAction.RECOVER: FastCommState.REC,
            UavAction.SAFE: FastCommState.SAFE,
            UavAction.RETURN: FastCommState.BACK,
        }
        selected_mode = mode_map[best.action]
        self._previous_mode = selected_mode

        # Generate waypoint target based on selected mode
        if selected_mode == FastCommState.BACK:
            target = (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
        elif selected_mode == FastCommState.SAFE:
            fb = goal_ne if gid is not None else (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
            target = pick_nofly_escape_target(
                env=self.env,
                contract=self.contract,
                mission_goal_ne=fb,
            )
        elif selected_mode == FastCommState.REC:
            fb = goal_ne if gid is not None else (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
            target = pick_link_recovery_waypoint(
                env=self.env,
                contract=self.contract,
                params=self.params,
                fallback_ne=fb,
            )
        else:
            target = goal_ne if gid is not None else (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))

        # Apply waypoint delta if allowed
        if bool(self.contract.allow_waypoint_delta) and selected_mode in (
            FastCommState.INS,
            FastCommState.SAFE,
            FastCommState.REC,
        ):
            wp = pick_waypoint(env=self.env, goal_ne=target, contract=self.contract)
        else:
            wp = target

        # Waypoint cross-track clamp
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
                    nc = math.hypot(nx, ny)
                    if nc > 1e-9:
                        scale = dmax / nc
                        wp = (bx + nx * scale, by + ny * scale)

        approach_goal = goal_ne if gid is not None else (
            float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]),
        )
        t_ne, vel = track_to_waypoint(env=self.env, waypoint=wp, contract=self.contract, dt=float(dt))

        cmd = FastCommand(
            target_ne=t_ne,
            next_comm_mode=selected_mode,
            vel_ne_cmd=vel,
            approach_goal_ne=approach_goal,
        )
        return cmd, best.prob
