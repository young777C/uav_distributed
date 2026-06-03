from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig, _slow_loop_pos_int
from uavlab.paper1.contracts.contract_types import SlowObservation, SlowPlan
from uavlab.paper1.loops.slow.event_classifier import classify_slow_events
from uavlab.paper1.loops.slow.replan_policy import decide_replan
from uavlab.paper1.loops.slow.optimizer_ortools import solve_short_horizon_tour
from uavlab.paper1.loops.slow.planner import (
    SlowLoopParams,
    build_candidate_window_with_degrade,
    nearest_energy_feasible_poi,
)
from uavlab.paper1.sim.energy_model import energy_return_need_from_env
from uavlab.paper1.sim.env import Paper1Env


# 我在此脚本上做出大幅度改动，主要原因是为了解决三种架构（CDSL, WCDL, FDLC）在慢环任务决策流程中的降级策略、通信约束逐步放宽、事件反馈分级等问题，这些都直接关联了论文方案实现的合理性和分布式自主任务推进能力的改进。
# 具体来说，核心改动包括：
# 1. 按照「CDSL_WCDL_FDLC_三种架构实现修改清单」的章节，梳理了慢环阶段的候选集过滤、能量判据、安全链路判据，并实现了不同架构下的降级策略（包含正常过滤、逐级降级、软评分、最终返航等）。
# 2. 实现了通信链路可用比例（如 rho, d, j 判据）硬过滤与逐步放松、状态转移与本地反馈机制，并结合 Info/Warning/Critical 事件做架构区分处理。
# 3. 遵循文档建议，把数据回传适宜性从硬性过滤逐步调整为软评分项，避免感知能力增强后系统反而过于保守，充分体现分布式任务推进的优势。
# 4. 能量模型部分拆分了物理安全底线与策略裕度，确保所有架构不突破返航底线，同时让 FDLC 架构下决策更灵活。
# 5. 整体重构的目的是让慢环兼容文档约定的递进关系：集中式架构最保守，分布式架构可灵活降级但不降低底线，从而反映“风险条件下的自适应推进”而不是“感知越多、行动越保守”。

def _periodic_tick(*, steps: int, slow_interval_steps: int, periodic_replan_scope: str) -> bool:
    pol = str(periodic_replan_scope or "").strip().lower()
    if pol == "init_only":
        return steps == 0
    return steps == 0 or (steps % int(max(1, slow_interval_steps)) == 0)


@dataclass
class SlowLoop:
    env: Paper1Env
    params: SlowLoopParams
    contract: Paper1ContractConfig
    slow_interval_steps: int

    _goal_id: Optional[int] = None
    _goal_ne: tuple[float, float] = (0.0, 0.0)
    _last_replan_step: int = -10**9
    _last_goal_spatial_complete: bool = False
    _last_goal_effective_complete: bool = False
    _last_goal_id_for_completion: Optional[int] = None
    _last_sequence: list[int] = field(default_factory=list)
    _future_sequence: list[int] = field(default_factory=list)
    _stuck_since_step: Optional[int] = None
    _last_degrade_level: int = 0
    _pending_warning: bool = False
    _replan_count: int = 0
    last_replan_reason: str = field(default="hold", init=False)
    last_event_level: str = field(default="info", init=False)

    def reset(self) -> None:
        self._goal_id = None
        self._goal_ne = (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
        self._last_replan_step = -10**9
        self._last_goal_spatial_complete = False
        self._last_goal_effective_complete = False
        self._last_goal_id_for_completion = None
        self._last_sequence.clear()
        self._future_sequence.clear()
        self._stuck_since_step = None
        self._last_degrade_level = 0
        self._pending_warning = False
        self._replan_count = 0
        self.last_replan_reason = "hold"
        self.last_event_level = "info"

    @property
    def replan_count(self) -> int:
        return int(self._replan_count)

    @property
    def last_degrade_level(self) -> int:
        return int(self._last_degrade_level)

    def _pending_return_count(self) -> int:
        q = self.env.return_queue
        if q is None:
            return 0
        return int(q.pending_count)

    def _now_s(self) -> float:
        hz = float(max(1, int(self.env.cfg.step_hz)))
        return float(self.env.t) / hz

    def _mission_phase(self) -> str:
        rp = self.contract.return_policy
        return rp.mission_phase(
            float(self.env.backlog_bits),
            pending_count=self._pending_return_count(),
        )

    def _slow_params_with_struct(self) -> SlowLoopParams:
        prof = self.contract.struct_profile
        rp = self.contract.return_policy
        beta_ret = float(prof.comm_ret_objective_weight)
        mu_dist = float(self.params.mu_dist)
        phase = self._mission_phase()
        if phase == "balance":
            beta_ret *= float(rp.balance_beta_ret_mult)
            mu_dist *= float(rp.balance_mu_dist_mult)
        return SlowLoopParams(
            lambda_q=float(self.params.lambda_q),
            mu_dist=mu_dist,
            beta_ret=beta_ret,
            t_obs_s=float(self.params.t_obs_s),
            t_safe_s=float(self.params.t_safe_s),
            dual_link=self.params.dual_link,
            energy_plan_margin_frac=float(prof.energy_plan_margin_frac),
        )

    def _goal_lock_stuck_threshold_steps(self) -> int:
        trig = dict(self.contract.slow_loop_triggers or {})
        stuck_s = float(trig.get("goal_lock_stuck_s", 90.0))
        hz = float(max(1, int(self.env.cfg.step_hz)))
        return max(1, int(round(stuck_s * hz)))

    def _goal_return_stalled(self) -> bool:
        """Per-POI return stall for the active slow-loop goal (P1.5)."""

        gid = self._goal_id
        if gid is None:
            return False
        ip = int(gid)
        if ip in self.env.effective:
            return False
        if ip not in self.env.covered:
            return False
        q = self.env.return_queue
        if q is None:
            return False
        stall_s = float(self.contract.return_policy.upload_stuck_s)
        return bool(q.is_poi_return_stalled(ip, now_s=self._now_s(), stall_s=stall_s))

    def _is_post_cover_upload_stuck(self, obs: SlowObservation) -> bool:
        """Upload stuck event: no return progress on current goal for ``upload_stuck_s``."""

        if self._goal_id is None:
            return False
        return bool(self._goal_return_stalled())

    def _update_stuck_timer(self, obs: SlowObservation) -> None:
        steps = int(obs.step)
        if self._is_post_cover_upload_stuck(obs):
            if self._stuck_since_step is None:
                self._stuck_since_step = steps
        else:
            self._stuck_since_step = None

    def _goal_lock_stuck_timeout(self, obs: SlowObservation) -> bool:
        if self._stuck_since_step is None:
            return False
        elapsed = int(obs.step) - int(self._stuck_since_step)
        return elapsed >= int(self._goal_lock_stuck_threshold_steps())

    def _event_trigger_flags(self) -> dict:
        trig = dict(self.contract.slow_loop_triggers or {})
        ev = trig.get("event_triggers") if isinstance(trig.get("event_triggers"), dict) else {}
        return dict(ev)

    def _goal_edge_from_packet(self, obs: SlowObservation) -> bool:
        if obs.fast_to_slow is None:
            return False
        ev = self._event_trigger_flags()
        use_spatial = bool(ev.get("on_goal_spatial_complete", False))
        use_effective = bool(ev.get("on_goal_effective_complete", ev.get("on_goal_completed", True)))
        if not (use_spatial or use_effective):
            return False
        comp = obs.fast_to_slow.completion
        gid = obs.fast_to_slow.goal_id
        if comp is None or gid is None:
            return False
        same_goal = self._last_goal_id_for_completion == gid
        prev_sp = self._last_goal_spatial_complete if same_goal else False
        prev_ef = self._last_goal_effective_complete if same_goal else False
        now_sp = bool(comp.spatial_complete)
        now_ef = bool(comp.effective)
        spatial_edge = use_spatial and (not prev_sp) and now_sp
        effective_edge = use_effective and (not prev_ef) and now_ef
        self._last_goal_id_for_completion = gid
        self._last_goal_spatial_complete = now_sp
        self._last_goal_effective_complete = now_ef
        return bool(spatial_edge or effective_edge)

    def _should_replan(
        self, *, pol: str, steps: int, obs: SlowObservation
    ) -> Tuple[bool, bool, str, str]:
        if pol == "advance_on_poi_done":
            need = (self._goal_id is None) or (self._goal_id in self.env.effective)
            return need, False, "advance_on_poi_done", "info"

        trig = dict(self.contract.slow_loop_triggers or {})
        trig_pol = str(trig.get("replan_trigger_policy", "event")).strip().lower()
        if trig_pol not in ("periodic", "event", "hybrid"):
            trig_pol = "hybrid"

        periodic = _periodic_tick(
            steps=steps,
            slow_interval_steps=int(self.slow_interval_steps),
            periodic_replan_scope=str(trig.get("periodic_replan_scope", "repeat")),
        )

        goal_edge = self._goal_edge_from_packet(obs)
        classified = classify_slow_events(
            contract=self.contract,
            obs=obs,
            ev_flags=self._event_trigger_flags(),
            post_cover_upload_stuck=self._is_post_cover_upload_stuck(obs),
            goal_edge=goal_edge,
        )

        step_hz = float(max(1, int(self.env.cfg.step_hz)))
        decision = decide_replan(
            contract=self.contract,
            classified=classified,
            steps=steps,
            last_replan_step=int(self._last_replan_step),
            step_hz=step_hz,
            periodic_tick=periodic,
            pending_warning=bool(self._pending_warning or classified.has_warning),
        )

        if decision.latch_warning or classified.has_warning:
            self._pending_warning = True

        immediate = bool(goal_edge or classified.has_critical_interrupt)
        periodic_only = bool(decision.should_replan and periodic and not immediate)

        if decision.should_replan:
            self._pending_warning = False

        self.last_replan_reason = str(decision.reason)
        self.last_event_level = str(classified.max_level.value)
        return decision.should_replan, periodic_only, self.last_replan_reason, self.last_event_level

    def _goal_ne_for(self, gid: Optional[int]) -> tuple[float, float]:
        if gid is None:
            return (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
        p = self.env.pois[int(gid)].pos_ne
        return (float(p[0]), float(p[1]))

    def _mission_elapsed_s(self) -> float:
        hz = float(max(1, int(self.env.cfg.step_hz)))
        return float(self.env.t) / hz

    def _mission_remain_s(self) -> float:
        hz = float(max(1, int(self.env.cfg.step_hz)))
        total_s = float(self.env.cfg.episode_steps) / hz
        return float(max(0.0, total_s - self._mission_elapsed_s()))

    def _energy_return_threshold(self) -> float:
        need = float(energy_return_need_from_env(self.env))
        margin = float(self.env.cfg.energy_safe_margin)
        plan = float(self.contract.struct_profile.energy_plan_margin_frac)
        return float(need + margin + plan)

    def _should_enter_return_phase(self) -> bool:
        n_pois = len(self.env.pois)
        if n_pois <= 0:
            return True
        if len(self.env.effective) >= n_pois:
            return True

        reserve_s = float(self.env.cfg.return_reserve_s)
        if reserve_s > 0.0 and self._mission_remain_s() < reserve_s:
            return True

        if bool(self.contract.use_energy_in_slow):
            if float(self.env.remaining_energy) < self._energy_return_threshold():
                return True

        rp = self.contract.return_policy
        pc = self._pending_return_count()
        if bool(rp.enable_backlog_gates):
            if float(self.env.backlog_bits) >= float(rp.backlog_hard_bits):
                return True
            if int(rp.backlog_hard_poi_count) > 0 and pc >= int(rp.backlog_hard_poi_count):
                return True

        return False

    def _sync_backlog_return_phase(self) -> None:
        """Force homing when backlog crosses the hard gate (FDLC P1)."""

        if self._mission_phase() != "return":
            return
        if self._goal_id is None:
            return
        self._goal_id = None
        self._goal_ne = self._goal_ne_for(None)
        self._last_sequence = []
        self._stuck_since_step = None

    def _upload_stuck_recovery_eligible(self) -> bool:
        """P1.5c: do not force return on the first pending POI / below soft backlog."""

        rp = self.contract.return_policy
        pc = self._pending_return_count()
        b = float(self.env.backlog_bits)
        min_pc = int(rp.upload_stuck_min_pending_count)
        if min_pc > 0 and pc < min_pc:
            return False
        if bool(rp.upload_stuck_requires_soft_backlog):
            soft = bool(b >= float(rp.backlog_soft_bits))
            if int(rp.backlog_soft_poi_count) > 0 and pc >= int(rp.backlog_soft_poi_count):
                soft = True
            if not soft:
                return False
        return True

    def _apply_upload_stuck_recovery(self, obs: SlowObservation) -> bool:
        """
        Post-cover upload stall → enter return phase without goal-changing replan.

        Clears the active goal so the fast loop can focus on ``S_tx``/``S_rec``.
        """

        if not bool(self.contract.return_policy.enable_upload_stuck_recovery):
            return False
        if not self._upload_stuck_recovery_eligible():
            return False
        if not self._is_post_cover_upload_stuck(obs):
            return False
        self._goal_id = None
        self._goal_ne = self._goal_ne_for(None)
        self._last_sequence = []
        self._stuck_since_step = None
        return True

    def _solve_from_candidate_window(self) -> tuple[Optional[int], list[int]]:
        trig = dict(self.contract.slow_loop_triggers or {})
        window_k = int(trig.get("window_k", 12))
        prefetch_mult = _slow_loop_pos_int(trig.get("prefetch_k_multiplier", 3), default=3, minimum=1)
        horizon_h = int(trig.get("horizon_h", 4))
        path_samples = int(trig.get("path_samples", 9))
        phase = self._mission_phase()
        if phase == "balance":
            window_k = max(4, int(window_k) // 2)
        params = self._slow_params_with_struct()
        profile = self.contract.struct_profile

        use_legacy_mask = bool(self.contract.comm_path_quality_mask) and not bool(
            profile.use_struct_comm_profile
        )

        win = build_candidate_window_with_degrade(
            env=self.env,
            params=params,
            profile=profile,
            window_k=window_k,
            prefetch_k_multiplier=prefetch_mult,
            comm_objective=bool(self.contract.comm_objective),
            comm_path_quality_mask=use_legacy_mask,
            apply_energy_per_node_mask=bool(
                self.contract.energy_hard_constraint or self.contract.energy_budget_constraint
            ),
            path_samples=path_samples,
        )
        self._last_degrade_level = int(win.degrade_level)

        if len(win.poi_ids) == 0:
            fallback = nearest_energy_feasible_poi(env=self.env, params=params)
            if fallback is not None:
                return int(fallback), [int(fallback)]
            return None, []

        budget = float(self.env.remaining_energy) - float(self.env.cfg.energy_safe_margin)
        budget -= float(profile.energy_plan_margin_frac)
        budget = float(max(0.0, budget))

        res = solve_short_horizon_tour(
            window=win,
            horizon_h=horizon_h,
            lambda_q=float(params.lambda_q),
            mu_dist=float(params.mu_dist),
            beta_ret=float(params.beta_ret),
            comm_objective=bool(self.contract.comm_objective),
            energy_budget_constraint=bool(self.contract.energy_budget_constraint),
            energy_objective_weight=float(self.contract.energy_objective_weight),
            energy_budget=budget,
        )
        seq = list(res.sequence_poi_ids)
        gid = int(seq[0]) if len(seq) > 0 else None
        return gid, seq

    def _apply_goal_lock(
        self,
        *,
        obs: SlowObservation,
        new_gid: Optional[int],
        new_seq: list[int],
        periodic_only: bool,
    ) -> Tuple[Optional[int], list[int]]:
        if not bool(self.contract.enable_goal_lock):
            return new_gid, new_seq
        if not periodic_only:
            return new_gid, new_seq
        if self._goal_id is None:
            return new_gid, new_seq
        if self._goal_id in self.env.effective:
            return new_gid, new_seq
        ev = self._event_trigger_flags()
        if bool(ev.get("on_goal_spatial_complete", False)) and self._goal_id in self.env.covered:
            return new_gid, new_seq
        if self._goal_lock_stuck_timeout(obs):
            return new_gid, new_seq
        if new_gid is None:
            seq_keep = list(self._last_sequence) if self._last_sequence else [int(self._goal_id)]
            return self._goal_id, seq_keep
        if int(new_gid) == int(self._goal_id):
            return new_gid, new_seq
        self._future_sequence = list(new_seq)
        return self._goal_id, list(self._last_sequence) if self._last_sequence else [int(self._goal_id)]

    def step(self, obs: SlowObservation) -> SlowPlan:
        pol = str(self.contract.slow_policy or "").strip().lower()
        steps = int(obs.step)

        self._update_stuck_timer(obs)
        self._apply_upload_stuck_recovery(obs)
        self._sync_backlog_return_phase()

        if self._should_enter_return_phase():
            self._goal_id = None
            self._goal_ne = self._goal_ne_for(None)
            self._last_sequence = []

        should, periodic_only, _reason, _level = self._should_replan(pol=pol, steps=steps, obs=obs)
        if should:
            self._replan_count += 1
            self._last_replan_step = steps
            if self._should_enter_return_phase():
                new_gid, new_seq = None, []
            else:
                new_gid, new_seq = self._solve_from_candidate_window()
                new_gid, new_seq = self._apply_goal_lock(
                    obs=obs,
                    new_gid=new_gid,
                    new_seq=new_seq,
                    periodic_only=periodic_only,
                )
            self._last_sequence = list(new_seq)
            self._goal_id = new_gid
            self._goal_ne = self._goal_ne_for(new_gid)

        return SlowPlan(goal_id=self._goal_id, goal_ne=(float(self._goal_ne[0]), float(self._goal_ne[1])))
