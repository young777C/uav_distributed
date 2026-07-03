"""
Paper 2 §5.2: GCS-side probability-driven target selection.

Overrides Paper 1 SlowLoop with:
- Effective completion probability as selection objective (Eq. 19)
- VoI feedback integration (Eq. 28–30)
- Adaptive horizon (Eq. 31)
- Beam-search sequence optimization

Inherits candidate window construction, energy masks, and control-link
thresholds from Paper 1 (``build_candidate_window_with_degrade``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace

from uavlab.paper1.contracts.contract_config import Paper1ContractConfig
from uavlab.paper1.contracts.contract_types import SlowObservation, SlowPlan
from uavlab.paper1.loops.slow.planner import (
    SlowLoopParams,
    build_candidate_window_with_degrade,
    nearest_energy_feasible_poi,
    nearest_uncovered_poi,
)
from uavlab.paper1.loops.slow.slow_loop import SlowLoop as Paper1SlowLoop
from uavlab.paper1.sim.energy_model import energy_return_need_from_env, energy_need_from_env
from uavlab.paper1.sim.env import Paper1Env

from uavlab.paper2.contracts.prob_model import (
    gcs_coverage_success_prob,
    gcs_return_conditional_prob,
    gcs_completion_probability,
    estimated_tx_time,
    gcs_return_link_suitability,
    prediction_execution_discrepancy,
)
from uavlab.paper2.contracts.feedback_policy import (
    VoIFeedbackPolicy,
    FeedbackDecision,
    CriticalEventType,
    FeedbackMessage,
    apply_feedback_state_update,
)
from uavlab.paper2.contracts.adaptive_schedule import (
    AdaptiveSchedule,
    ScheduleState,
)

Point2D = Tuple[float, float]


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, float(x))))


@dataclass
class ProbCandidateScore:
    """Scoring results for one candidate POI during beam search."""

    poi_id: int
    prob: float                         # P̂^eff_{i,G}(t)
    coverage_prob: float                # P̂^cov_{i,G}(t)
    return_cond_prob: float             # P̂^ret|cov_{i,G}(t)
    dist: float
    energy_feasible: bool


def _compute_poi_probability(
    *,
    env: Paper1Env,
    poi_id: int,
    params: SlowLoopParams,
    contract: Paper1ContractConfig,
    estimated_uav_pos: Point2D,
    estimated_energy: float,
    estimated_backlog_bits: float,
) -> Optional[ProbCandidateScore]:
    """Compute effective completion probability for one POI candidate."""
    poi = env.pois[int(poi_id)]
    ppos = (float(poi.pos_ne[0]), float(poi.pos_ne[1]))
    key_bits = float(env.cfg.key_bits_per_poi)
    th = params.dual_link

    # Distance
    dist = math.hypot(estimated_uav_pos[0] - ppos[0], estimated_uav_pos[1] - ppos[1])
    d_max = float(env.cfg.n_max) - float(env.cfg.n_min)  # map extent for normalization
    nd = _clip01(dist / d_max)

    # Energy estimates
    fly_e = float(env.cfg.energy_per_meter) * dist
    obs_e = float(env.cfg.energy_hover_per_s) * float(params.t_obs_s + params.t_safe_s)
    home_e = float(env.cfg.energy_per_meter) * math.hypot(
        ppos[0] - float(env.cfg.gcs_ne[0]),
        ppos[1] - float(env.cfg.gcs_ne[1]),
    )
    safe_margin = float(env.cfg.energy_safe_margin)
    em = _clip01((float(estimated_energy) - (fly_e + obs_e + home_e + safe_margin)) / (float(estimated_energy) + 1e-12))

    # Path control quality — reuse paper1's link infrastructure
    from uavlab.paper1.comm.link_proxy import quality_from_loss
    from uavlab.paper1.comm.thresholds import link_proxy_at, control_link_ok

    # Path samples for control link quality
    from uavlab.paper1.loops.slow.planner import _q_path_min_sampled  # noqa

    path_q = _q_path_min_sampled(env=env, poi_ne=ppos, samples=9) if hasattr(env, 'pos_ne') else 1.0

    # Path risk (simple: based on nofly proximity)
    risk = 0.0
    for nf in getattr(env, 'nofly_zones', []):
        try:
            cx, cy = float(nf.cx), float(nf.cy)
            cr = float(nf.radius)
            d_to_nf = math.hypot(estimated_uav_pos[0] - cx, estimated_uav_pos[1] - cy)
            if d_to_nf < cr * 2:
                risk = max(risk, 1.0 - d_to_nf / (cr * 2))
        except (AttributeError, IndexError):
            risk += 0.0
    risk_max = 1.0

    # Boundary proximity penalty: keep 100m margin at all degradation levels.
    # At high degradation, aggressive INS near boundaries requires safe margins.
    boundary_margin = float(getattr(env.cfg, 'boundary_safety_margin_m', 100.0))
    n_min = float(env.cfg.n_min)
    n_max = float(env.cfg.n_max)
    px, py = float(ppos[0]), float(ppos[1])
    boundary_penalty = 0.0
    for coord, lo, hi in [(px, n_min, n_max), (py, n_min, n_max)]:
        d_lo = coord - lo
        d_hi = hi - coord
        margin_2x = boundary_margin * 2.0
        if d_lo < margin_2x:
            boundary_penalty = max(boundary_penalty, 1.0 - d_lo / margin_2x)
        if d_hi < margin_2x:
            boundary_penalty = max(boundary_penalty, 1.0 - d_hi / margin_2x)
    boundary_penalty = _clip01(boundary_penalty)
    # V5: Degradation-aware boundary penalty. At high degradation (loss_max ≥ 0.50),
    # boundary POIs are hard to reach — link is already bad everywhere, and SAFE
    # avoidance near boundaries wastes flight time. Stronger penalty deprioritizes
    # edge POIs in favour of interior POIs that the UAV can actually reach.
    loss_max_cfg = float(getattr(env.cfg, 'distance_loss_max', 0.40))
    boundary_weight = 2.5 if loss_max_cfg >= 0.50 else 0.7
    effective_risk = max(_clip01(risk / risk_max), boundary_penalty * boundary_weight)

    pcov = gcs_coverage_success_prob(
        norm_dist=nd, energy_margin=em, path_ctrl_quality=path_q, path_risk=effective_risk,
    )

    # P0 fix: Execution-layer calibrated return link evaluation.
    # The probability model evaluates static link quality at the POI position.
    # However, the execution layer (FSM RECOVER or V5 RECOVER) can actively move
    # the UAV to improve link position before transmitting.  This capability
    # is captured by a calibration factor that bridges the gap between static
    # link quality and observed return success rates:
    #
    #   EPA-Centralized (FSM):  R_fail|cov ≈ 0.05  → exec_cal ≈ 0.85
    #   V5 (probability-driven): R_fail|cov ≈ 0.20  → exec_cal ≈ 0.60
    #
    # The calibration formula: pret_cal = pret + exec_cal × (1 − pret)
    # This recovers exec_cal fraction of the gap between static estimate and
    # perfect transmission, matching observed execution-layer capability.
    link = link_proxy_at(env, ppos)
    poi_loss = float(link.loss_p)
    poi_delay = float(link.delay_s)
    poi_bw = float(link.bandwidth_bps)

    tx_time = estimated_tx_time(key_bits=key_bits, loss_p=poi_loss, delay_s=poi_delay, bandwidth_bps=poi_bw)
    loss_max = float(th.data_max_loss_p)
    t_max = float(th.data_max_return_time_s)
    # bw_max: actual max bandwidth at GCS (zero distance), scaled slightly for headroom
    gcs_link = link_proxy_at(env, (float(env.cfg.gcs_ne[0]), float(env.cfg.gcs_ne[1])))
    bw_max = float(gcs_link.bandwidth_bps) * 1.2
    # backlog_max: from config or default to 100 Mbit (~25 POIs of 4 Mbit each)
    backlog_max = float(getattr(env.cfg, 'key_bits_per_poi', 4_000_000)) * 25.0

    lm = _clip01(1.0 - poi_loss / max(loss_max, 1e-12))
    dm = _clip01(1.0 - tx_time / max(t_max, 1e-12))
    bm = _clip01(poi_bw / max(bw_max, 1e-12))
    bkm = _clip01(1.0 - float(estimated_backlog_bits) / max(backlog_max, 1e-12))

    pret_raw = gcs_return_conditional_prob(
        link_reliability_margin=lm, delay_margin=dm, bandwidth_margin=bm, backlog_margin=bkm,
    )
    # Execution-layer calibration: the UAV can actively move (FSM RECOVER /
    # V5 RECOVER) to improve link position after coverage.  The calibration
    # factor bridges static link quality to observed return success rates.
    # Default 0.70 works for both EPA-Cent (FSM: ~0.85 actual) and V5 (~0.60).
    # Execution-layer calibration: the UAV can actively move (FSM RECOVER /
    # V5 RECOVER) to improve link position after coverage.  The calibration
    # factor bridges static link quality to observed return success rates.
    # Degradation-adaptive: higher calibration for severe degradation to
    # prevent probability-floor collapse; lower for mild degradation to
    # preserve inter-candidate discrimination.
    loss_max_cfg = float(getattr(env.cfg, 'distance_loss_max', 0.40))
    if loss_max_cfg >= 0.50:
        exec_cal = 0.85   # High/Severe: prevent all-candidate floor collapse
    else:
        exec_cal = 0.70   # Low/Medium: preserve discrimination between candidates
    pret = pret_raw + exec_cal * (1.0 - pret_raw)
    prob = gcs_completion_probability(coverage_prob=pcov, return_cond_prob=pret)

    return ProbCandidateScore(
        poi_id=int(poi_id),
        prob=prob,
        coverage_prob=pcov,
        return_cond_prob=pret,
        dist=dist,
        energy_feasible=em > 0.01,
    )


def _beam_search_sequence(
    *,
    candidates: List[ProbCandidateScore],
    horizon_h: int,
    beam_width: int = 4,
    max_dist: float = 2500.0,
    poi_positions: Optional[Dict[int, Point2D]] = None,
    transition_weight: float = 0.06,
    # Low-probability fallback threshold: when top prob < this, use distance-priority
    low_prob_fallback_threshold: float = 0.08,
) -> List[int]:
    """Beam search for max-probability sequence (Paper 2 complexity: O(B·M·H)).

    Three factors in the objective (higher = better):
      1. P̂^eff(i)  — effective completion probability (primary)
      2. dist(i)    — exploration bonus for far POIs (secondary)
      3. Δ(i|prev)  — transition distance penalty (keeps tours efficient)

    The transition penalty is proportional to distance between consecutive
    POIs, mimicking the travel-cost term in CP-SAT.  Without it the beam
    search produces zigzag tours that waste travel time.

    Two regimes:
      [Normal] prob > threshold → beam search over P̂^eff with distance bonus.
      [Low-prob fallback] top prob ≤ threshold → pure distance ordering.
    """
    if not candidates:
        return []

    # Build a lookup for POI positions (if available)
    pos_map: Dict[int, Point2D] = dict(poi_positions or {})
    _dist_cache: Dict[Tuple[int, int], float] = {}

    def _poi_dist(ida: int, idb: int) -> float:
        key = (ida, idb) if ida < idb else (idb, ida)
        if key not in _dist_cache:
            pa = pos_map.get(ida)
            pb = pos_map.get(idb)
            if pa is not None and pb is not None:
                _dist_cache[key] = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
            else:
                _dist_cache[key] = 0.0
        return _dist_cache[key]

    H = min(int(horizon_h), len(candidates))
    B = min(int(beam_width), len(candidates))

    # Find top probability
    top_prob = max(sc.prob for sc in candidates)

    # ——— Low-probability fallback regime ———
    if top_prob <= float(low_prob_fallback_threshold):
        by_dist = sorted(candidates, key=lambda sc: sc.dist)
        return [sc.poi_id for sc in by_dist[:H]]

    # ——— Normal regime ———
    if top_prob < 0.20:
        alpha = 1.0
    else:
        alpha = 0.6

    def _explore_score(sc):
        """Pure probability score (no distance bonus — that would hurt tour efficiency)."""
        return sc.prob

    def _first_poi_score(sc):
        """Score for the FIRST POI: probability minus transit cost from current pos.

        This replaces the old distance-exploration bonus.  Nearby POIs with
        similar probability are preferred because they cost less to reach,
        producing more efficient tours.
        """
        return sc.prob - float(transition_weight) * sc.dist / max(max_dist, 1.0)

    def _transition_penalty(ida: int, idb: int) -> float:
        """Penalty for jumping between two POIs, normalised to [0, transition_weight]."""
        d = _poi_dist(ida, idb)
        return float(transition_weight) * d / max(max_dist, 1.0)

    # Sort candidates by first-POI score (probability minus transit cost)
    sorted_cands = sorted(candidates, key=_first_poi_score, reverse=True)

    # Beam: list of (sequence, total_score)
    beam: List[Tuple[List[int], float]] = []
    for sc in sorted_cands[:B]:
        beam.append(([sc.poi_id], _first_poi_score(sc)))

    for step in range(1, H):
        new_beam: List[Tuple[List[int], float]] = []
        for seq, score in beam:
            prev_id = seq[-1]
            used = set(seq)
            for sc in sorted_cands:
                if sc.poi_id in used:
                    continue
                new_seq = seq + [sc.poi_id]
                discount = 1.0 / (1.0 + step * 0.1)
                travel_penalty = _transition_penalty(prev_id, sc.poi_id)
                new_score = score + _explore_score(sc) * discount - travel_penalty
                new_beam.append((new_seq, new_score))
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:B]

    if not beam:
        return [sc.poi_id for sc in sorted_cands[:H]]
    return beam[0][0]


@dataclass
class Paper2SlowLoop:
    """Probability-driven slow loop (§5.2).

    Extends Paper1 SlowLoop: uses P̂^eff_{i,G}(t) for target selection.
    Integrates VoI feedback and adaptive horizon.
    """

    env: Paper1Env
    params: SlowLoopParams
    contract: Paper1ContractConfig
    slow_interval_steps: int

    # Paper 1 slow loop (delegate for common functionality)
    _p1: Paper1SlowLoop = field(init=False)

    # Paper 2 modules
    feedback_policy: VoIFeedbackPolicy = field(default_factory=VoIFeedbackPolicy)
    adaptive_schedule: AdaptiveSchedule = field(default_factory=AdaptiveSchedule)

    # Internal state
    _goal_id: Optional[int] = None
    _goal_ne: Point2D = (0.0, 0.0)
    _last_replan_step: int = -10**9
    _last_sequence: List[int] = field(default_factory=list)
    _replan_count: int = 0

    # Adaptive schedule state
    _current_schedule: ScheduleState = field(default_factory=ScheduleState)

    # VoI state
    _last_gcs_prob: float = 0.0
    _last_uav_prob: float = 0.0
    _last_feedback_decision: Optional[FeedbackDecision] = None

    def __post_init__(self) -> None:
        self._p1 = Paper1SlowLoop(
            env=self.env,
            params=self.params,
            contract=self.contract,
            slow_interval_steps=int(self.slow_interval_steps),
        )
        self._goal_id = None
        self._goal_ne = (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
        # Initialize schedule with contract horizon (from config)
        trig = dict(getattr(self.contract, 'slow_loop_triggers', {}) or {})
        cfg_h = int(trig.get("horizon_h", 4))
        if cfg_h > 0:
            self.adaptive_schedule.state.horizon_h = cfg_h
        self._current_schedule = self.adaptive_schedule.state

    def reset(self) -> None:
        self._p1.reset()
        self._goal_id = None
        self._goal_ne = (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
        self._last_replan_step = -10**9
        self._last_sequence.clear()
        self._replan_count = 0
        self.feedback_policy.reset()
        self.adaptive_schedule.reset()
        self._current_schedule = self.adaptive_schedule.state
        self._last_gcs_prob = 0.0
        self._last_uav_prob = 0.0

    @property
    def replan_count(self) -> int:
        return int(self._replan_count)

    @property
    def current_horizon(self) -> int:
        return int(self._current_schedule.horizon_h)

    # ——— Core override: probability-driven target selection ———


    def _solve_beam_search(self) -> tuple[Optional[int], List[int]]:
        """Replace Paper1's CP-SAT solver with beam search over P̂^eff (§5.2, Eq. 19).

        Uses simple distance-prefetch for candidate building (matches EPA-Centralized).
        """
        H = int(self._current_schedule.horizon_h)

        # Simple distance-prefetch candidate building (matches EPA-Centralized).
        # Filter out effective POIs (covered + data returned) only.
        done = set(self.env.effective)
        cand: list[tuple[float, int]] = []
        for poi in self.env.pois:
            pid = int(poi.poi_id)
            if pid in done:
                continue
            d = math.hypot(
                float(self.env.pos_ne[0]) - float(poi.pos_ne[0]),
                float(self.env.pos_ne[1]) - float(poi.pos_ne[1]),
            )
            cand.append((d, pid))
        cand.sort(key=lambda x: x[0])
        # Take up to 2× horizon candidates (matches EPA-Centralized)
        prefetch_n = min(max(2 * H, 12), len(cand))
        candidate_ids = [pid for _, pid in cand[:prefetch_n]]

        if not candidate_ids:
            fallback = nearest_energy_feasible_poi(env=self.env, params=self.params)
            if fallback is not None:
                return int(fallback), [int(fallback)]
            fb2 = nearest_uncovered_poi(env=self.env)
            if fb2 is not None:
                return int(fb2), [int(fb2)]
            return None, []

        # Compute probability for each candidate
        uav_pos = (float(self.env.pos_ne[0]), float(self.env.pos_ne[1]))
        est_energy = float(self.env.remaining_energy)
        backlog = float(getattr(self.env, 'backlog_bits', 0.0))

        scores: List[ProbCandidateScore] = []
        for pid in candidate_ids:
            sc = _compute_poi_probability(
                env=self.env,
                poi_id=int(pid),
                params=self.params,
                contract=self.contract,
                estimated_uav_pos=uav_pos,
                estimated_energy=est_energy,
                estimated_backlog_bits=backlog,
            )
            if sc is not None:
                scores.append(sc)

        if not scores:
            fb = nearest_energy_feasible_poi(env=self.env, params=self.params)
            if fb is not None:
                return int(fb), [int(fb)]
            return None, []

        # Build POI position map for transition-aware beam search
        poi_positions: Dict[int, Point2D] = {
            pid: (float(self.env.pois[int(pid)].pos_ne[0]),
                  float(self.env.pois[int(pid)].pos_ne[1]))
            for pid in candidate_ids
        } if hasattr(self.env, 'pois') else {}

        # Beam search with transition penalty for tour efficiency
        horizon = int(self._current_schedule.horizon_h)
        map_extent = float(self.env.cfg.n_max) - float(self.env.cfg.n_min)
        seq = _beam_search_sequence(
            candidates=scores,
            horizon_h=horizon,
            max_dist=map_extent,
            poi_positions=poi_positions,
            transition_weight=0.06,
        )

        if not seq:
            return None, []

        return seq[0], seq

    # ——— VoI feedback integration ———


    def evaluate_feedback(
        self,
        *,
        step: int,
        uav_probability: Optional[float] = None,
        critical_events: Optional[List[CriticalEventType]] = None,
    ) -> FeedbackDecision:
        """Evaluate whether to send feedback based on D_pe(t) and events (Eq. 28).

        Args:
            step: current simulation step
            uav_probability: P̂^eff_{g,U}(t) from UAV fast loop (or None if unavailable)
            critical_events: E^crit_t events detected by UAV
        """
        gcs_prob = self._last_gcs_prob
        uav_prob = uav_probability if uav_probability is not None else self._last_uav_prob

        if critical_events is None:
            critical_events = []

        decision = self.feedback_policy.evaluate(
            step=step,
            gcs_probability_estimate=gcs_prob,
            uav_probability_evaluation=uav_prob,
            current_critical_events=critical_events,
        )
        self._last_feedback_decision = decision
        return decision

    # ——— Sequence tracking ———

    def _advance_in_sequence(self) -> Optional[int]:
        """If the current goal is **covered** (data queued for background return),
        return the *next* POI from the planned sequence *without* re-solving.

        Advances on ``covered`` rather than ``effective``: data return continues
        during transit to the next POI via the return queue, eliminating the
        per-POI hover penalty under C3 intermittent links.
        """
        if self._goal_id is None or not self._last_sequence:
            return None
        if self._goal_id in self.env.covered:
            try:
                idx = self._last_sequence.index(self._goal_id)
                if idx + 1 < len(self._last_sequence):
                    return int(self._last_sequence[idx + 1])
            except ValueError:
                pass
        return None

    # ——— Main step ———


    def step(self, obs: SlowObservation) -> SlowPlan:
        """One slow-loop step with probability-driven selection + VoI feedback.

        Hybrid advancement: coverage-based when return queue is healthy (< 5 pending),
        effective-only when queue is backed up (let data return catch up).
        Boundary penalties prevent selection of near-edge POIs.
        """
        steps = int(obs.step)
        pol = str(self.contract.slow_policy or "").strip().lower()

        # Delegate stuck recovery to Paper1 (but NOT return phase — FSM handles RTH)
        self._p1._update_stuck_timer(obs)
        self._p1._apply_upload_stuck_recovery(obs)
        self._p1._sync_backlog_return_phase()

        # Replan decision: Paper1 event/periodic triggers
        should, periodic_only, reason, level = self._p1._should_replan(pol=pol, steps=steps, obs=obs)

        # Consume stale VoI feedback decision
        if self._last_feedback_decision is not None:
            self._last_feedback_decision = None

        # ── Hybrid advancement: check return queue pressure ──
        queue_pending = getattr(self.env, 'return_queue', None)
        pending_count = int(queue_pending.pending_count) if queue_pending is not None else 0
        queue_backed_up = pending_count >= 3  # best balance: R_task=0.55, R_fail=0.14

        if should:
            self._replan_count += 1
            self._last_replan_step = steps

            if (self._goal_id is not None
                  and self._goal_id not in self.env.covered):
                # Goal not yet covered: persist (keep flying).
                new_gid, new_seq = self._goal_id, self._last_sequence
            elif queue_backed_up:
                # Queue pressure: use beam-search natural hold.
                # The search re-picks the current POI (distance≈0 → highest prob)
                # until it becomes effective, giving data return time to catch up.
                new_gid, new_seq = self._solve_beam_search()
            else:
                # Queue healthy: coverage-based advance to next POI in sequence.
                next_gid = self._advance_in_sequence()
                if next_gid is not None:
                    new_gid, new_seq = next_gid, self._last_sequence
                else:
                    new_gid, new_seq = self._solve_beam_search()

            self._last_sequence = list(new_seq)
            self._goal_id = new_gid
            self._goal_ne = self._goal_ne_for(new_gid)

            # Update GCS probability estimate using UAV-consistent action-conditional
            # formula for the current goal (same semantics as fast-loop evaluation).
            if new_gid is not None:
                from uavlab.paper2.contracts.prob_model import uav_action_conditional_probability, estimated_tx_time
                from uavlab.paper1.comm.thresholds import link_proxy_at
                uav_pos = (float(self.env.pos_ne[0]), float(self.env.pos_ne[1]))
                goal_pos = self._goal_ne_for(new_gid)
                obs_radius = float(self.env.cfg.poi_cover_radius) if hasattr(self.env.cfg, 'poi_cover_radius') else 15.0
                d_to_goal = math.hypot(uav_pos[0] - goal_pos[0], uav_pos[1] - goal_pos[1])
                obs_suit = 1.0 if d_to_goal <= obs_radius else max(0.0, 1.0 - (d_to_goal - obs_radius) / 500.0)
                home_need = float(energy_return_need_from_env(self.env))
                safe_margin = float(self.env.cfg.energy_safe_margin)
                em_gcs = max(0.0, (float(self.env.remaining_energy) - home_need - safe_margin)
                             / (float(self.env.remaining_energy) + 1e-12))
                loss_p = float(getattr(self.env, 'link_loss_p', 0.0))
                ctrl_q = max(0.0, 1.0 - loss_p)
                th = self.contract.dual_link
                tx_time = estimated_tx_time(key_bits=float(self.env.cfg.key_bits_per_poi),
                    loss_p=loss_p, delay_s=float(self.env.cfg.delay_mean_s), bandwidth_bps=1_000_000.0)
                lm = max(0.0, 1.0 - loss_p / max(float(th.data_max_loss_p), 1e-12))
                dm = max(0.0, 1.0 - tx_time / max(float(th.data_max_return_time_s), 1e-12))
                bm = max(0.0, 1_000_000.0 / max(float(link_proxy_at(self.env, (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))).bandwidth_bps) * 1.2, 1e-12))
                bkm = max(0.0, 1.0 - float(getattr(self.env, 'backlog_bits', 0.0)) / max(float(self.env.cfg.key_bits_per_poi) * 25.0, 1e-12))
                self._last_gcs_prob = uav_action_conditional_probability(
                    goal_already_covered=bool(new_gid in self.env.covered),
                    goal_already_returned=bool(new_gid in self.env.returned),
                    obs_suitability=obs_suit, energy_margin=em_gcs,
                    ctrl_quality=ctrl_q, safety_margin=1.0,
                    link_reliability_margin=lm, delay_margin=dm,
                    bandwidth_margin=bm, backlog_margin=bkm,
                )

        return SlowPlan(goal_id=self._goal_id, goal_ne=(float(self._goal_ne[0]), float(self._goal_ne[1])))

    def _goal_ne_for(self, gid: Optional[int]) -> Point2D:
        if gid is None:
            return (float(self.env.cfg.gcs_ne[0]), float(self.env.cfg.gcs_ne[1]))
        p = self.env.pois[int(gid)].pos_ne
        return (float(p[0]), float(p[1]))
