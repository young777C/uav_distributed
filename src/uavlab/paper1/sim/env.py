from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from uavlab.paper1.sim.config import Paper1SimConfig
from uavlab.paper1.types import CommState, EpisodeLog, FastCommState, FastState, Poi, SlowGoal
from uavlab.related_models.comm_link_state import LinkState
from uavlab.paper1.sim.return_queue import ReturnProgressResult, ReturnQueue
from uavlab.related_models.key_data_return import ReturnDecisionParams
from uavlab.paper1.metrics.paper_metrics import compute_paper_episode_metrics
from uavlab.paper1.sim.energy_model import deduct_step_energy
from uavlab.scene.geometry import in_nofly as _geom_in_nofly
from uavlab.scene.loader import SceneConfig


Point2D = Tuple[float, float]


def _dist(a: Point2D, b: Point2D) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))

def _seg_intersects_circle(*, a: Point2D, b: Point2D, c: Point2D, r: float) -> bool:
    """
    Return True if segment a->b intersects or touches circle centered at c with radius r.
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    rr = float(r)
    # if an endpoint is inside, it intersects
    if (ax - cx) ** 2 + (ay - cy) ** 2 <= rr ** 2:
        return True
    if (bx - cx) ** 2 + (by - cy) ** 2 <= rr ** 2:
        return True
    dx, dy = bx - ax, by - ay
    denom = float(dx * dx + dy * dy)
    if denom <= 1e-12:
        return False
    t = ((cx - ax) * dx + (cy - ay) * dy) / denom
    t = float(min(1.0, max(0.0, t)))
    px, py = ax + t * dx, ay + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2 <= rr ** 2


@dataclass
class Paper1Env:
    """
    Lightweight task-level environment for Paper 1.

    This is NOT a Gym env; it is a minimal simulator used by slow/fast loops.
    """

    cfg: Paper1SimConfig
    pois: List[Poi]
    blackholes: List[Tuple[float, float, float]]
    scene_geom: SceneConfig

    # dynamic state
    t: int = 0
    pos_ne: Point2D = (0.0, 0.0)
    vel_ne: Point2D = (0.0, 0.0)
    yaw_rad: float = 0.0
    remaining_energy: float = 1.0
    comm_mode: FastCommState = FastCommState.INS
    rng: random.Random = None  # type: ignore[assignment]
    covered: set[int] = None  # type: ignore[assignment]  # c_i: spatial observation complete
    returned: set[int] = None  # type: ignore[assignment]  # r_i: key data returned (e_i = c_i * r_i)
    effective: set[int] = None  # type: ignore[assignment]  # mirror of ``returned`` for legacy callers
    return_queue: ReturnQueue = None  # type: ignore[assignment]
    backlog_bits: float = 0.0
    poi_dwell_steps: List[int] = None  # type: ignore[assignment]
    poi_entered: List[bool] = None  # type: ignore[assignment]
    collision_count: int = 0
    terminated_by_collision: bool = False
    terminated_by_oob: bool = False
    terminated_by_energy_depleted: bool = False
    terminated_by_returned_home: bool = False
    nofly_dwell_s: float = 0.0
    _prev_in_nofly: bool = False
    nofly_entry_count: int = 0
    # Paper Eq. (61): wall time (s) when c_i=1 / when r_i=1 first holds
    poi_cov_time_s: Dict[int, float] = None  # type: ignore[assignment]
    poi_return_time_s: Dict[int, float] = None  # type: ignore[assignment]

    def _dt_s(self) -> float:
        return 1.0 / float(max(1, int(self.cfg.step_hz)))

    def reset(self, *, seed: int = 0) -> FastState:
        self.rng = random.Random(int(seed))
        self.t = 0
        self.pos_ne = self.cfg.gcs_ne
        self.vel_ne = (0.0, 0.0)
        self.yaw_rad = 0.0
        self.remaining_energy = 1.0
        self.comm_mode = FastCommState.INS
        self.covered = set()
        self.returned = set()
        self.effective = set()
        self.return_queue = ReturnQueue()
        self.backlog_bits = 0.0
        self.poi_dwell_steps = [0 for _ in range(len(self.pois))]
        self.poi_entered = [False for _ in range(len(self.pois))]
        self.collision_count = 0
        self.terminated_by_collision = False
        self.terminated_by_oob = False
        self.terminated_by_energy_depleted = False
        self.terminated_by_returned_home = False
        self.nofly_dwell_s = 0.0
        self._prev_in_nofly = False
        self.nofly_entry_count = 0
        self.poi_cov_time_s = {}
        self.poi_return_time_s = {}
        return self.fast_state()

    def fast_state(self) -> FastState:
        link = self.observe_link_state()
        comm = CommState(
            loss_p=float(link.loss_p),
            delay_s=float(getattr(link, "delay_s", 0.0)),
            bandwidth_bps=float(getattr(link, "bandwidth_bps", 0.0)),
        )
        return FastState(
            p_ne=self.pos_ne,
            v_ne=self.vel_ne,
            yaw_rad=self.yaw_rad,
            comm=comm,
            energy_loc=float(self.remaining_energy),
            mode=self.comm_mode,
        )

    def slow_goal(self, poi_id: Optional[int]) -> SlowGoal:
        return SlowGoal(poi_id=poi_id, slow_mode="default")

    # ---- Link state helpers (paper uses c_t = (ℓ, τ, b); we reuse current CommChannel in legacy env later) ----
    def in_blackhole(self, p: Point2D) -> bool:
        if not self.cfg.use_scene_blackholes:
            return False
        x, y = float(p[0]), float(p[1])
        for cx, cy, r in self.blackholes:
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                return True
        return False

    def observe_link_state(self) -> LinkState:
        """
        Bridge to paper link state: we re-use related_models.compute_link_state later.
        For now return a placeholder that is compatible with key-return model.
        """
        # Paper1Lite link proxy:
        # - if enable_distance_decay: piecewise-linear loss over distance (d0..d1, min..max)
        # - else: constant base_loss
        # - optional blackhole bump when use_scene_blackholes is enabled
        # - optional Gaussian fading (loss_jitter_sigma) driven by env.reset(seed=...)
        x, y = float(self.pos_ne[0]), float(self.pos_ne[1])
        gx, gy = float(self.cfg.gcs_ne[0]), float(self.cfg.gcs_ne[1])
        d = float(math.hypot(x - gx, y - gy))

        if bool(self.cfg.enable_distance_decay):
            d0 = float(self.cfg.distance_d0_m)
            d1 = float(self.cfg.distance_d1_m)
            lmin = float(self.cfg.distance_loss_min)
            lmax = float(self.cfg.distance_loss_max)
            if d <= d0:
                loss = lmin
            elif d >= d1:
                loss = lmax
            else:
                t = (d - d0) / max(1e-9, (d1 - d0))
                loss = lmin + (lmax - lmin) * float(t)
        else:
            loss = float(self.cfg.base_loss)

        if self.in_blackhole(self.pos_ne):
            loss = float(loss) + float(self.cfg.blackhole_extra_loss)

        sigma = float(self.cfg.loss_jitter_sigma)
        if sigma > 0:
            r = self.rng if self.rng is not None else random
            loss = float(loss) + float(r.gauss(0.0, sigma))

        loss = float(min(max(loss, 0.0), 0.99))

        # delay and bandwidth are lightweight placeholders used by key-return model
        r2 = self.rng if self.rng is not None else random
        delay = float(self.cfg.delay_mean_s)
        dj = float(self.cfg.delay_jitter_s)
        if dj > 0:
            delay = float(max(0.0, delay + r2.gauss(0.0, dj)))
        bw = 1_000_000.0 * (1.0 - loss)
        return LinkState(loss_p=loss, delay_s=delay, bandwidth_bps=bw)

    def in_nofly(self, p: Point2D) -> bool:
        """True if ``p`` lies in a scene no-fly circle/rect (includes boundary fence when configured)."""

        return bool(_geom_in_nofly(float(p[0]), float(p[1]), self.scene_geom))

    def in_obstacle(self, p: Point2D) -> bool:
        """Backward-compatible alias: spatial risk is modeled as no-fly, not solid obstacles."""

        return self.in_nofly(p)

    def _inside_any_poi_cover(self) -> bool:
        """
        True if inside an **uncovered** POI cover disk (hover + observation energy, doc §4).

        Already-covered disks must not force ``vx=vy=0``; otherwise a large ``visit_radius_m``
        traps the UAV after the first POI and blocks travel to the next slow-loop goal.
        """

        for poi in self.pois:
            pid = int(poi.poi_id)
            if pid in self.covered:
                continue
            if _dist(self.pos_ne, poi.pos_ne) <= float(poi.cover_radius_m):
                return True
        return False

    def _nominal_pursuit_speed_mps(self, dist_to_target_m: float) -> float:
        """
        Speed toward a waypoint: up to ``v_xy_cruise`` (average cruise), ramping down in an approach zone.

        ``v_xy_max`` is reserved for explicit ``vel_ne_cmd`` (e.g. PID); default pursuit does not use it.
        """

        vc = float(self.cfg.v_xy_cruise)
        r = float(max(self.cfg.approach_slowdown_radius_m, 2.0 * float(self.cfg.visit_radius_m)))
        d = float(max(0.0, dist_to_target_m))
        if d >= r:
            return vc
        return float(vc * (d / r))

    def step_fast(
        self,
        *,
        target_ne: Point2D,
        dt: float,
        vel_ne_cmd: Point2D | None = None,
        approach_goal_ne: Point2D | None = None,
    ) -> None:
        """
        Advance a single fast-loop step.

        - If ``vel_ne_cmd`` is provided: clip to ``v_xy_max`` (urgent / controller command).
        - Else: steer toward ``target_ne`` (local lookahead / waypoint); approach slowdown uses
          distance to ``approach_goal_ne`` when set (slow-loop POI / GCS), else ``target_ne``.
        """
        prev_pos = self.pos_ne
        px, py = prev_pos
        if vel_ne_cmd is not None:
            vx, vy = float(vel_ne_cmd[0]), float(vel_ne_cmd[1])
            spd = float(math.hypot(vx, vy))
            vmax = float(self.cfg.v_xy_max)
            if spd > vmax and spd > 1e-9:
                s = vmax / spd
                vx, vy = vx * s, vy * s
        else:
            tx, ty = float(target_ne[0]), float(target_ne[1])
            dx, dy = tx - px, ty - py
            dist_steer = float(math.hypot(dx, dy))
            if dist_steer < 1e-6:
                vx, vy = 0.0, 0.0
            else:
                if approach_goal_ne is not None:
                    gx, gy = float(approach_goal_ne[0]), float(approach_goal_ne[1])
                else:
                    gx, gy = tx, ty
                dist_slow = float(math.hypot(gx - px, gy - py))
                spd = self._nominal_pursuit_speed_mps(dist_slow)
                vx = spd * dx / dist_steer
                vy = spd * dy / dist_steer

        # POI dwell: hold inside the cover disk so ``poi_dwell_steps`` can reach ``poi_dwell_s``.
        if self._inside_any_poi_cover():
            vx, vy = 0.0, 0.0

        self.vel_ne = (vx, vy)
        self.pos_ne = (px + vx * dt, py + vy * dt)
        self.yaw_rad = float(math.atan2(vy, vx)) if (abs(vx) + abs(vy)) > 1e-9 else self.yaw_rad

        # Energy (UAV_GCS doc §3–§7): flight by distance when speed >= threshold; else hover by dt.
        traveled = float(math.hypot(vx * dt, vy * dt))
        speed_mps = float(math.hypot(vx, vy))
        self.remaining_energy = deduct_step_energy(
            remaining_energy=float(self.remaining_energy),
            speed_mps=speed_mps,
            dt=float(dt),
            traveled_m=traveled,
            energy_per_meter=float(self.cfg.energy_per_meter),
            energy_hover_per_s=float(self.cfg.energy_hover_per_s),
            hover_speed_threshold_mps=float(self.cfg.hover_speed_threshold_mps),
        )

        # energy depletion termination
        if bool(self.cfg.terminate_on_energy_depleted) and float(self.remaining_energy) <= float(
            self.cfg.energy_depleted_threshold
        ):
            self.terminated_by_energy_depleted = True

        inside_nf = self.in_nofly(self.pos_ne)
        if inside_nf:
            self.nofly_dwell_s = float(self.nofly_dwell_s) + float(dt)
            if not self._prev_in_nofly:
                self.nofly_entry_count += 1
        self._prev_in_nofly = bool(inside_nf)

        # out-of-bounds termination (with optional buffer)
        if bool(self.cfg.terminate_on_oob):
            bb = float(self.cfg.boundary_buffer_m)
            if not (
                (self.cfg.n_min + bb) <= float(self.pos_ne[0]) <= (self.cfg.n_max - bb)
                and (self.cfg.e_min + bb) <= float(self.pos_ne[1]) <= (self.cfg.e_max - bb)
            ):
                self.terminated_by_oob = True

        # POI coverage update:
        # require (1) segment enters/touches the POI circle at least once, AND
        #        (2) cumulative dwell within the circle reaches cfg.poi_dwell_s seconds.
        req_steps = int(max(0, math.ceil(float(self.cfg.poi_dwell_s) / max(1e-9, float(dt)))))
        for poi in self.pois:
            if poi.poi_id in self.covered:
                continue
            pid = int(poi.poi_id)
            if 0 <= pid < len(self.poi_entered):
                if _seg_intersects_circle(a=prev_pos, b=self.pos_ne, c=poi.pos_ne, r=float(poi.cover_radius_m)):
                    self.poi_entered[pid] = True

                inside = _dist(self.pos_ne, poi.pos_ne) <= float(poi.cover_radius_m)
                if inside:
                    self.poi_dwell_steps[pid] += 1
                else:
                    self.poi_dwell_steps[pid] = 0

                if bool(self.poi_entered[pid]) and (req_steps <= 0 or self.poi_dwell_steps[pid] >= req_steps):
                    if pid not in self.covered:
                        self.poi_cov_time_s[int(pid)] = float(self.t) * self._dt_s()
                    self.covered.add(pid)
                    bits = float(poi.key_bits)
                    if pid not in self.returned:
                        t_cov = float(self.poi_cov_time_s.get(int(pid), float(self.t) * self._dt_s()))
                        self.return_queue.enqueue(
                            poi_id=int(pid),
                            bits=bits,
                            covered_time_s=t_cov,
                        )
                        self._sync_backlog_from_queue()

        self.t += 1

    def _sync_effective_from_returned(self) -> None:
        """``effective`` mirrors ``returned`` (e_i = c_i * r_i with returned ⊆ covered)."""

        self.effective = set(self.returned)

    @property
    def poi_pending_bits(self) -> Dict[int, float]:
        """Legacy view: POI id → remaining bits (for diagnostics / tests)."""

        q = self.return_queue
        if q is None:
            return {}
        return dict(q.remaining_bits_by_poi())

    def _sync_backlog_from_queue(self) -> None:
        self.backlog_bits = float(self.return_queue.pending_bits) if self.return_queue else 0.0

    def progress_key_return(
        self,
        *,
        dt_s: float | None = None,
        params: ReturnDecisionParams | None = None,
    ) -> ReturnProgressResult:
        """
        Incrementally return pending key data (per-POI FCFS, ``b_eff * dt`` per step).

        Replaces bulk ``return_success`` on the full backlog sum.
        """
        if self.return_queue is None or self.return_queue.pending_count <= 0:
            return ReturnProgressResult()

        dt = float(self._dt_s() if dt_s is None else dt_s)
        t_now = float(self.t) * self._dt_s()
        link = self.observe_link_state()
        result = self.return_queue.progress_step(
            link=link,
            dt_s=dt,
            now_s=t_now,
            params=params,
        )
        for pid in result.completed_ids:
            ip = int(pid)
            self.returned.add(ip)
            if ip not in self.poi_return_time_s:
                ret_t = self.return_queue.returned_time_s_for(ip)
                self.poi_return_time_s[ip] = float(ret_t if ret_t is not None else t_now)
        self._sync_backlog_from_queue()
        self._sync_effective_from_returned()
        return result

    def try_return_key(self, *, params: ReturnDecisionParams | None = None) -> ReturnProgressResult:
        """Backward-compatible alias for one progress step."""

        return self.progress_key_return(params=params)

    def done(self) -> bool:
        if self.terminated_by_collision:
            return True
        if self.terminated_by_oob:
            return True
        if self.terminated_by_energy_depleted:
            return True
        if self.terminated_by_returned_home:
            return True
        if self.t >= int(self.cfg.episode_steps):
            return True
        return False

    def metrics(
        self,
        *,
        link_recovery_latencies_s: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        self._sync_effective_from_returned()
        dist_home = _dist(self.pos_ne, self.cfg.gcs_ne)
        return_radius_m = float(self.cfg.return_radius_m)
        returned_home = bool(dist_home <= return_radius_m)

        # success return (optional termination mode)
        if (
            returned_home
            and bool(self.cfg.terminate_on_returned_home)
            and self.t > 0
            and (not self.terminated_by_collision)
            and (not self.terminated_by_oob)
            and (not self.terminated_by_energy_depleted)
        ):
            if float(self.backlog_bits) <= 0.0:
                self.terminated_by_returned_home = True

        if self.terminated_by_collision:
            reason = "collision"
        elif self.terminated_by_oob:
            reason = "oob"
        elif self.terminated_by_energy_depleted:
            reason = "energy_depleted"
        elif self.terminated_by_returned_home:
            reason = "returned_home"
        elif self.t >= int(self.cfg.episode_steps):
            reason = "time_limit"
        else:
            reason = "running"

        paper = compute_paper_episode_metrics(
            n_pois=len(self.pois),
            covered=set(self.covered),
            returned=set(self.returned),
            poi_cov_time_s=dict(self.poi_cov_time_s),
            poi_return_time_s=dict(self.poi_return_time_s),
            nofly_dwell_s=float(self.nofly_dwell_s),
            link_recovery_latencies_s=link_recovery_latencies_s,
            nofly_entry_count=int(self.nofly_entry_count),
        )
        return {
            "steps": int(self.t),
            **paper,
            "remaining_energy": float(self.remaining_energy),
            "in_blackhole": bool(self.in_blackhole(self.pos_ne)),
            "in_nofly": bool(self.in_nofly(self.pos_ne)),
            "dist_to_home_m": float(dist_home),
            "returned_home": bool(returned_home),
            "returned_ids": sorted(int(x) for x in self.returned),
            "covered_ids": sorted(int(x) for x in self.covered),
            "effective_ids": sorted(int(x) for x in self.effective),
            "collision_count": int(self.collision_count),
            "terminated_by_collision": bool(self.terminated_by_collision),
            "terminated_by_oob": bool(self.terminated_by_oob),
            "terminated_by_energy_depleted": bool(self.terminated_by_energy_depleted),
            "terminated_by_returned_home": bool(self.terminated_by_returned_home),
            "termination_reason": str(reason),
        }


def build_env(cfg: Paper1SimConfig, scene_geom: SceneConfig) -> Paper1Env:
    bh = list(getattr(scene_geom, "communication_blackholes", None) or [])
    blackholes = [(float(x), float(y), float(r)) for x, y, r in bh]
    pois: List[Poi] = []
    for i, p in enumerate(cfg.poi_list):
        pois.append(
            Poi(
                poi_id=i,
                pos_ne=(float(p[0]), float(p[1])),
                cover_radius_m=float(cfg.visit_radius_m),
                key_bits=float(cfg.key_bits_per_poi),
            )
        )
    return Paper1Env(cfg=cfg, pois=pois, blackholes=blackholes, scene_geom=scene_geom)

