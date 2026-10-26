"""Scene management — actor lifecycle, spawning, and cleanup.

Composes each episode as: 1 language-specified TARGET + N distractors, with
>=2 distractors visually similar to the target (design §5.1/§6.3), co-located on
the same road segment so they are actually co-visible from the UAV. Supports
multiple target classes (car / motorcycle / scooter / bicycle / pedestrian).
"""

from __future__ import annotations

import os
import math
import random
import time
from typing import Any

import carla
import numpy as np


def _tm_port() -> int:
    """TrafficManager port. Distinct per parallel worker (set via CARLA_TM_PORT) so
    concurrent generators on the same host don't share/clobber one TM on port 8000."""
    return int(os.environ.get("CARLA_TM_PORT", "8000"))

from carla_uav.drone import KinematicDrone
from carla_uav.expert_policy import ExpertPolicy
from carla_uav.safety import SafetyMonitor
from targets.maneuver_injector import ManeuverConfig, ManeuverInjector
from targets.loss_event_scheduler import LossEventScheduler
from targets.vehicle_target import PedestrianTarget, VehicleTarget
from scene.language_generator import LanguageGenerator, VEHICLE_CLASSES, COLOR_RGB


class SceneManager:
    """Manages the lifecycle of all CARLA actors for one tracking episode."""

    def __init__(
        self,
        world: carla.World,
        config: dict[str, Any],
        carla_client: carla.Client,
    ):
        self._world = world
        self._config = config
        self._client = carla_client

        self.drone: KinematicDrone | None = None
        self.expert: ExpertPolicy | None = None
        self.safety: SafetyMonitor | None = None
        self._target: VehicleTarget | PedestrianTarget | None = None
        self._target_type: str = ""
        self._target_class: str = "car"
        self._maneuver_injector: ManeuverInjector | None = None
        self._distractors: list = []
        self._distractor_meta: list[dict] = []
        self._retarget_log: list[dict] = []    # 1003_dynamic-retarget-design.md §2.2: [{tick, slot}, ...]
        self._ambient_walkers: list = []   # background pedestrians (manual control)
        self._perturbation_injector = None
        self._obstacle_avoider = None
        self._language_gen = None
        self.episode_language: str = ""
        self.target_meta: dict = {}
        self.target_plan: dict = {}
        self._tier_alt = [24.0, 32.0]
        self._distance_tier = "suitable"
        self._map = world.get_map()
        self._spawn_points = self._map.get_spawn_points()

    # ------------------------------------------------------------------
    # Episode Lifecycle
    # ------------------------------------------------------------------

    def setup_episode(self) -> dict[str, Any]:
        """Spawn all actors for a new episode. Returns episode metadata."""
        env_cfg = self._config.get("environment", {})
        scene_cfg = self._config.get("scene", {})
        self._route_anchor_calls, self._route_meta, self._route_plan = 0, None, None   # 1026 stress_v2 (per episode)

        # -- Seeding (P1-b): make the whole episode reproducible --
        seed = env_cfg.get("seed", None)
        if seed is not None:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
        tm = self._client.get_trafficmanager(_tm_port())
        tm.set_synchronous_mode(True)
        # Official generate_traffic.py settings → calm, rule-abiding traffic.
        tm.set_global_distance_to_leading_vehicle(2.5)
        tm.global_percentage_speed_difference(20.0)  # ambient drives 20% below limit
        if seed is not None and hasattr(tm, "set_random_device_seed"):
            tm.set_random_device_seed(seed)

        # -- Clean slate: remove any leftover actors from a prior episode so the
        #    TrafficManager never ticks a destroyed vehicle (avoids C++ aborts). --
        self._destroy_all_traffic()

        # -- Randomize weather --
        self._set_random_weather(self._config.get("weather", {}))

        # -- Language / scene composer --
        self._language_gen = LanguageGenerator()

        # -- Select wide-road town (loaded by the batch driver; sanity fallback) --
        wide_road_towns = env_cfg.get("towns", ["Town03", "Town04", "Town05"])
        if self._world.get_map().name.split("/")[-1] not in wide_road_towns:
            try:
                self._client.load_world(random.choice(wide_road_towns))
                time.sleep(2.0)
                self._world = self._client.get_world()
                self._map = self._world.get_map()
            except RuntimeError:
                pass

        # -- Choose target class + similarity strategy --
        self._target_class = self._pick_target_class(scene_cfg)
        strategy = self._pick_strategy(scene_cfg, self._target_class)
        num_distractors = random.randint(*scene_cfg.get("num_distractors", [4, 8]))
        min_similar = int(scene_cfg.get("min_similar_distractors", 2))

        # Pick tracking-distance tier → UAV maintained altitude + language distance.
        tier_cfg = self._config.get("distance_tiers", {})
        self._distance_tier = self._pick_distance_tier(tier_cfg)
        self._tier_alt = tier_cfg.get(self._distance_tier, {}).get("altitude", [24, 32])
        # §8 (redo): per-episode HORIZONTAL tracking standoff (metres the UAV TRAILS the
        # target — applied in expert_action) → randomises the target's forward/depth position
        # in the frame → decorrelates its depth RANK. NOT altitude (that only scales absolute
        # depth uniformly, leaving the rank — the actual `nearest` shortcut — unchanged).
        _dec = self._config.get("scene", {}).get("decorrelate_position", {})
        if _dec.get("enabled", False):
            sr = _dec.get("standoff_range", [8.0, 40.0])
            self._tracking_standoff = float(random.uniform(float(sr[0]), float(sr[1])))
            # §9: smooth per-episode camera-AIM offset so the target is NOT locked at image
            # centre → decorrelates 'most-central'. The aim point = target + a ground-plane
            # offset from two low-freq sinusoids (random per-episode phase/freq) whose
            # amplitude scales with viewing distance (constant IMAGE spread ~aim_frac,
            # target stays in frame). Smooth + independent of maneuver/loss events (§9.3.5).
            self._aim_frac = float(_dec.get("aim_offset_frac", 0.32))
            self._aim_fx = random.uniform(0.04, 0.11); self._aim_px = random.uniform(0.0, 6.283)
            self._aim_fy = random.uniform(0.04, 0.11); self._aim_py = random.uniform(0.0, 6.283)
        else:
            self._tracking_standoff = 0.0
            self._aim_frac = 0.0

        # A1c③ — decide per-episode whether to (a) spawn the target so it drives into an
        # overhead occluder (in-frame structural occlusion), and (b) use an oblique,
        # lower-altitude camera (surfaces urban side-wall occlusion). Both default OFF.
        occ_cfg = self._config.get("occlusion", {})
        cam_cfg = self._config.get("camera", {})
        self._occlusion_biased = (bool(occ_cfg.get("enabled", False))
                                  and random.random() < float(occ_cfg.get("bias_fraction", 0.0)))
        self._camera_pitch, self._camera_oblique = self._pick_camera(cam_cfg)
        if self._camera_oblique:
            self._tier_alt = list(cam_cfg.get("oblique", {}).get("altitude_range", self._tier_alt))

        plan = self._language_gen.compose_scene(
            target_class=self._target_class,
            num_distractors=num_distractors,
            strategy=strategy,
            min_similar=min_similar,
        )
        self.target_meta = plan["target"]

        # -- Spawn target + distractors (co-located & co-visible) --
        if self._target_class in VEHICLE_CLASSES:
            # v6.1: the spawn filter narrows the anchor set → a slot spawn can fail at every try; re-pick
            # the anchor (destroying the distractors of the failed attempt) instead of crashing later.
            for _anchor_try in range(5):
                self._setup_vehicle_scene(plan, scene_cfg)
                if self._target is not None:
                    break
                for o in self._distractors:
                    try:
                        o.destroy()
                    except (RuntimeError, AttributeError):
                        pass
            if self._target is None:
                raise RuntimeError("target spawn failed at 5 anchors")
        else:
            self._setup_pedestrian_scene(plan, scene_cfg)

        self._target_type = "vehicle" if self._target_class in VEHICLE_CLASSES else "pedestrian"
        print(f"  [{self._target_class}/{plan['strategy']}] target={plan['target']['desc']} "
              f"| distractors={len(self._distractors)} (similar={plan['num_similar']})")

        # -- Background traffic (official-style): diverse vehicles + pedestrians --
        self._spawn_ambient_traffic(scene_cfg)

        # v6: the target sat still 40-70 % of the time (red lights / queues behind 80 ambient cars) →
        # little motion, no real losses. Let the scene actors (target + distractors, so the look-alike
        # group stays together) ignore lights/signs; ambient traffic keeps obeying them.
        # v6.1: queues behind ambient cars waiting at long reds (CARLA default 4-way red ≈ 30-45 s) kept the
        # target still → shorten every light's cycle (set per episode; lights persist on the server).
        tl_cfg = scene_cfg.get("traffic_lights") or {}
        if tl_cfg:
            for tl in self._world.get_actors().filter("traffic.traffic_light"):
                try:
                    tl.set_green_time(float(tl_cfg.get("green_s", 6.0)))
                    tl.set_yellow_time(float(tl_cfg.get("yellow_s", 2.0)))
                    tl.set_red_time(float(tl_cfg.get("red_s", 2.0)))
                except RuntimeError:
                    pass
        ign = float(scene_cfg.get("scene_actors_ignore_lights", 0.0))
        if ign > 0:
            tm = self._client.get_trafficmanager(_tm_port())
            for o in [self._target] + list(self._distractors):
                v = getattr(o, "_vehicle", None)
                if v is None:
                    continue
                try:
                    tm.ignore_lights_percentage(v, ign)
                    tm.ignore_signs_percentage(v, ign)
                except RuntimeError:
                    pass
        if scene_cfg.get("scene_actors_lane_change", False):
            # v6.1: VehicleTarget disables lane changes → a stopped car ahead blocked the target for good
            tm = self._client.get_trafficmanager(_tm_port())
            for o in [self._target] + list(self._distractors):
                v = getattr(o, "_vehicle", None)
                if v is not None:
                    try:
                        tm.auto_lane_change(v, True)
                    except RuntimeError:
                        pass

        # 2026-10-09: realistic urban speeds + anti-gridlock (scene.kinematics; absent = legacy behaviour)
        self._apply_kinematics()

        # -- Spawn UAV behind & above the target --
        self._spawn_uav(self._config.get("uav", {}))

        # -- Expert policy / safety (maintain the tier's tracking altitude) --
        exp_cfg = dict(self._config.get("uav", {}).get("expert_policy", {}))
        exp_cfg["viewing_height_range"] = self._tier_alt
        self.expert = ExpertPolicy(exp_cfg)
        self.expert.reset_episode()
        self.safety = SafetyMonitor()
        self.safety.reset()

        # -- Perturbation injector (P: honor config) --
        from carla_uav.perturbation import PerturbationInjector, PerturbationConfig
        pert_cfg = self._config.get("uav", {}).get("perturbation", {})
        self._perturbation_injector = PerturbationInjector(
            PerturbationConfig(**{k: v for k, v in pert_cfg.items()
                                  if k in PerturbationConfig.__dataclass_fields__})
        )
        self._perturbation_injector.reset()

        # -- Maneuver injector (TM-based, smooth — modulates speed/lane, never raw
        #    control, so the target stays on-road and obeys traffic rules) --
        man_cfg = self._config.get("target", {}).get("maneuver", {})
        self._maneuver_injector = ManeuverInjector(
            ManeuverConfig(**{k: v for k, v in man_cfg.items()
                              if k in ManeuverConfig.__dataclass_fields__}),
            traffic_manager=self._client.get_trafficmanager(_tm_port()),
        )
        self._maneuver_injector.reset()

        # -- (c) deliberate SUSTAINED loss events for predict-intercept recovery --
        env_cfg = self._config.get("environment", {})
        self._loss_scheduler = LossEventScheduler(env_cfg, fps=env_cfg.get("fps", 10))
        self._loss_scheduler.reset(seed=env_cfg.get("seed"))

        # -- Obstacle avoider --
        from carla_uav.obstacle_avoidance import ObstacleAvoider
        self._obstacle_avoider = ObstacleAvoider()

        # -- Language instruction (full referring expression). Spatial + distance from
        #    real geometry now; motion intent refined post-hoc from the trajectory. --
        from scene.language_generator import spatial_and_distance
        self.target_plan = plan
        spatial, dist_m = "", None
        try:
            u = self.drone.transform
            t = self._target.transform.location
            spatial, dist_m = spatial_and_distance(
                u.location.x, u.location.y, u.rotation.yaw, t.x, t.y,
                uz=u.location.z, tz=t.z)
        except (RuntimeError, AttributeError):
            pass
        heading = getattr(self._target, "direction", "")
        if (self._config.get("scene", {}).get("instruction", {}) or {}).get("geometry", True) is False:
            # 1020 stress scenario T2: no position / heading / distance phrase (a geometric shortcut at the
            # opening) — the instruction names only the colour and model
            spatial, heading, dist_m = "", "", None
        self.episode_language = self._language_gen.build_instruction(
            plan["target"]["desc"], plan.get("num_similar", 0),
            spatial=spatial, motion=(f"heading {heading}" if heading else ""),
            distance_m=dist_m)

        return {
            "target_type": self._target_type,
            "target_class": self._target_class,
            "target_blueprint": plan["target"]["bp"],
            "target_color": plan["target"].get("color"),
            "target_desc": plan["target"]["desc"],
            "strategy": plan["strategy"],
            "num_similar": plan["num_similar"],
            "num_distractors": len(self._distractors),
            "distractors": self._distractor_meta,
            "seed": seed,
            "uav_start_altitude": float(self.drone.transform.location.z) if self.drone else None,
            "weather": self._get_current_weather_dict(),
            "language": self.episode_language,
            "occlusion_biased": bool(getattr(self, "_occlusion_biased", False)),
            # 1020 stress scenario: clone composition + the scheduled loss windows (E3, seconds since start)
            "num_clones": int(plan.get("num_clones", 0)),
            "num_near_clones": int(plan.get("num_near_clones", 0)),
            "loss_windows": [list(w) for w in self._loss_scheduler.windows],
            "camera_pitch": float(getattr(self, "_camera_pitch", -50.0)),
            # 1026 stress_v2 (absent keys = legacy): difficulty tier of the town + the planned convoy route
            **({"town": self._world.get_map().name.split("/")[-1],
                "town_tier": next((k for k, v in (env_cfg.get("town_tiers") or {}).items()
                                   if self._world.get_map().name.split("/")[-1] in v), None),
                "route_plan": getattr(self, "_route_meta", None)} if self._route_cfg() else {}),
        }

    # ------------------------------------------------------------------
    # Spawning helpers
    # ------------------------------------------------------------------

    def _pick_distance_tier(self, tier_cfg: dict) -> str:
        if not tier_cfg:
            return "suitable"
        names = list(tier_cfg.keys())
        w = np.array([float(tier_cfg[n].get("weight", 1.0)) for n in names])
        return str(np.random.choice(names, p=w / w.sum()))

    def _pick_camera(self, cam_cfg: dict) -> tuple[float, bool]:
        """(camera_pitch, is_oblique). A1c③: a fraction of episodes use a shallower
        pitch (+ lower altitude, set by the caller) to expose side-wall occlusion."""
        default_pitch = float(cam_cfg.get("pitch_deg", -50.0))
        ob = cam_cfg.get("oblique", {})
        if ob and random.random() < float(ob.get("fraction", 0.0)):
            return float(ob.get("pitch_deg", -38.0)), True
        return default_pitch, False

    def _pick_target_anchor(self) -> "carla.Transform":
        """Target spawn transform. On occlusion-biased episodes, spawn on a map spawn
        point whose forward lane drives under an overhead occluder (A1c③); otherwise a
        random spawn point. Falls back to random if no occluder anchors in this town."""
        cands = None
        rp = self._route_cfg()
        if rp and rp.get("anchor_search", 0):
            a = self._route_anchor(int(rp["anchor_search"]))
            if a is not None:
                return a
        if getattr(self, "_occlusion_biased", False):
            anchors = self._occluder_anchors()
            if anchors:
                cands = anchors
        if cands is None:
            cands = self._spawn_points
        if not cands:
            return carla.Transform(carla.Location(0, 0, 1))
        sf = self._config.get("scene", {}).get("spawn_filter") or {}
        if not sf:
            return random.choice(cands)
        # v6.1 explicit scope: a UAV trailing at 16-34 m must be ABOVE the local rooftops, so only spawn
        # where every non-ground surface within 40 m is <= max_roof_m above the road (downtown high-rise
        # canyons put the camera against walls/roofs). Recorded in attrs as spawn_filter.
        # Per-town fallback (probe 10-03: Town10HD has only 3 % of spawns <= 12 m, median roof 33 m): if fewer
        # than `min_frac` qualify, use the town's lowest-roof `min_frac` quantile instead (keeps diversity).
        max_roof, min_frac = float(sf.get("max_roof_m", 12.0)), float(sf.get("min_frac", 0.2))
        c0 = cands[0].location
        key = (self._map.name, len(cands), round(c0.x, 1), round(c0.y, 1))   # stable across episodes in a process
        cache = getattr(SceneManager, "_roof_cache", {})
        if key not in cache:
            cache[key] = [self._max_roof_around(sp.location, (20.0, 40.0)) for sp in cands]
            SceneManager._roof_cache = cache
        roofs = cache[key]
        ok = [i for i, r in enumerate(roofs) if r <= max_roof]
        if len(ok) < min_frac * len(cands):
            ok = sorted(range(len(cands)), key=lambda i: roofs[i])[:max(1, int(round(min_frac * len(cands))))]
        i = random.choice(ok)
        self._spawn_roof_m = float(roofs[i])
        return cands[i]

    def _max_roof_around(self, loc, radii=(20.0, 40.0), n_dir: int = 8) -> float:
        """Max height (m above loc.z) of non-ground surfaces sampled by vertical rays on rings around loc."""
        L = carla.CityObjectLabel
        ground = {getattr(L, n) for n in ("Roads", "Sidewalks", "Ground", "Terrain", "RoadLines", "Water",
                                          "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pedestrians", "Rider")
                  if hasattr(L, n)}
        best = 0.0
        pts = [(0.0, 0.0)] + [(r * math.cos(2 * math.pi * k / n_dir), r * math.sin(2 * math.pi * k / n_dir))
                              for r in radii for k in range(n_dir)]
        for dx, dy in pts:
            a = carla.Location(loc.x + dx, loc.y + dy, loc.z + 150.0)
            b = carla.Location(loc.x + dx, loc.y + dy, loc.z - 5.0)
            try:
                for h in self._world.cast_ray(a, b):
                    if h.label not in ground:
                        best = max(best, h.location.z - loc.z)
                        break
            except RuntimeError:
                pass
        return float(best)

    def _occluder_anchors(self) -> list:
        """Lazily load (build + cache once) the current town's occluder-leading spawns."""
        town = self._world.get_map().name.split("/")[-1]
        cache = getattr(self, "_occ_anchor_cache", {})
        if town not in cache:
            from scene.occluder_cache import load_or_build, anchors_to_transforms
            occ_cfg = self._config.get("occlusion", {})
            params = {k: occ_cfg[k] for k in
                      ("look_ahead_m", "step_m", "up_ray_m",
                       "min_clearance_m", "max_clearance_m",
                       "min_enter_m", "max_cover_span_m") if k in occ_cfg}
            try:
                data = load_or_build(self._world, town,
                                     occ_cfg.get("cache_dir", "/data/occ_cache"), params)
                cache[town] = anchors_to_transforms(data)
            except RuntimeError:
                cache[town] = []
            self._occ_anchor_cache = cache
            print(f"  [occlusion] {town}: {len(cache[town])} occluder-leading spawns")
        return cache[town]

    def _pick_target_class(self, scene_cfg: dict) -> str:
        weights = scene_cfg.get("target_class_weights", {"car": 1.0})
        classes = list(weights.keys())
        probs = np.array([weights[c] for c in classes], dtype=float)
        probs = probs / probs.sum()
        return str(np.random.choice(classes, p=probs))

    def _pick_strategy(self, scene_cfg: dict, target_class: str) -> str:
        if target_class != "car":
            return "same_class"
        weights = scene_cfg.get("similarity_strategy_weights", {
            "same_color_diff_shape": 0.4, "same_shape_diff_color": 0.4, "distinct": 0.2})
        strategies = list(weights.keys())
        probs = np.array([weights[s] for s in strategies], dtype=float)
        probs = probs / probs.sum()
        return str(np.random.choice(strategies, p=probs))

    def _setup_vehicle_scene(self, plan: dict, scene_cfg: dict) -> None:
        """Spawn a vehicle target + distractors on the same road segment.

        Position-decorrelation (2026-07-29, per acot_note/data-fix-spec.md): the target
        must NOT sit in a predictable geometric slot, or a pure-geometry classifier
        selects it WITHOUT language (the benchmark leak — position-only MLP ~3-5%). Two
        levers, both config-gated by `scene.decorrelate_position.enabled`:
          L1 slot: the target + its SIMILAR look-alikes are drawn from ONE symmetric
             slot band and the target takes a RANDOM slot (was: target always at the
             anchor, distractors biased ahead) → target's depth/rank == a distractor's.
          L2 framing: the expert servos the {target+similar} CENTROID, not the target
             (`self._frame_centroid`, applied in expert_action + _cluster_centroid), so
             the target is a random cluster member (was: always the framed reference) →
             its image position / depth / rank distribution == a distractor's.
        """
        anchor = self._pick_target_anchor()
        tgt = plan["target"]
        # Positive % = SLOWER than the speed limit (official generate_traffic.py uses
        # +30%). Calm, shared convoy speed so they travel together, not aggressively.
        convoy_speed = random.uniform(0, 18)
        dec_cfg = scene_cfg.get("decorrelate_position", {})
        decorrelate = bool(dec_cfg.get("enabled", True))

        self._distractors = []
        self._distractor_meta = []
        anchor_wp = self._map.get_waypoint(anchor.location)
        # Candidate lanes = anchor lane + same-direction neighbours, so vehicles spread
        # ACROSS lanes (co-visible) instead of gridlocking one lane.
        lanes = [anchor_wp]
        for getter in ("get_left_lane", "get_right_lane"):
            try:
                cand = getattr(anchor_wp, getter)()
            except RuntimeError:
                cand = None
            if (cand is not None and cand.lane_type == carla.LaneType.Driving
                    and cand.lane_id * anchor_wp.lane_id > 0):
                lanes.append(cand)

        if decorrelate:
            band = float(dec_cfg.get("cluster_band_m", 24.0))
            # L1 — target gets a RANDOM slot; capture its along-road offset off_t so the
            # distractors can be placed to BRACKET it in depth (§8: the target used to be
            # the depth-nearest candidate ~50% of frames because fillers were pushed forward
            # /farther by the down-forward camera — no candidate was ever nearer than the
            # target. Bracketing off_t makes the target a uniform depth-RANK member).
            off_t = 0.0
            self._target = None
            for _try in range(6):
                off_t = 0.0 if _try == 5 else random.uniform(-band, band)
                sp = (anchor if _try == 5 else
                      self._lane_offset_transform(random.choice(lanes), off_t))
                try:
                    self._target = VehicleTarget(
                        self._world, tgt["bp"], sp, carla_client=self._client,
                        color=COLOR_RGB.get(tgt.get("color") or ""),
                        # SAME speed distribution as distractors so the target does not drift
                        # to a systematic depth within the group (hero-only dynamics leaked).
                        role_name="hero", speed_diff=self._target_speed_diff_pick(convoy_speed))
                    break
                except RuntimeError:
                    continue
            for d in plan["distractors"]:
                # Similars share the anchor ±band (INDEPENDENT of the target — they + the
                # target form the framed centroid, so the target stays a random member).
                # NOTE: the residual DEPTH shortcut (target is depth-nearest ~50% of frames)
                # is NOT fixed by any distractor PLACEMENT — the target sits at the back of
                # the followed cluster by tracking geometry; see data-fix-spec §8 follow-up.
                if d.get("similar"):
                    off = random.uniform(-band, band)
                else:
                    off = random.uniform(8, 30) * random.choice([-1, 1])
                self._spawn_distractor(
                    d, self._lane_offset_transform(random.choice(lanes), off), convoy_speed)
            # L2 — the expert servos the CLUSTER CENTROID (target + similar look-alikes),
            # NOT the target (applied in expert_action). The target is then a uniformly
            # random member of a tight co-visible cluster → its image position / depth /
            # rank distribution is IDENTICAL to a distractor's by construction (any
            # deterministic target-centred framing, even an offset one, leaks). The tight
            # ±band cluster + wide FOV keeps every member on-screen.
            self._frame_centroid = True
        else:
            # Legacy layout (leaky): target at the anchor, distractors biased ahead.
            self._frame_centroid = False
            self._target = VehicleTarget(
                self._world, tgt["bp"], anchor, carla_client=self._client,
                color=COLOR_RGB.get(tgt.get("color") or ""),
                role_name="hero", speed_diff=convoy_speed + random.uniform(-3, 3))
            ahead_bias = float(scene_cfg.get("distractor_ahead_bias", 0.7))
            for d in plan["distractors"]:
                near, far = (5, 28) if d.get("similar") else (10, 55)
                off = (random.uniform(near, far) if random.random() < ahead_bias
                       else -random.uniform(6, 18))
                self._spawn_distractor(
                    d, self._lane_offset_transform(random.choice(lanes), off), convoy_speed)

        self._target_in_frame = True      # updated each step by the recorder (recovery gate)
        self._recovery_mode = False       # hysteresis state for the centroid↔recovery framing switch
        self._reacq = 0
        if self._route_cfg():
            self._assign_planned_route()                   # 1026 stress_v2: replaces the (disabled) convoy route
        else:
            self._assign_convoy_route()

    def _lane_offset_transform(self, base_wp, offset_m: float) -> "carla.Transform":
        """Walk `offset_m` along the lane (forward if +, backward if −) from base_wp and
        return a spawn transform with small lateral jitter (raised 0.3 m)."""
        wp = base_wp
        for _ in range(int(abs(offset_m) / 8)):
            nxt = wp.next(8.0) if offset_m > 0 else wp.previous(8.0)
            if not nxt:
                break
            wp = nxt[0]
        loc = wp.transform.location
        return carla.Transform(
            carla.Location(x=loc.x + random.uniform(-1.0, 1.0),
                           y=loc.y + random.uniform(-1.0, 1.0), z=loc.z + 0.3),
            wp.transform.rotation)

    def _target_speed_diff_pick(self, convoy_speed: float) -> float:
        """the target's TM speed difference (% below the limit); remembered for the clones' E1 offsets."""
        self._target_speed_diff = convoy_speed + random.uniform(-5, 5)
        return self._target_speed_diff

    def _distractor_speed_diff(self, d: dict, convoy_speed: float) -> float:
        """1020 stress scenario E1 (crossing / overtaking): a CLONE drives a configured percentage faster or slower
        than the target (sign random per clone), so clones and target exchange places inside the cluster band — the
        tracker's ID swaps are provoked by construction. Everything else keeps the legacy convoy speed."""
        cd = self._config.get("scene", {}).get("clone_dynamics", {}) or {}
        base = getattr(self, "_target_speed_diff", None)
        if d.get("clone") and cd.get("enabled", False) and base is not None:
            lo, hi = cd.get("speed_offset_pct", [12.0, 22.0])
            v = base + random.choice([-1.0, 1.0]) * random.uniform(float(lo), float(hi))
            return float(min(max(v, -30.0), 60.0))
        return convoy_speed + random.uniform(-5, 5)

    def _spawn_distractor(self, d: dict, sp: "carla.Transform", convoy_speed: float) -> None:
        """Spawn one distractor (vehicle or pedestrian) at sp; skip on spawn collision."""
        try:
            cls = d["cls"]
            if cls in VEHICLE_CLASSES:
                obj = VehicleTarget(
                    self._world, d["bp"], sp, carla_client=self._client,
                    color=COLOR_RGB.get(d.get("color") or ""),
                    role_name="distractor", speed_diff=self._distractor_speed_diff(d, convoy_speed),
                    lane_change=bool(d.get("clone")) and bool(
                        (self._config.get("scene", {}).get("clone_dynamics", {}) or {}).get("lane_change", False)))
            else:
                obj = PedestrianTarget(self._world, d["bp"], sp, walk_speed=random.uniform(2.5, 3.5))
            self._distractors.append(obj)
            self._distractor_meta.append({
                "id": obj.id, "bp": d["bp"], "color": d.get("color"),
                "desc": d["desc"], "similar": d.get("similar", False), "cls": cls,
                "clone": bool(d.get("clone", False)), "near_clone": bool(d.get("near_clone", False)),
                "speed_diff": getattr(obj, "_speed_diff_set", None),
                "desired_off": d.get("_desired_off"),   # §8 depth station-keeping target (or None)
            })
        except RuntimeError:
            pass  # spawn collision — skip this distractor

    def _cluster_centroid(self, target_pos: np.ndarray) -> np.ndarray:
        """Mean world position of the co-visible cluster = target + its SIMILAR look-alikes,
        used as the expert's framing setpoint so the target is a random cluster member (not
        the framed reference). Reuses the distractor positions the recorder already queried
        this step (`self._distractor_states`, set in get_distractor_states) → no extra CARLA
        RPCs. Falls back to the target if no similar distractor is currently alive."""
        tp = np.asarray(target_pos, dtype=float)
        pts = [tp]
        # 1020 stress scenario (no target-anchored recovery): only look-alikes NEAR the target join the centroid, so
        # the framing never leaves the target behind when it separates from its clones (a lone target is followed
        # directly — no look-alike is near it then, so following it reveals nothing about which car it is)
        radius = float(getattr(self, "_centroid_radius_m", 0.0) or 0.0)
        ds = getattr(self, "_distractor_states", None)
        if ds is not None and len(ds):
            for row, meta in zip(ds, self._distractor_meta):
                if not meta.get("similar"):
                    continue
                p = np.asarray(row[:3], dtype=float)
                if np.isnan(p).any():
                    continue
                if radius > 0.0 and float(np.linalg.norm(p[:2] - tp[:2])) > radius:
                    continue
                pts.append(p)
        return np.mean(pts, axis=0)

    def _assign_convoy_route(self) -> None:
        """Give the target + vehicle distractors the SAME route (official TM.set_route)
        so they drive together on the same roads → stay co-visible while obeying
        traffic rules, with no teleporting (replaces recycle_lost_lookalikes)."""
        cfg = self._config.get("scene", {}).get("convoy", {})
        if not cfg.get("enabled", True):
            return
        tm = self._client.get_trafficmanager(_tm_port())
        if not hasattr(tm, "set_route"):
            return
        sb = float(cfg.get("straight_bias", 0.7))
        opts = ["Straight"] * max(1, int(sb * 10)) + ["Left", "Right"] * max(1, int((1 - sb) * 5))
        route = [random.choice(opts) for _ in range(int(cfg.get("route_len", 80)))]
        members = ([self._target] if self._target else []) + self._distractors
        for m in members:
            v = getattr(m, "_vehicle", None)
            if v is None:
                continue
            try:
                tm.set_route(v, list(route))
            except (RuntimeError, TypeError):
                pass

    # ------------------------------------------------------------------
    # 1026 stress_v2: planned convoy route (scene/route_planner.py)
    # ------------------------------------------------------------------
    def _route_cfg(self) -> dict | None:
        """scene.route_plan (dict, enabled) or None — absent / disabled = legacy behaviour everywhere."""
        rp = (self._config.get("scene", {}) or {}).get("route_plan") or None
        return rp if (rp and rp.get("enabled", False)) else None

    def _route_graph(self) -> dict | None:
        """the current town's lane graph + overhead coverage (built once per town, cached on disk and in-process)."""
        rp = self._route_cfg()
        town = self._world.get_map().name.split("/")[-1]
        cache = getattr(SceneManager, "_route_graph_cache", {})
        if town not in cache:
            from scene.route_planner import load_or_build_graph
            params = {k: rp[k] for k in ("step_m", "up_ray_m", "min_clear", "max_clear") if k in rp}
            t0 = time.time()
            try:
                cache[town] = load_or_build_graph(self._world, town, rp.get("cache_dir", "/data/occ_cache"), params)
            except RuntimeError as e:
                print(f"  [route] {town}: graph build failed ({e})")
                cache[town] = None
            SceneManager._route_graph_cache = cache
            g = cache[town]
            if g is not None:
                print(f"  [route] {town}: {len(g['edges'])} lane edges, "
                      f"{sum(len(e['pts']) for e in g['edges'])} samples ({time.time() - t0:.0f} s)")
        return cache[town]

    def _route_rng(self, salt: int) -> random.Random:
        """route planning has its own RNG (seeded from the episode seed) so it never shifts the scene's draws"""
        seed = int(self._config.get("environment", {}).get("seed") or 0)
        return random.Random(seed * 1000003 + salt)

    def _route_anchor(self, n: int):
        """the best of n random map spawn points by a quick plan score (turns + occluders ahead), or None"""
        from scene.route_planner import plan_route
        g = self._route_graph()
        if g is None or not self._spawn_points:
            return None
        rp = self._route_cfg()
        self._route_anchor_calls = getattr(self, "_route_anchor_calls", 0) + 1   # a re-pick after a failed spawn
        rng = self._route_rng(100 + self._route_anchor_calls)                   # draws a different pool
        pool = list(self._spawn_points)
        rng.shuffle(pool)
        quick = dict(rp.get("planner", {}) or {}, n_walks=int(rp.get("anchor_walks", 40)))
        best, best_sc = None, None
        for sp in pool[:n]:
            loc = sp.location
            plan = plan_route(g, (loc.x, loc.y, loc.z), sp.rotation.yaw, rng, quick)
            if plan is None:
                continue
            if best_sc is None or plan["score"] > best_sc:
                best, best_sc = sp, plan["score"]
        return best

    def _assign_planned_route(self) -> None:
        """plan the convoy route from the target's spawn pose and hand every scene vehicle (target + vehicle
        distractors) the planned locations still ahead of it (TM.set_path). Result summary → self._route_meta."""
        from scene.route_planner import plan_route, points_for
        self._route_meta = {"planned": False}
        g = self._route_graph()
        if g is None or self._target is None:
            return
        rp = self._route_cfg()
        try:
            t = self._target.transform
        except RuntimeError:
            return
        plan = plan_route(g, (t.location.x, t.location.y, t.location.z), t.rotation.yaw, self._route_rng(2),
                          rp.get("planner", {}) or {})
        if plan is None:
            print("  [route] target pose not on the lane graph → no planned route")
            return
        self._route_plan = plan
        tm = self._client.get_trafficmanager(_tm_port())
        n_set = 0
        for o in [self._target] + list(self._distractors):
            v = getattr(o, "_vehicle", None)
            if v is None:
                continue
            try:
                loc = v.get_transform().location
                pts = points_for(plan, (loc.x, loc.y, loc.z))
                if pts:
                    tm.set_path(v, [carla.Location(x, y, z + 0.5) for _s, x, y, z in pts])
                    n_set += 1
            except (RuntimeError, TypeError):
                pass
        self._route_meta = {"planned": True, "length_m": round(plan["length_m"], 1), "score": plan["score"],
                            "occluders": plan["occluders"], "too_long": plan["too_long"],
                            "turns": plan["turns"], "n_points": len(plan["points"]), "n_vehicles_set": n_set,
                            "polyline": [[round(x, 1), round(y, 1)] for _s, x, y, *_r in plan["samples"][::5]]}
        print(f"  [route] {plan['length_m']:.0f} m, {len(plan['turns'])} turns, occluders "
              f"{[(o['enter_m'], o['span_m'], o['label']) for o in plan['occluders']]}, set_path on {n_set} vehicles")

    def _reroute_after_relocation(self, actor) -> None:
        """a scene distractor teleported by unjam: hand it the planned locations still ahead of the TARGET, so it
        drives back toward the convoy's route instead of chasing locations it can no longer consume."""
        from scene.route_planner import points_for
        plan = getattr(self, "_route_plan", None)
        if plan is None or self._target is None:
            return
        try:
            t = self._target.transform.location
            pts = points_for(plan, (t.x, t.y, t.z))
            if pts:
                self._client.get_trafficmanager(_tm_port()).set_path(
                    actor, [carla.Location(x, y, z + 0.5) for _s, x, y, z in pts])
        except (RuntimeError, TypeError):
            pass

    def _nav_near(self, center, radius: float, count: int, attempts: int = 120) -> list:
        """Sample navmesh locations within `radius` metres of `center`."""
        out = []
        for _ in range(attempts):
            loc = self._world.get_random_location_from_navigation()
            if loc is not None and loc.distance(center) <= radius:
                out.append(loc)
                if len(out) >= count:
                    break
        return out

    def _spawn_ambient_traffic(self, scene_cfg: dict) -> None:
        """Populate background traffic CONCENTRATED NEAR THE TARGET the official way:
        diverse vehicles (cars/bus/trucks/motos/bikes) via batch SpawnActor+SetAutopilot,
        kept near the hero (target) by hybrid-physics + respawn_dormant_vehicles; plus
        manually-controlled background pedestrians on the nearby navmesh."""
        amb = scene_cfg.get("ambient", {})
        self._ambient_ids = []                 # 2026-10-09: ids of this episode's ambient cars (kinematics / unjam)
        n_veh = int(amb.get("vehicles", 0))
        n_walk = int(amb.get("walkers", 0))
        radius = float(amb.get("radius", 90.0))
        if self._target is None:
            return
        try:
            tloc = self._target.transform.location
        except RuntimeError:
            return
        tm = self._client.get_trafficmanager(_tm_port())

        # Official mechanism: hybrid physics around the hero (target) + respawn dormant
        # vehicles → ambient traffic auto-concentrates near the moving target.
        if self._target_class in VEHICLE_CLASSES:
            try:
                tm.set_hybrid_physics_mode(True)
                tm.set_hybrid_physics_radius(radius)
                tm.set_respawn_dormant_vehicles(True)
            except (RuntimeError, AttributeError):
                pass

        # --- vehicles: batch spawn at the spawn points NEAREST the target ---
        if n_veh > 0:
            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            FutureActor = carla.command.FutureActor
            vbps = list(self._world.get_blueprint_library().filter("vehicle.*"))
            if amb.get("cars_only", False):        # 4-wheel cars only (no bikes/motos) — spec §4d
                cars = [b for b in vbps if b.has_attribute("number_of_wheels")
                        and int(b.get_attribute("number_of_wheels")) == 4]
                if cars:
                    vbps = cars
            sps = sorted(self._world.get_map().get_spawn_points(),
                         key=lambda sp: sp.location.distance(tloc))
            if amb.get("spawn_mode", "nearest") == "ring":
                # v6.1: the n nearest points packed the target into a queue from t=0 (stationary 40-100 %);
                # sample uniformly from a ring instead so traffic is around, not on top of, the target.
                r0, r1 = (float(x) for x in amb.get("ring_m", [30.0, 200.0]))
                ring = [sp for sp in sps if r0 <= sp.location.distance(tloc) <= r1]
                sel = random.sample(ring, min(n_veh, len(ring)))
            else:
                sel = sps[1:n_veh + 1]     # nearest points (skip target's own) — original v5 behaviour
            batch = []
            for sp in sel:
                bp = random.choice(vbps)
                if bp.has_attribute("color"):
                    try:
                        bp.set_attribute("color", random.choice(
                            bp.get_attribute("color").recommended_values))
                    except (RuntimeError, ValueError, IndexError):
                        pass
                bp.set_attribute("role_name", "ambient")
                batch.append(SpawnActor(bp, sp)
                             .then(SetAutopilot(FutureActor, True, tm.get_port())))
            try:
                res = self._client.apply_batch_sync(batch, True)
            except RuntimeError:
                res = []
            self._ambient_ids = [r.actor_id for r in res if not r.error]
            kin = self._kin()
            if kin and kin.get("ambient_speed_mps"):
                # 2026-10-09: ambient cars at urban speed too (legacy: limit − 20 % ≈ 6.7 m/s) so the convoy is not
                # permanently queued behind them; TM keeps the per-vehicle value across dormant respawns.
                lo, hi = (float(x) for x in kin["ambient_speed_mps"])
                for a in self._world.get_actors(self._ambient_ids):
                    try:
                        tm.set_desired_speed(a, self._kmh(random.uniform(lo, hi)))
                    except RuntimeError:
                        pass

        # --- pedestrians: manual WalkerControl on the navmesh NEAR the target ---
        self._ambient_walkers = []
        wbps = [b.id for b in self._world.get_blueprint_library().filter("walker.pedestrian.*")]
        for loc in self._nav_near(tloc, radius, n_walk):
            if not wbps:
                break
            try:
                self._ambient_walkers.append(PedestrianTarget(
                    self._world, random.choice(wbps), carla.Transform(loc),
                    walk_speed=random.uniform(1.2, 2.2), snap_to_road=False,
                    role_name="ambient_walker"))
            except RuntimeError:
                pass

    def _setup_pedestrian_scene(self, plan: dict, scene_cfg: dict) -> None:
        """Spawn a pedestrian target + nearby distractors on a road (manual walkers)."""
        anchor = random.choice(self._spawn_points) if self._spawn_points else \
            carla.Transform(carla.Location(0, 0, 1))
        tgt = plan["target"]
        self._target = PedestrianTarget(
            self._world, tgt["bp"], anchor, walk_speed=random.uniform(1.4, 2.4))
        self._distractors = []
        self._distractor_meta = []
        anchor_wp = self._map.get_waypoint(anchor.location)
        for d in plan["distractors"]:
            try:
                offset_m = random.uniform(5, 30) if random.random() < 0.7 \
                    else -random.uniform(5, 15)
                d_wp = anchor_wp
                for _ in range(int(abs(offset_m) / 8)):
                    nxt = d_wp.next(8.0) if offset_m > 0 else d_wp.previous(8.0)
                    if nxt:
                        d_wp = nxt[0]
                dsp = carla.Transform(
                    carla.Location(
                        d_wp.transform.location.x + random.uniform(-2, 2),
                        d_wp.transform.location.y + random.uniform(-2, 2),
                        d_wp.transform.location.z + 0.5),
                    d_wp.transform.rotation)
                if d["cls"] == "pedestrian":
                    obj = PedestrianTarget(self._world, d["bp"], dsp,
                                           walk_speed=random.uniform(1.4, 2.4))
                else:
                    obj = VehicleTarget(self._world, d["bp"], dsp, carla_client=self._client,
                                        color=COLOR_RGB.get(d.get("color") or ""),
                                        role_name="distractor")
                self._distractors.append(obj)
                self._distractor_meta.append({
                    "id": obj.id, "bp": d["bp"], "color": d.get("color"),
                    "desc": d["desc"], "similar": d.get("similar", False), "cls": d["cls"],
                })
            except RuntimeError:
                pass

    def _spawn_uav(self, uav_cfg: dict) -> None:
        start = uav_cfg.get("start", {})
        t = self._target.transform
        target_loc = t.location
        target_yaw_rad = np.radians(t.rotation.yaw)

        # Oblique episodes need more standoff so the camera's shallower depression aims
        # at the target (depression = atan(alt/behind) ≈ |pitch|); else it drops off-screen.
        if getattr(self, "_camera_oblique", False):
            ob = self._config.get("camera", {}).get("oblique", {})
            behind_dist = random.uniform(*ob.get("behind_range", [22, 30]))
        else:
            behind_dist = random.uniform(*start.get("distance_range", [12, 22]))
        lateral_offset = random.uniform(*start.get("lateral_offset_range", [6, 14]))
        lateral_side = random.choice([-1, 1])
        alt = random.uniform(*self._tier_alt)   # tier-controlled tracking altitude

        fwd_x, fwd_y = np.cos(target_yaw_rad), np.sin(target_yaw_rad)
        right_x, right_y = -fwd_y, fwd_x
        uav_loc = carla.Location(
            x=target_loc.x - fwd_x * behind_dist + right_x * lateral_offset * lateral_side,
            y=target_loc.y - fwd_y * behind_dist + right_y * lateral_offset * lateral_side,
            z=target_loc.z + alt,
        )
        look_yaw = np.degrees(np.arctan2(target_loc.y - uav_loc.y, target_loc.x - uav_loc.x))
        uav_transform = carla.Transform(uav_loc, carla.Rotation(pitch=-10, yaw=look_yaw, roll=0))
        self.drone = KinematicDrone(self._world, uav_transform,
                                    camera_pitch=getattr(self, "_camera_pitch", -50.0))

    def _distance_category(self) -> str:
        """Map the UAV↔target horizontal standoff to close / suitable / long, so the
        language distance constraint matches the actual tracking distance."""
        try:
            u = self.drone.transform.location
            t = self._target.transform.location
            d = ((u.x - t.x) ** 2 + (u.y - t.y) ** 2) ** 0.5
        except (RuntimeError, AttributeError):
            return "suitable"
        if d <= 16.0:
            return "close"
        if d <= 26.0:
            return "suitable"
        return "long"

    # ------------------------------------------------------------------
    # Per-step API
    # ------------------------------------------------------------------

    def get_target_state(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        assert self._target is not None
        return self._target.get_state()

    def retarget(self, k: int, tick: int | None = None) -> bool:
        """1003_dynamic-retarget-design.md §2.2 — mid-episode re-designation: swap the TARGET
        with distractor slot `k`. Slot `k` keeps its scene-design meta (similar/desired_off —
        those describe the SLOT, not the occupying vehicle); only the two vehicle objects
        swap, so get_distractor_states()/Candidate.idx for every OTHER slot is untouched and
        downstream (candidates.py build_candidates, scorer) needs no changes: is_target is
        always derived from whichever object self._target currently points to. Returns False
        if k is out of range or either side is already gone (actor destroyed)."""
        if self._target is None or not (0 <= k < len(self._distractors)):
            return False
        new_t = self._distractors[k]
        if new_t is None:
            return False
        old_t = self._target
        self._target, self._distractors[k] = new_t, old_t
        self._retarget_log.append({"tick": tick, "slot": k})
        return True

    def recycle_lost_lookalikes(self) -> None:
        """Smooth off-screen recycle (spec §4d): keep >=2 SIMILAR look-alikes co-visible
        with the target by teleporting one that has drifted FAR off-camera back into view
        AHEAD of the target. Robustness guards (the two failure modes that disabled the old
        version): only OFF-screen actors move (real-camera projection → no visible jump);
        the destination slot is on a DRIVING lane and validated CLEAR of every nearby
        vehicle (no collision). Only fires when <2 look-alikes are currently on-screen, so
        it does no work (and no world query) while co-visibility is already fine. Keeping
        the look-alikes near the target also keeps the framing centroid near the target →
        the target stays in-frame (fixes the earlier off-screen instability)."""
        cfg = self._config.get("scene", {}).get("covisibility", {})
        if not cfg.get("enabled", True) or self._target is None or self.drone is None:
            return
        try:
            cam_t = self.drone.get_camera_transform()
            tloc = self._target.transform.location
        except RuntimeError:
            return
        tgt_wp = self._map.get_waypoint(tloc)
        if tgt_wp is None:
            return
        W, H = self._config.get("output", {}).get("rgb_resolution", [336, 336])
        _fov = float(self._config.get("output", {}).get("fov", 90.0))   # §10: match sensor FOV
        M = np.array(cam_t.get_inverse_matrix())
        focal = W / (2.0 * np.tan(np.radians(_fov) / 2.0))

        def _visible(p) -> bool:
            q = M @ np.array([p[0], p[1], p[2], 1.0])
            if q[0] <= 0.1:
                return False
            u = focal * (q[1] / q[0]) + W / 2.0
            v = focal * (-q[2] / q[0]) + H / 2.0
            return 0 <= u < W and 0 <= v < H

        far = float(cfg.get("recycle_radius", 55.0))
        clr = float(cfg.get("min_clearance", 6.0))
        tpos = np.array([tloc.x, tloc.y, tloc.z])
        # Cheap pass: how many similar are on-screen; which off-screen ones drifted far.
        on_screen, needy = 0, []
        for obj, meta in zip(self._distractors, self._distractor_meta):
            if not meta.get("similar"):
                continue
            try:
                p, _, _, _ = obj.get_state()
            except RuntimeError:
                continue
            if _visible(p):
                on_screen += 1
            elif np.linalg.norm(np.asarray(p)[:2] - tpos[:2]) > far:
                v_ = getattr(obj, "_vehicle", None)          # just relocated by _unjam_step → not back in front yet
                if v_ is None or getattr(self, "_kin_quarantine", {}).get(v_.id, -1) < getattr(self, "_kin_tick", 0):
                    needy.append(obj)
        if on_screen >= 2 or not needy:
            return                               # co-visibility already fine → do nothing

        # Candidate driving lanes = target lane + same-direction neighbours.
        lanes = [tgt_wp]
        for g in ("get_left_lane", "get_right_lane"):
            try:
                c = getattr(tgt_wp, g)()
            except RuntimeError:
                c = None
            if (c is not None and c.lane_type == carla.LaneType.Driving
                    and c.lane_id * tgt_wp.lane_id > 0):
                lanes.append(c)
        # Occupancy = every vehicle within 45 m of the target (collision check).
        occupied = [tpos]
        try:
            for a in self._world.get_actors().filter("vehicle.*"):
                loc = a.get_location()
                if abs(loc.x - tloc.x) < 45 and abs(loc.y - tloc.y) < 45:
                    occupied.append(np.array([loc.x, loc.y, loc.z]))
        except RuntimeError:
            pass

        random.shuffle(needy)
        for obj in needy[:max(0, 2 - on_screen)]:
            actor = getattr(obj, "_vehicle", None)
            if actor is None:
                continue
            for _try in range(8):                # find a clear on-road slot near the target, in view
                wp = random.choice(lanes)
                off = random.uniform(-14.0, 14.0)
                for _ in range(int(abs(off) / 8)):
                    nxt = wp.next(8.0) if off > 0 else wp.previous(8.0)
                    if not nxt:
                        break
                    wp = nxt[0]
                loc = wp.transform.location
                slot = np.array([loc.x, loc.y, loc.z])
                if all(np.linalg.norm(slot[:2] - o[:2]) > clr for o in occupied):
                    try:
                        actor.set_transform(carla.Transform(
                            carla.Location(loc.x, loc.y, loc.z + 0.3), wp.transform.rotation))
                        actor.set_target_velocity(carla.Vector3D(0, 0, 0))
                        actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                        occupied.append(slot)
                    except RuntimeError:
                        pass
                    break                        # placed (or failed) → next needy actor

    def station_keep_lookalikes(self) -> None:
        """§8 depth decorrelation: hold each SIMILAR look-alike at its assigned signed
        along-road offset relative to the target (some BEHIND → nearer camera, some ahead →
        farther) via smooth TM speed nudges, so the target's depth RANK stays ~uniform
        instead of systematically nearest. Speed-based, not teleport: a teleported decoy is
        immediately driven forward again by autopilot; a proportional speed offset keeps it
        stationed. No-op for distractors without a desired_off (fillers)."""
        if self._target is None or self._maneuver_injector is None:
            return
        tm = getattr(self._maneuver_injector, "_tm", None)
        if tm is None:
            return
        try:
            tloc = self._target.transform.location
            tyaw = np.radians(self._target.transform.rotation.yaw)
        except (RuntimeError, AttributeError):
            return
        fwd = np.array([np.cos(tyaw), np.sin(tyaw)])   # target's forward (+ = ahead / farther)
        tp = np.array([tloc.x, tloc.y])
        for obj, meta in zip(self._distractors, self._distractor_meta):
            desired = meta.get("desired_off")
            if desired is None:                 # only stationed fillers carry a desired_off
                continue
            veh = getattr(obj, "_vehicle", None)
            if veh is None:
                continue
            try:
                p = obj.get_state()[0]
            except (RuntimeError, AttributeError):
                continue
            off = float(np.dot(np.array([p[0], p[1]]) - tp, fwd))   # signed along-road offset
            err = off - float(desired)          # >0 → ahead of slot → slow down; <0 → speed up
            # TM convention: +% = drive that % BELOW the speed limit (slower). Proportional,
            # clamped so it never stops dead or races off.
            pct = float(np.clip(err * 5.0, -45.0, 70.0))
            try:
                tm.vehicle_percentage_speed_difference(veh, pct)
            except (RuntimeError, AttributeError):
                pass

    def get_distractor_states(self) -> np.ndarray:
        """Return (D, 6) array: [x, y, z, vx, vy, vz] for each distractor (order = meta)."""
        out = []
        for d in self._distractors:
            try:
                pos, vel, _, _ = d.get_state()
                out.append([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])
            except (RuntimeError, AttributeError):
                out.append([np.nan] * 6)
        arr = np.array(out, dtype=np.float32) if out else np.zeros((0, 6), dtype=np.float32)
        self._distractor_states = arr        # cache for _cluster_centroid (same step)
        return arr

    def get_distractor_similar(self) -> np.ndarray:
        """(D,) bool: is each distractor a look-alike of the target? Same order as
        get_distractor_states / distractors/positions. Lets (d)-evaluation slice
        look-alike vs plain distractors per-FRAME (§4.1 request)."""
        return np.array([bool(m.get("similar", False)) for m in self._distractor_meta],
                        dtype=np.uint8)

    def step_target(self, dt: float = 0.1) -> None:
        self._unjam_step()
        if self._target is not None and hasattr(self._target, "step"):
            try:
                self._target.step(dt)
            except RuntimeError:
                pass
        for d in self._distractors:      # a distractor may be respawn_dormant-destroyed
            if hasattr(d, "step"):
                try:
                    d.step(dt)
                except RuntimeError:
                    pass
        for pw in self._ambient_walkers:   # drive background pedestrians
            try:
                pw.step(dt)
            except RuntimeError:
                pass
        # Keep background pedestrians near the moving target: every 25 frames, move
        # any that have fallen far behind (off-screen) to navmesh near the target.
        self._amb_tick = getattr(self, "_amb_tick", 0) + 1
        if self._ambient_walkers and self._target is not None and self._amb_tick % 25 == 0:
            try:
                tloc = self._target.transform.location
            except RuntimeError:
                return
            far = []
            for pw in self._ambient_walkers:
                try:
                    if pw.transform.location.distance(tloc) > 120.0:
                        far.append(pw)
                except RuntimeError:
                    pass
            for pw, loc in zip(far, self._nav_near(tloc, 80.0, len(far), attempts=60)):
                a = getattr(pw, "_walker", None)
                if a is not None:
                    try:
                        a.set_transform(carla.Transform(
                            carla.Location(loc.x, loc.y, loc.z + 1.0)))
                    except RuntimeError:
                        pass

    def is_target_turning(self) -> bool:
        if self._target is not None and self._target_type == "vehicle":
            return getattr(self._target, "is_turning", False)
        return False

    def check_obstacles(self, action, uav_pos, uav_yaw, timestamp) -> tuple[tuple, Any]:
        ev = None
        if self._obstacle_avoider is not None:
            action, ev = self._obstacle_avoider.check(action, uav_pos, timestamp)
        if self._config.get("uav", {}).get("geometry_avoidance", False):
            action, gev = self._geometry_avoid(action, uav_pos, timestamp, uav_yaw)
            ev = ev or gev
        return action, ev

    def _geometry_avoid(self, action, uav_pos, timestamp, uav_yaw=0.0):
        """v6: the kinematic drone has no collision and the altitude floor assumes <30 m buildings, so
        it could fly THROUGH high-rises (camera inside geometry → black / wall frames). Raycast the
        horizontal path ~3 s ahead (at camera height and 3 m below); if blocked, stop horizontally and
        climb; never descend onto a roof directly below."""
        from carla_uav.obstacle_avoidance import ObstacleEvent
        L = carla.CityObjectLabel
        block = {getattr(L, n) for n in ("Buildings", "Walls", "Fences", "Poles", "Vegetation", "Bridge",
                                         "Static", "GuardRail", "TrafficLight", "TrafficSigns") if hasattr(L, n)}
        dx, dy, dz, dyaw = action
        p = carla.Location(float(uav_pos[0]), float(uav_pos[1]), float(uav_pos[2]))

        def hit(b):
            try:
                return any(h.label in block for h in self._world.cast_ray(p, b))
            except RuntimeError:
                return False
        # clearance ring: 8 horizontal 5 m rays at camera height and 2 m below — structures beside the
        # drone (e.g. an elevated highway it flies parallel to) would otherwise slowly swallow the camera
        push = np.zeros(2)
        for k in range(8):
            ang = k * math.pi / 4.0
            ux, uy = math.cos(ang), math.sin(ang)
            for zoff in (0.0, -2.0):
                if hit(carla.Location(p.x + 5.0 * ux, p.y + 5.0 * uy, p.z + zoff)):
                    push -= (ux, uy)
                    break
        if np.any(push != 0.0):
            n = float(np.linalg.norm(push)) or 1.0
            return ((0.5 * push[0] / n), (0.5 * push[1] / n), 0.6, dyaw), ObstacleEvent(timestamp, "geom_clear", 5.0)
        hz = math.hypot(dx, dy)
        # camera optical axis: never let the camera come within 12 m of a surface it looks at (a roof or
        # facade filling the frame from 2-5 m away is not a realistic aerial view)
        pitch = math.radians(float(getattr(self, "_camera_pitch", -50.0)))
        yaw = math.radians(float(uav_yaw))
        ax = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
        if hit(carla.Location(p.x + 12.0 * ax[0], p.y + 12.0 * ax[1], p.z + 12.0 * ax[2])):
            return (0.0, 0.0, 0.8, dyaw), ObstacleEvent(timestamp, "geom_axis_climb", 12.0)
        if hz > 1e-3:
            look = max(8.0, 30.0 * hz)
            ux, uy = dx / hz, dy / hz
            for zoff in (0.0, -3.0):
                if hit(carla.Location(p.x + ux * look, p.y + uy * look, p.z + zoff)):
                    return (0.0, 0.0, 0.8, dyaw), ObstacleEvent(timestamp, "geom_climb", look)
        if dz < 0 and hit(carla.Location(p.x, p.y, p.z + dz - 4.0)):
            return (dx, dy, 0.0, dyaw), ObstacleEvent(timestamp, "geom_hold_alt", -dz)
        return action, None

    def expert_action(
        self, target_pos: np.ndarray, uav_pos: np.ndarray, uav_yaw: float,
        target_visible: bool, timestamp: float,
    ) -> tuple[tuple[float, float, float, float], Any]:
        assert self.expert is not None
        # L2 (position-decorrelation) + RECOVERY:
        #  - WHILE the target is co-visible with the group, servo the CLUSTER CENTROID so
        #    the target is a random group member (not the framed reference → no position
        #    leak).
        #  - ONCE the target has drifted OFF-screen, servo the TRUE target instead, so the
        #    UAV flies to re-acquire it (aerial tracking has no road constraint → it cuts
        #    straight to where the target re-appears). The episode then continues through
        #    the loss and generates recovery demonstrations, rather than being truncated.
        servo_pos = target_pos
        # 1020 stress scenario: the expert never switches to TARGET-anchored recovery framing (it would re-centre the
        # true target exactly at the re-capture moment → a GT leak through the camera); it always servos the
        # {target + similar} centroid, so the target re-enters as a random member of its clone group.
        if getattr(self, "_no_target_recovery", False):
            self._target_in_frame = True
            self._recovery_mode = False
        # During a deliberate loss window the expert is flown BLIND: target_visible=False
        # makes the PID dead-reckon (predict) instead of re-centering, so the forced-evading
        # target genuinely leaves frame and stays off — the predict-intercept loss arc.
        if target_visible and target_pos is not None and getattr(self, "_frame_centroid", False):
            # HYSTERESIS on the centroid↔recovery switch: once the target is lost, stay in
            # target-recovery framing until it is STABLY back (in-frame ≥8 consecutive frames).
            # Without this, a target flickering at the frame edge flips the servo setpoint
            # between the cluster centroid and the (far) target every frame → the camera shakes.
            if not getattr(self, "_target_in_frame", True):
                self._recovery_mode = True
                self._reacq = 0
            elif getattr(self, "_recovery_mode", False):
                self._reacq = getattr(self, "_reacq", 0) + 1
                if self._reacq >= 8:
                    self._recovery_mode = False
            if not getattr(self, "_recovery_mode", False):
                servo_pos = self._cluster_centroid(np.asarray(target_pos, dtype=float))
        # §8 (redo): TRAIL the target by a per-episode random HORIZONTAL standoff so the
        # target sits at a random forward/depth position in the frame (near the bottom when
        # the UAV trails close, higher/farther when it trails far) → the ground BETWEEN the
        # UAV and the target holds distractors NEARER than the target → the target's depth
        # RANK is decorrelated (was systematically nearest because the UAV hovered ~over the
        # target). The camera still FACES the target (look_pos), only its POSITION trails.
        # Skip during loss/recovery so re-acquisition still cuts straight to the target.
        pos_setpoint, look_pos = servo_pos, None
        standoff = getattr(self, "_tracking_standoff", 0.0)
        aim_frac = getattr(self, "_aim_frac", 0.0)
        if (target_visible and target_pos is not None
                and not getattr(self, "_recovery_mode", False)
                and (standoff > 0.0 or aim_frac > 0.0)):
            try:
                tyaw = np.radians(self._target.transform.rotation.yaw)
                fwd = np.array([np.cos(tyaw), np.sin(tyaw)])
                aim = np.asarray(servo_pos, dtype=float).copy()
                if aim_frac > 0.0:
                    # §9: LATERAL-only AIM offset (perpendicular to the target's heading) → the
                    # target's horizontal image position (u) spreads like a distractor's, killing
                    # the 'most-central' shortcut. Lateral ONLY (not forward/back) so it does NOT
                    # move the target in DEPTH → leaves the §8 `nearest` decorrelation intact
                    # (a forward/back aim component pushed the target far → nearest collapsed).
                    # Vertical (v) spread is already provided by the per-episode standoff.
                    # Amplitude ∝ viewing distance → constant image spread, target stays in frame.
                    depth = float(np.linalg.norm(aim - np.asarray(uav_pos, dtype=float)))
                    perp = np.array([-fwd[1], fwd[0]])
                    lat = aim_frac * depth * np.sin(2 * np.pi * self._aim_fx * timestamp + self._aim_px)
                    aim[:2] = aim[:2] + lat * perp
                look_pos = aim                                # camera FACES the shifted aim point
                pos_setpoint = aim.copy()
                pos_setpoint[:2] = pos_setpoint[:2] - standoff * fwd   # §8: trail the aim point
            except (RuntimeError, AttributeError):
                pos_setpoint, look_pos = servo_pos, None
        action = self.expert.act(pos_setpoint, uav_pos, uav_yaw, target_visible, timestamp,
                                 look_pos=look_pos)
        pert_event = None
        if self._perturbation_injector is not None and target_pos is not None:
            action, pert_event = self._perturbation_injector.maybe_perturb(
                action, uav_pos, target_pos, timestamp)
        return action, pert_event

    def expert_look_away(self, target_pos: np.ndarray, uav_pos: np.ndarray, uav_yaw: float,
                         timestamp: float) -> tuple[float, float, float, float]:
        """1020 stress scenario E3 (group out-of-frame): during a scheduled loss window the expert HOVERS and turns
        the camera away from the {target + similar} cluster by a per-window angle (120-180 deg, random side), so the
        whole group leaves the frame for the window; afterwards expert_action brings the camera back to the cluster
        centroid. The camera motion depends only on the window schedule and the cluster, never on which member is
        the target — the recorded observation stream stays identity-independent."""
        uav = np.asarray(uav_pos, dtype=float)
        c = self._cluster_centroid(np.asarray(target_pos, dtype=float))
        d = c[:2] - uav[:2]
        base = np.arctan2(d[1], d[0]) if np.hypot(*d) > 1e-3 else np.radians(uav_yaw)
        if not getattr(self, "_look_away_on", False):         # one angle per window, drawn at its first tick
            self._look_away_on = True
            self._look_away_ang = np.radians(random.uniform(120.0, 180.0)) * random.choice([-1.0, 1.0])
        ang = base + self._look_away_ang
        look = uav.copy()
        look[:2] = uav[:2] + 40.0 * np.array([np.cos(ang), np.sin(ang)])
        hold = np.array([uav[0], uav[1], float(np.asarray(target_pos, dtype=float)[2])])   # hover, keep altitude
        return self.expert.act(hold, uav, uav_yaw, True, timestamp, look_pos=look)

    def end_look_away(self) -> None:
        self._look_away_on = False

    def check_maneuver(self, timestamp: float) -> Any | None:
        if self._maneuver_injector is not None and self._target_type == "vehicle":
            return self._maneuver_injector.maybe_inject(self._target._vehicle, timestamp)
        return None

    # -- (c) deliberate loss-event hooks (driven by the recorder loop) --
    def in_loss_window(self, elapsed: float) -> bool:
        """True while inside a scheduled loss window → fly the expert blind (predict)."""
        sched = getattr(self, "_loss_scheduler", None)
        return bool(sched and sched.in_predict(elapsed))

    def loss_window_onset(self, elapsed: float) -> float | None:
        """Window duration once at onset (→ force the target to deviate); else None."""
        sched = getattr(self, "_loss_scheduler", None)
        return sched.onset(elapsed) if sched else None

    def _target_tm(self):
        veh = getattr(self._target, "_vehicle", None) if self._target_type == "vehicle" else None
        tm = getattr(self._maneuver_injector, "_tm", None) if self._maneuver_injector else None
        return veh, tm

    def force_target_evade(self, elapsed: float, duration: float) -> None:
        """§11: at loss-window onset, force the target to TURN at the upcoming junctions
        (set_route) so it leaves the UAV's forward view. A pure forward sprint does NOT work
        (target just gets far & small while staying in the forward frame, and the gap becomes
        un-re-acquirable); a TURN makes the target go off-frame while staying CLOSE, so the
        intercepting UAV must cut across and can re-acquire → genuine long loss + intercept."""
        veh, tm = self._target_tm()
        if veh is None or tm is None:
            return
        try:
            tm.set_route(veh, [random.choice(["Left", "Right"]) for _ in range(4)])
            tm.set_desired_speed(veh, self._evade_kmh())   # moderate → stays close (re-acquirable), not sprint-away
        except (RuntimeError, AttributeError):
            pass

    def sustain_target_evade(self, elapsed: float) -> None:
        """§11: keep the target moving at a moderate speed through its forced turns for the
        whole window (TM can otherwise reset it at a light), so it stays off the forward view
        while the expert flies to INTERCEPT the true target."""
        veh, tm = self._target_tm()
        if veh is None or tm is None:
            return
        try:
            tm.set_desired_speed(veh, self._evade_kmh())
        except (RuntimeError, AttributeError):
            pass

    def restore_target_speed(self) -> None:
        """§11: at loss-window end, drop the target back to normal traffic speed so the
        intercepting UAV catches up and re-acquires."""
        veh, tm = self._target_tm()
        if veh is None or tm is None:
            return
        try:
            # legacy: 9.0 — TM.set_desired_speed takes km/h, i.e. 2.5 m/s for the rest of the episode. In kinematics
            # mode the target returns to its own episode speed instead.
            base = getattr(self, "_kin_target_speed", None)
            tm.set_desired_speed(veh, self._kmh(base) if (self._kin() and base) else 9.0)
        except (RuntimeError, AttributeError):
            pass

    def loss_window_offset(self, elapsed: float) -> bool:
        """True once when a loss window just ended (target re-appearing)."""
        sched = getattr(self, "_loss_scheduler", None)
        return bool(sched and sched.offset(elapsed))

    def force_covisible_lookalikes(self, min_n: int = 2) -> None:
        """§11: at target re-appearance, GUARANTEE >=min_n similar look-alikes co-visible near
        the target, so re-acquisition must use LANGUAGE (not 'whichever car just appeared').
        Teleports off-screen/far similars to clear, on-road, IN-FRAME slots beside the target."""
        if self._target is None or self.drone is None:
            return
        try:
            cam_t = self.drone.get_camera_transform()
            tloc = self._target.transform.location
        except (RuntimeError, AttributeError):
            return
        tgt_wp = self._map.get_waypoint(tloc)
        if tgt_wp is None:
            return
        W, H = self._config.get("output", {}).get("rgb_resolution", [336, 336])
        _fov = float(self._config.get("output", {}).get("fov", 90.0))
        M = np.array(cam_t.get_inverse_matrix())
        focal = W / (2.0 * np.tan(np.radians(_fov) / 2.0))

        def _visible(p) -> bool:
            q = M @ np.array([p[0], p[1], p[2], 1.0])
            if q[0] <= 0.1:
                return False
            u = focal * (q[1] / q[0]) + W / 2.0
            v = focal * (-q[2] / q[0]) + H / 2.0
            return 0 <= u < W and 0 <= v < H

        tpos = np.array([tloc.x, tloc.y, tloc.z])
        on_screen, movable = 0, []
        for obj, meta in zip(self._distractors, self._distractor_meta):
            if not meta.get("similar"):
                continue
            try:
                p, _, _, _ = obj.get_state()
            except RuntimeError:
                continue
            if _visible(p):
                on_screen += 1
            else:
                movable.append(obj)
        need = max(0, min_n - on_screen)
        if need <= 0 or not movable:
            return
        lanes = [tgt_wp]
        for g in ("get_left_lane", "get_right_lane"):
            try:
                c = getattr(tgt_wp, g)()
            except RuntimeError:
                c = None
            if c is not None and c.lane_type == carla.LaneType.Driving and c.lane_id * tgt_wp.lane_id > 0:
                lanes.append(c)
        occupied = [tpos]
        try:
            for a in self._world.get_actors().filter("vehicle.*"):
                loc = a.get_location()
                if abs(loc.x - tloc.x) < 45 and abs(loc.y - tloc.y) < 45:
                    occupied.append(np.array([loc.x, loc.y, loc.z]))
        except RuntimeError:
            pass
        random.shuffle(movable)
        for obj in movable[:need]:
            actor = getattr(obj, "_vehicle", None)
            if actor is None:
                continue
            for _try in range(10):
                wp = random.choice(lanes)
                off = random.uniform(-12.0, 12.0)
                for _ in range(int(abs(off) / 8)):
                    nxt = wp.next(8.0) if off > 0 else wp.previous(8.0)
                    if not nxt:
                        break
                    wp = nxt[0]
                loc = wp.transform.location
                slot = np.array([loc.x, loc.y, loc.z])
                if _visible(slot) and all(np.linalg.norm(slot[:2] - o[:2]) > 6.0 for o in occupied):
                    try:
                        actor.set_transform(carla.Transform(
                            carla.Location(loc.x, loc.y, loc.z + 0.3), wp.transform.rotation))
                        actor.set_target_velocity(carla.Vector3D(0, 0, 0))
                        occupied.append(slot)
                    except RuntimeError:
                        pass
                    break

    # ------------------------------------------------------------------ 2026-10-09 kinematics / anti-gridlock
    def _kin(self):
        """scene.kinematics config (dict) or None (legacy behaviour everywhere)."""
        return (self._config.get("scene", {}) or {}).get("kinematics") or None

    @staticmethod
    def _kmh(mps: float) -> float:
        return float(mps) * 3.6               # TM.set_desired_speed takes km/h

    def _evade_kmh(self) -> float:
        kin = self._kin()
        return self._kmh(kin["evade_speed_mps"]) if (kin and kin.get("evade_speed_mps")) else 16.0

    def _apply_kinematics(self) -> None:
        """Target: TM desired speed ~ U(actor_speed_mps); vehicle distractors: the target's ± jitter (clipped to the
        range) so the look-alike convoy stays together. set_desired_speed replaces the speed-limit percentage given
        at spawn (TM keeps whichever was set last; nothing else re-sets these in this scenario: maneuvers are off,
        station_keep_lookalikes is unused)."""
        self._kin_still, self._kin_tick, self._kin_relocated, self._kin_lane_kicks = {}, 0, 0, 0
        self._kin_events = []                 # unjam relocations → episode metadata (recorder)
        self._kin_quarantine = {}             # relocated actor id → tick until which recycle must not bring it back
        self._kin_target_speed = None
        kin = self._kin()
        if not kin or self._target_type != "vehicle" or self._target is None:
            return
        tm = self._client.get_trafficmanager(_tm_port())
        lo, hi = (float(x) for x in kin.get("actor_speed_mps", [10.0, 12.0]))
        jit = float(kin.get("distractor_jitter_mps", 0.5))
        base = random.uniform(lo, hi)
        speeds = [base] + [float(np.clip(base + random.uniform(-jit, jit), lo, hi)) for _ in self._distractors]
        if kin.get("clone_speed_offset", False):
            # 1026 stress_v2: set_desired_speed overrides the E1 percentage offsets given at spawn → re-apply them here:
            # a clone drives base · (1 − Δ/100), Δ = its spawn speed_diff − the target's (sign random, 12-22 %)
            tsd = getattr(self, "_target_speed_diff", None)
            for n, meta in enumerate(self._distractor_meta[:len(self._distractors)]):
                if meta.get("clone") and meta.get("speed_diff") is not None and tsd is not None:
                    speeds[n + 1] = float(base * (1.0 - (float(meta["speed_diff"]) - float(tsd)) / 100.0))
        for obj, mps in zip([self._target] + list(self._distractors), speeds):
            v = getattr(obj, "_vehicle", None)
            if v is None:
                continue
            try:
                tm.set_desired_speed(v, self._kmh(mps))
            except (RuntimeError, AttributeError):
                pass
        self._kin_target_speed = base
        print(f"  [kinematics] target {base:.1f} m/s, distractors {lo:.0f}-{hi:.0f} m/s (±{jit} around the target)")

    def _unjam_step(self) -> None:
        """No single stuck vehicle may freeze the scene (scene.kinematics.unjam). Every 10 steps (1 s at 10 Hz) on one
        snapshot of all vehicles:
          · "still" = speed < still_mps and NOT part of a queue waiting at a red light (the lead chain of up to 8 cars
            in the same lane, ≤ 15 m apart, ends at a car that is_at_traffic_light with state Red); counted per actor;
          · target / vehicle distractor still >= actor_still_s → TM force_lane_change every 4 s, sides alternating;
            still >= blocker_still_s → the vehicle right in front of it in its lane (≤ 15 m; else the nearest stopped
            car in a ±4 m corridor ≤ 12 m — junction box) is relocated: an ambient car anywhere 150-300 m away, a
            distractor too (recycle_lost_lookalikes later brings a look-alike back near the target off-camera);
          · ambient car within radius_m of the target still >= ambient_still_s → relocated, off-camera ones first;
            one visible in the frame only once still >= ambient_visible_still_s.
        Relocation = set_transform to a free spawn point (no vehicle within 10 m) + zero linear/angular velocity;
        TM keeps driving it (the same mechanism as its own dormant respawn). Last resort (recorder): target moved < 5 m
        in truncate_stuck_s → episode truncated → fails QC. Counters: self._kin_relocated / self._kin_lane_kicks."""
        kin = self._kin()
        un = (kin or {}).get("unjam")
        if not un or self._target is None or getattr(self, "_target_type", None) != "vehicle":
            return
        self._kin_tick = getattr(self, "_kin_tick", 0) + 1
        if self._kin_tick % 10:
            return
        try:
            tloc = self._target.transform.location
            vehicles = list(self._world.get_actors().filter("vehicle.*"))
        except (RuntimeError, AttributeError):
            return
        tm = self._client.get_trafficmanager(_tm_port())
        still_mps, radius = float(un.get("still_mps", 0.3)), float(un.get("radius_m", 80.0))
        a_still, b_still = int(un.get("actor_still_s", 12)), int(un.get("blocker_still_s", 20))
        amb_still, amb_vis_still = int(un.get("ambient_still_s", 25)), int(un.get("ambient_visible_still_s", 40))
        red = carla.TrafficLightState.Red
        snap = {}                                            # id → (actor, x, y, yaw_rad, speed, z)
        for a in vehicles:
            try:
                tr, vel = a.get_transform(), a.get_velocity()
                snap[a.id] = (a, tr.location.x, tr.location.y, np.radians(tr.rotation.yaw), vel.length(), tr.location.z)
            except RuntimeError:
                pass

        ids = list(snap)
        P = np.array([[snap[i][1], snap[i][2]] for i in ids]).reshape(-1, 2)
        SP = np.array([snap[i][4] for i in ids])

        def ahead(aid, lat_max=1.8, fwd_max=15.0, need_stopped=False):
            """nearest vehicle in front of aid: 0.5 < forward ≤ fwd_max, |lateral| < lat_max (its own lane)"""
            _, x, y, yaw, _, _ = snap[aid]
            r = P - np.array([x, y])
            df = r[:, 0] * np.cos(yaw) + r[:, 1] * np.sin(yaw)
            dl = np.abs(r[:, 0] * np.sin(yaw) - r[:, 1] * np.cos(yaw))
            ok = (df > 0.5) & (df <= fwd_max) & (dl < lat_max)
            if need_stopped:
                ok &= SP < still_mps
            if not ok.any():
                return None
            k = np.flatnonzero(ok)
            return ids[int(k[np.argmin(df[k])])]

        def at_red(aid):
            try:
                a = snap[aid][0]
                return bool(a.is_at_traffic_light()) and a.get_traffic_light_state() == red
            except RuntimeError:
                return False

        def queued_at_red(aid):
            cur = aid
            for _ in range(8):
                if at_red(cur):
                    return True
                nxt = ahead(cur)
                if nxt is None:
                    return False
                cur = nxt
            return False

        def tick_still(aid):
            n = self._kin_still.get(aid, 0)
            if snap[aid][4] >= still_mps:
                n = 0                                        # real motion → reset
            elif not queued_at_red(aid):
                n += 1                                       # queued at a red → paused, not reset (codex R2-1)
            self._kin_still[aid] = n
            return n

        cam = None
        try:
            cam_t = self.drone.get_camera_transform() if self.drone is not None else None
            if cam_t is not None:
                W, H = self._config.get("output", {}).get("rgb_resolution", [336, 336])
                fov = float(self._config.get("output", {}).get("fov", 90.0))
                cam = (np.array(cam_t.get_inverse_matrix()), W, H, W / (2.0 * np.tan(np.radians(fov) / 2.0)))
        except RuntimeError:
            cam = None

        def visible(aid):
            if cam is None:
                return True                                  # unknown → treat as visible (conservative)
            M, W, H, foc = cam
            _, x, y, _, _, z = snap[aid]
            q = M @ np.array([x, y, z, 1.0])
            if q[0] <= 0.1:
                return False
            u, v = foc * (q[1] / q[0]) + W / 2.0, foc * (-q[2] / q[0]) + H / 2.0
            return -20 <= u < W + 20 and -20 <= v < H + 20

        free_pts = None

        def relocate(aid, why):
            nonlocal free_pts
            if free_pts is None:
                occ = [(sx, sy) for (_, sx, sy, _, _, _) in snap.values()]
                free_pts = [sp for sp in self._map.get_spawn_points()
                            if 150.0 <= sp.location.distance(tloc) <= 300.0
                            and all(np.hypot(sp.location.x - ox, sp.location.y - oy) > 10.0 for ox, oy in occ)]
                random.shuffle(free_pts)
            if not free_pts:
                return False
            a, sp, vis = snap[aid][0], free_pts.pop(), visible(aid)
            try:
                a.set_transform(sp)
                a.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                a.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            except RuntimeError:
                return False
            self._kin_still[aid] = 0
            if aid in dis_ids and self._route_cfg():
                self._reroute_after_relocation(a)          # 1026 stress_v2: its planned path is stale now
            self._kin_relocated += 1
            self._kin_events.append({"tick": self._kin_tick, "kind": why, "actor": aid, "in_view": bool(vis)})
            self._kin_quarantine[aid] = self._kin_tick + 10 * int(un.get("quarantine_s", 30))   # codex R2-2
            print(f"  [unjam] relocated {why} {aid} (in view: {vis}); total relocated {self._kin_relocated}, "
                  f"lane kicks {self._kin_lane_kicks}", flush=True)
            return True

        tgt_v = getattr(self._target, "_vehicle", None)
        tgt_id = tgt_v.id if tgt_v is not None else None
        dis_ids = {v.id for v in (getattr(o, "_vehicle", None) for o in self._distractors) if v is not None}
        scene_ids = [i for i in [tgt_id] + sorted(dis_ids) if i is not None and i in snap]
        for aid in scene_ids:                                # target + distractors: lane kick → clear the blocker
            n = tick_still(aid)
            if n >= a_still and (n - a_still) % 4 == 0:
                try:
                    tm.force_lane_change(snap[aid][0], bool(((n - a_still) // 4) % 2))
                    self._kin_lane_kicks += 1
                except RuntimeError:
                    pass
            if n >= b_still and (n - b_still) % 5 == 0:
                blk = ahead(aid) or ahead(aid, lat_max=4.0, fwd_max=12.0, need_stopped=True)
                if blk is not None and blk != tgt_id:           # never teleport the target itself
                    relocate(blk, ("distractor" if blk in dis_ids else "ambient") + f" blocking {aid}")
        amb = set(getattr(self, "_ambient_ids", [])) & set(snap)
        cand = []
        for aid in amb:
            _, x, y, _, _, _ = snap[aid]
            if np.hypot(x - tloc.x, y - tloc.y) > radius:
                self._kin_still[aid] = 0
                continue
            n = tick_still(aid)
            if n >= amb_still:
                vis = visible(aid)
                if not vis or n >= amb_vis_still:
                    cand.append((vis, -n, aid))
        for _, _, aid in sorted(cand):                       # off-camera first, then the longest-stuck
            relocate(aid, "ambient")

    def step_drone(self, action: tuple[float, float, float, float]) -> None:
        assert self.drone is not None
        self.drone.step(*action)

    def cleanup(self) -> None:
        self.drone = None
        if self._target is not None:
            try:
                self._target.destroy()
            except RuntimeError:
                pass
            self._target = None
        for d in self._distractors:
            try:
                d.destroy()
            except RuntimeError:
                pass
        for pw in self._ambient_walkers:
            try:
                pw.destroy()
            except RuntimeError:
                pass
        self._distractors = []
        self._distractor_meta = []
        self._ambient_walkers = []
        self.expert = None
        self.safety = None

    def _destroy_all_traffic(self) -> None:
        """Batch-destroy all vehicles / walkers / walker-controllers on the server."""
        actors = self._world.get_actors()
        victims = [a for a in actors if a.type_id.startswith(
            ("vehicle.", "walker.pedestrian", "controller.ai.walker"))]
        for a in victims:  # stop autopilot/controllers so TM drops them first
            try:
                if a.type_id.startswith("vehicle."):
                    a.set_autopilot(False, _tm_port())  # default port 8000 may be another worker's TM → hang
                elif a.type_id.startswith("controller."):
                    a.stop()
            except RuntimeError:
                pass
        if victims:
            try:
                self._client.apply_batch_sync(
                    [carla.command.DestroyActor(a.id) for a in victims], True)
            except RuntimeError:
                for a in victims:
                    try:
                        a.destroy()
                    except RuntimeError:
                        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_random_weather(self, cfg: dict) -> None:
        weather = carla.WeatherParameters(
            cloudiness=float(random.uniform(*self._cfg_range(cfg, "cloudiness", 0, 90))),
            precipitation=float(random.uniform(*self._cfg_range(cfg, "precipitation", 0, 80))),
            precipitation_deposits=float(random.uniform(*self._cfg_range(cfg, "precipitation_deposits", 0, 80))),
            sun_altitude_angle=float(random.uniform(*self._cfg_range(cfg, "sun_altitude", -30, 60))),
            fog_density=float(random.uniform(*self._cfg_range(cfg, "fog_density", 0, 30))),
            wetness=float(random.uniform(*self._cfg_range(cfg, "wetness", 0, 60))),
        )
        self._world.set_weather(weather)

    @staticmethod
    def _cfg_range(cfg: dict, key: str, lo: float, hi: float) -> tuple[float, float]:
        v = cfg.get(key, [lo, hi])
        if isinstance(v, (int, float)):
            return (float(v), float(v))
        return (float(v[0]), float(v[1]))

    def _get_current_weather_dict(self) -> dict:
        w = self._world.get_weather()
        return {
            "cloudiness": w.cloudiness, "precipitation": w.precipitation,
            "sun_altitude": w.sun_altitude_angle, "fog_density": w.fog_density,
        }
