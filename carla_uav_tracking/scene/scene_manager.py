"""Scene management — actor lifecycle, spawning, and cleanup.

Composes each episode as: 1 language-specified TARGET + N distractors, with
>=2 distractors visually similar to the target (design §5.1/§6.3), co-located on
the same road segment so they are actually co-visible from the UAV. Supports
multiple target classes (car / motorcycle / scooter / bicycle / pedestrian).
"""

from __future__ import annotations

import random
import time
from typing import Any

import carla
import numpy as np

from carla_uav.drone import KinematicDrone
from carla_uav.expert_policy import ExpertPolicy
from carla_uav.safety import SafetyMonitor
from targets.maneuver_injector import ManeuverConfig, ManeuverInjector
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

        # -- Seeding (P1-b): make the whole episode reproducible --
        seed = env_cfg.get("seed", None)
        if seed is not None:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
        tm = self._client.get_trafficmanager(8000)
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
            self._setup_vehicle_scene(plan, scene_cfg)
        else:
            self._setup_pedestrian_scene(plan, scene_cfg)

        self._target_type = "vehicle" if self._target_class in VEHICLE_CLASSES else "pedestrian"
        print(f"  [{self._target_class}/{plan['strategy']}] target={plan['target']['desc']} "
              f"| distractors={len(self._distractors)} (similar={plan['num_similar']})")

        # -- Background traffic (official-style): diverse vehicles + pedestrians --
        self._spawn_ambient_traffic(scene_cfg)

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
            traffic_manager=self._client.get_trafficmanager(8000),
        )
        self._maneuver_injector.reset()

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
            "camera_pitch": float(getattr(self, "_camera_pitch", -50.0)),
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
        if getattr(self, "_occlusion_biased", False):
            anchors = self._occluder_anchors()
            if anchors:
                return random.choice(anchors)
        if self._spawn_points:
            return random.choice(self._spawn_points)
        return carla.Transform(carla.Location(0, 0, 1))

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
            # L1 — target gets a RANDOM slot from the same ± band as the look-alikes.
            self._target = None
            for _try in range(6):
                sp = (anchor if _try == 5 else
                      self._lane_offset_transform(random.choice(lanes),
                                                  random.uniform(-band, band)))
                try:
                    self._target = VehicleTarget(
                        self._world, tgt["bp"], sp, carla_client=self._client,
                        color=COLOR_RGB.get(tgt.get("color") or ""),
                        # SAME speed distribution as distractors so the target does not drift
                        # to a systematic depth within the group (hero-only dynamics leaked).
                        role_name="hero", speed_diff=convoy_speed + random.uniform(-5, 5))
                    break
                except RuntimeError:
                    continue
            for d in plan["distractors"]:
                # Similar look-alikes share the target's ± band (framed-centroid cluster →
                # target is a random member). Plain fillers spread moderately & symmetrically.
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
        self._servo_smooth = None          # reset the camera-setpoint EMA (anti-jitter)
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

    def _spawn_distractor(self, d: dict, sp: "carla.Transform", convoy_speed: float) -> None:
        """Spawn one distractor (vehicle or pedestrian) at sp; skip on spawn collision."""
        try:
            cls = d["cls"]
            if cls in VEHICLE_CLASSES:
                obj = VehicleTarget(
                    self._world, d["bp"], sp, carla_client=self._client,
                    color=COLOR_RGB.get(d.get("color") or ""),
                    role_name="distractor", speed_diff=convoy_speed + random.uniform(-5, 5))
            else:
                obj = PedestrianTarget(self._world, d["bp"], sp, walk_speed=random.uniform(2.5, 3.5))
            self._distractors.append(obj)
            self._distractor_meta.append({
                "id": obj.id, "bp": d["bp"], "color": d.get("color"),
                "desc": d["desc"], "similar": d.get("similar", False), "cls": cls,
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
        ds = getattr(self, "_distractor_states", None)
        if ds is not None and len(ds):
            for row, meta in zip(ds, self._distractor_meta):
                if not meta.get("similar"):
                    continue
                p = np.asarray(row[:3], dtype=float)
                if not np.isnan(p).any():
                    pts.append(p)
        return np.mean(pts, axis=0)

    def _assign_convoy_route(self) -> None:
        """Give the target + vehicle distractors the SAME route (official TM.set_route)
        so they drive together on the same roads → stay co-visible while obeying
        traffic rules, with no teleporting (replaces recycle_lost_lookalikes)."""
        cfg = self._config.get("scene", {}).get("convoy", {})
        if not cfg.get("enabled", True):
            return
        tm = self._client.get_trafficmanager(8000)
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
        n_veh = int(amb.get("vehicles", 0))
        n_walk = int(amb.get("walkers", 0))
        radius = float(amb.get("radius", 90.0))
        if self._target is None:
            return
        try:
            tloc = self._target.transform.location
        except RuntimeError:
            return
        tm = self._client.get_trafficmanager(8000)

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
            batch = []
            for sp in sps[1:n_veh + 1]:    # nearest points (skip target's own)
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
                self._client.apply_batch_sync(batch, True)
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
        M = np.array(cam_t.get_inverse_matrix())
        focal = W / (2.0 * np.tan(np.radians(90.0) / 2.0))

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
                # SYMMETRIC ± offset around the target (behind → lower frame, ahead → upper)
                # so the framing centroid stays ~on the target; wide enough to reliably find
                # a collision-free slot (a too-tight slot fails and co-visibility collapses).
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

    def step_target(self, dt: float = 0.1) -> None:
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
        if self._obstacle_avoider is not None:
            return self._obstacle_avoider.check(action, uav_pos, timestamp)
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
        if (target_pos is not None and getattr(self, "_frame_centroid", False)
                and getattr(self, "_target_in_frame", True)):
            servo_pos = self._cluster_centroid(np.asarray(target_pos, dtype=float))
        # Time-smooth the setpoint (EMA) so the camera does NOT lurch when it jumps — the
        # centroid↔target framing switch on loss, or the centroid shift when a look-alike is
        # teleported by the recycle. This is what fixes the severe shaking during
        # occlusion/off-screen (the UAV eases toward the new setpoint over a few frames).
        if servo_pos is not None:
            sp = np.asarray(servo_pos, dtype=float)
            prev = getattr(self, "_servo_smooth", None)
            servo_pos = sp if prev is None else 0.3 * sp + 0.7 * prev
            self._servo_smooth = np.asarray(servo_pos, dtype=float)
        action = self.expert.act(servo_pos, uav_pos, uav_yaw, target_visible, timestamp)
        pert_event = None
        if self._perturbation_injector is not None and target_pos is not None:
            action, pert_event = self._perturbation_injector.maybe_perturb(
                action, uav_pos, target_pos, timestamp)
        return action, pert_event

    def check_maneuver(self, timestamp: float) -> Any | None:
        if self._maneuver_injector is not None and self._target_type == "vehicle":
            return self._maneuver_injector.maybe_inject(self._target._vehicle, timestamp)
        return None

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
                    a.set_autopilot(False)
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
