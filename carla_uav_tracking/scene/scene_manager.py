"""Scene management — actor lifecycle, spawning, and cleanup."""

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


class SceneManager:
    """Manages the lifecycle of all CARLA actors for one tracking episode.

    Usage:
        mgr = SceneManager(world, config)
        mgr.setup_episode()
        for step in range(max_steps):
            target_state = mgr.get_target_state()
            action = mgr.expert_action(target_state, timestamp)
            mgr.step(action)
        mgr.cleanup()
    """

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
        self._maneuver_injector: ManeuverInjector | None = None
        self._distractors: list = []  # spawned in setup_episode
        self._perturbation_injector = None  # set in setup_episode
        self._obstacle_avoider = None       # set in setup_episode
        self._language_gen = None           # set in setup_episode
        self._map = world.get_map()
        self._spawn_points = self._map.get_spawn_points()

    # ------------------------------------------------------------------
    # Episode Lifecycle
    # ------------------------------------------------------------------

    def setup_episode(self) -> dict[str, Any]:
        """Spawn all actors for a new episode. Returns episode metadata."""
        # -- Randomize weather --
        weather_cfg = self._config.get("weather", {})
        self._set_random_weather(weather_cfg)

        # -- Setup language generator (needed before distractor selection) --
        from scene.language_generator import LanguageGenerator, CARS_ONLY
        self._language_gen = LanguageGenerator()

        # -- Select town with wide roads --
        wide_road_towns = ["Town03", "Town04", "Town05"]
        if self._world.get_map().name.split("/")[-1] not in wide_road_towns:
            try:
                self._client.load_world(random.choice(wide_road_towns))
                time.sleep(2.0)
                self._world = self._client.get_world()
            except RuntimeError:
                pass  # keep current map

        # -- Spawn target vehicle --
        target = self._config.get("target", {})
        self._target_type = "vehicle"

        target_bp = random.choice(CARS_ONLY)

        spawn_pt = random.choice(self._world.get_map().get_spawn_points()) if \
            self._world.get_map().get_spawn_points() else \
            carla.Transform(carla.Location(0, 0, 1))

        speed_range = target.get("speed_range", [5, 25])
        speed = random.uniform(speed_range[0], speed_range[1])

        self._target = VehicleTarget(
            self._world, target_bp, spawn_pt,
            speed_kmh=speed * 3.6, carla_client=self._client,
        )

        # -- Spawn distractor vehicles (visual noise, not tracked) --
        self._distractors = []
        num_distractors = random.randint(4, 8)
        distractor_bps = self._language_gen.get_distractors(
            target_bp, num_distractors
        ) if self._language_gen else random.sample(
            [b for b in CARS_ONLY if b != target_bp], min(num_distractors, len(CARS_ONLY)-1)
        )

        # Spawn distractors NEAR the target (same road segment)
        target_wp = self._world.get_map().get_waypoint(spawn_pt.location)
        for dbp in distractor_bps:
            # Place distractor ahead or behind target on same road
            offset_m = random.uniform(-60, 60)  # meters ahead (+) or behind (-)
            d_wp = target_wp
            if offset_m > 0:
                for _ in range(int(offset_m / 10)):
                    nxt = d_wp.next(10.0)
                    if nxt: d_wp = nxt[0]
            else:
                for _ in range(int(-offset_m / 10)):
                    nxt = d_wp.previous(10.0)
                    if nxt: d_wp = nxt[0]

            dsp = carla.Transform(
                carla.Location(
                    x=d_wp.transform.location.x + random.uniform(-3, 3),
                    y=d_wp.transform.location.y + random.uniform(-3, 3),
                    z=d_wp.transform.location.z + 0.5,
                ),
                d_wp.transform.rotation,
            )
            try:
                d = VehicleTarget(
                    self._world, dbp, dsp,
                    speed_kmh=random.uniform(25, 50),
                    carla_client=self._client,
                )
                self._distractors.append(d)
            except RuntimeError:
                pass  # skip if spawn fails

        # -- Log --
        print(f"  Target: {target_bp} | Distractors: {len(self._distractors)} vehicles")

        # -- Spawn UAV --
        uav_cfg = self._config.get("uav", {})
        start = uav_cfg.get("start", {})

        # Position UAV behind and to the side of target (relative to its heading)
        target_transform = self._target.transform
        target_loc = target_transform.location
        target_yaw_rad = np.radians(target_transform.rotation.yaw)

        # Behind the vehicle: negative of forward direction
        behind_dist = random.uniform(*start.get("distance_range", [30, 80]))
        # Lateral offset: perpendicular to vehicle heading (avoid tree line)
        lateral_offset = random.uniform(*start.get("lateral_offset_range", [5, 15]))
        lateral_side = random.choice([-1, 1])
        # Altitude: above tree canopy
        alt = random.uniform(*start.get("altitude_range", [35, 60]))

        # Forward vector (vehicle's heading)
        fwd_x, fwd_y = np.cos(target_yaw_rad), np.sin(target_yaw_rad)
        # Right vector (perpendicular to forward)
        right_x, right_y = -fwd_y, fwd_x

        uav_loc = carla.Location(
            x=target_loc.x - fwd_x * behind_dist + right_x * lateral_offset * lateral_side,
            y=target_loc.y - fwd_y * behind_dist + right_y * lateral_offset * lateral_side,
            z=target_loc.z + alt,
        )
        # UAV faces toward the vehicle
        look_yaw = np.degrees(np.arctan2(
            target_loc.y - uav_loc.y, target_loc.x - uav_loc.x
        ))
        uav_transform = carla.Transform(uav_loc, carla.Rotation(pitch=-10, yaw=look_yaw, roll=0))

        self.drone = KinematicDrone(self._world, uav_transform)

        # -- Setup expert policy --
        expert_cfg = uav_cfg.get("expert_policy", {})
        self.expert = ExpertPolicy(expert_cfg)
        self.expert.reset_episode()

        # -- Setup safety monitor --
        self.safety = SafetyMonitor()
        self.safety.reset()

        # -- Setup perturbation injector --
        from carla_uav.perturbation import PerturbationInjector
        pert_cfg = uav_cfg.get("perturbation", {})
        self._perturbation_injector = PerturbationInjector()

        # -- Setup obstacle avoider --
        from carla_uav.obstacle_avoidance import ObstacleAvoider
        self._obstacle_avoider = ObstacleAvoider()

        # Generate language instruction with distractor context
        distractor_bp_list = [d._vehicle.type_id for d in self._distractors
                              if hasattr(d, '_vehicle')]
        direction = self._target.direction if self._target_type == "vehicle" else "unknown"
        language = self._language_gen.generate(
            target_bp,
            distractor_bps=distractor_bp_list,
            direction=direction,
            road_type="multi-lane urban road",
            lane_info=random.choice(["the right lane", "the center lane", "the left lane"]),
        )

        return {
            "target_type": self._target_type,
            "target_blueprint": target_bp,
            "target_initial_speed": speed,
            "uav_start_distance": behind_dist,
            "uav_start_altitude": alt,
            "weather": self._get_current_weather_dict(),
            "language": language,
            "expert_style": {
                "noise_scale": self.expert.current_style.noise_scale
                if self.expert.current_style else None,
            },
        }

    def get_target_state(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Return (position, velocity, yaw, speed) for current target."""
        assert self._target is not None
        return self._target.get_state()

    def step_target(self, dt: float = 0.1) -> None:
        """Advance target and distractor vehicles."""
        if self._target is not None and self._target_type == "vehicle":
            self._target.step(dt)
        for d in self._distractors:
            d.step(dt)

    def is_target_turning(self) -> bool:
        if self._target is not None and self._target_type == "vehicle":
            return self._target.is_turning
        return False

    def check_obstacles(
        self, action, uav_pos, uav_yaw, timestamp,
    ) -> tuple[tuple, Any]:
        """Apply obstacle avoidance to expert action."""
        if self._obstacle_avoider is not None:
            return self._obstacle_avoider.check(action, uav_pos, timestamp)
        return action, None

    def expert_action(
        self,
        target_pos: np.ndarray,
        uav_pos: np.ndarray,
        uav_yaw: float,
        target_visible: bool,
        timestamp: float,
    ) -> tuple[tuple[float, float, float, float], Any]:
        """Compute expert action with perturbation injection.

        Returns:
            (action, perturbation_event_or_None)
        """
        assert self.expert is not None
        action = self.expert.act(
            target_pos, uav_pos, uav_yaw, target_visible, timestamp
        )

        # Inject APF perturbation
        pert_event = None
        if self._perturbation_injector is not None and target_pos is not None:
            action, pert_event = self._perturbation_injector.maybe_perturb(
                action, uav_pos, target_pos, timestamp,
            )

        return action, pert_event

    def check_maneuver(self, timestamp: float) -> Any | None:
        """Check and possibly inject a target maneuver."""
        if self._maneuver_injector is not None and self._target_type == "vehicle":
            return self._maneuver_injector.maybe_inject(self._target._vehicle, timestamp)
        return None

    def step_drone(self, action: tuple[float, float, float, float]) -> None:
        """Apply action to drone."""
        assert self.drone is not None
        self.drone.step(*action)

    def cleanup(self) -> None:
        """Destroy all spawned actors."""
        if self.drone is not None:
            # Spectator doesn't need explicit destroy
            self.drone = None
        if self._target is not None:
            self._target.destroy()
            self._target = None
        for d in self._distractors:
            d.destroy()
        self._distractors = []
        self.expert = None
        self.safety = None

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
            "cloudiness": w.cloudiness,
            "precipitation": w.precipitation,
            "sun_altitude": w.sun_altitude_angle,
            "fog_density": w.fog_density,
        }
