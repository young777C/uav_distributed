"""Main recording loop — drives the synchronous CARLA mode for one episode."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import carla
import h5py
import numpy as np
from PIL import Image

from scene.scene_manager import SceneManager
from recording.sensors import RGBSensor


class EpisodeRecorder:
    """Records one tracking episode in CARLA synchronous mode.

    Output: an HDF5 file with all frames + a metadata JSON sidecar.
    """

    def __init__(
        self,
        scene: SceneManager,
        world: carla.World,
        output_dir: Path,
        episode_id: int,
        fps: int = 10,
        resolution: tuple[int, int] = (336, 336),
    ):
        self._scene = scene
        self._world = world
        self._output_dir = output_dir
        self._episode_id = episode_id
        self._fps = fps
        self._resolution = resolution
        # §10: horizontal FOV (deg) — narrowed 90→70 so small aerial targets project to
        # more pixels (make/body readable). Drives the sensor AND the projection maths below.
        self._fov = float(scene._config.get("output", {}).get("fov", 90.0))
        self._dt = 1.0 / fps

        # Buffers (flushed every N frames)
        self._buffer_size = 100
        self._frames: list[np.ndarray] = []
        self._actions: list[list[float]] = []
        self._states: list[dict] = []
        self._bboxes: list[dict] = []
        self._occlusions: list[float] = []
        self._maneuvers: list[dict] = []
        self._target_positions: list[list[float]] = []  # for waypoint GT
        self._distractor_positions: list[np.ndarray] = []  # each (D, 6): xyz + vxyz
        self._episode_attrs: dict = {}                     # written once on file create

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_steps: int | None = None) -> dict[str, Any]:
        """Run one episode. Returns summary dict."""
        # Setup scene first
        metadata = self._scene.setup_episode()
        assert self._scene.drone is not None

        # Episode-level attributes written into the HDF5 file (design §5.2 attrs).
        self._episode_attrs = {
            "language": metadata.get("language", ""),
            "target_bp": metadata.get("target_blueprint", ""),
            "target_desc": metadata.get("target_desc", ""),
            "target_class": metadata.get("target_class", ""),
            "target_color": str(metadata.get("target_color")),
            "strategy": metadata.get("strategy", ""),
            "num_similar": int(metadata.get("num_similar", 0)),
            "town": self._world.get_map().name.split("/")[-1],
            "seed": str(metadata.get("seed")),
            "weather": json.dumps(metadata.get("weather", {})),
            "distractors": json.dumps(metadata.get("distractors", [])),
            "occlusion_biased": bool(metadata.get("occlusion_biased", False)),
            "camera_pitch": float(metadata.get("camera_pitch", -50.0)),
        }

        # Scenarios set this under `environment:` (like the other env keys); fall back to
        # a top-level value, then the default. (Was read only at top level → scenario
        # `environment.max_duration_seconds` was silently ignored, e.g. mvp's 150 ran 180.)
        cfg = self._scene._config
        max_duration = cfg.get("environment", {}).get(
            "max_duration_seconds", cfg.get("max_duration_seconds", 180))
        max_steps = max_steps or int(max_duration * self._fps)

        # Setup synchronous mode
        settings = self._world.get_settings()
        original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._dt
        self._world.apply_settings(settings)

        # Validate target is moving AND let traffic disperse (tick 90x = 9 seconds)
        max_speed = 0.0
        for _ in range(90):
            self._scene.step_target(0.1)
            # Also move UAV toward target during warmup
            tpos, _, _, _ = self._scene.get_target_state()
            drone_state = self._scene.drone.get_state()
            action, _ = self._scene.expert_action(
                tpos, drone_state.position, drone_state.yaw, True, 0.0,
            )
            self._scene.step_drone(action)
            self._world.tick()
            _, _, _, ts = self._scene.get_target_state()
            max_speed = max(max_speed, ts)
        if max_speed < 1.2:
            self._scene.cleanup()
            try:
                self._world.tick()
            except RuntimeError:
                pass
            self._world.apply_settings(original_settings)
            return {"episode_id": self._episode_id, "steps": 0,
                    "duration_seconds": 0, "fps_actual": 0,
                    "metadata": metadata, "skipped": True,
                    "reason": f"target stationary (max_speed={max_speed:.1f}m/s)"}

        # Create RGB sensor (manually positioned each tick)
        rgb_sensor = self._create_attached_sensor()

        # Structural-occlusion detector: rays from target bbox corners → camera. Gives
        # a graded occlusion level (target hidden by building/tunnel while still in
        # frame) — the label §6.2/§6.4/H2 need. Computed live, stored per frame.
        from scene.occlusion import OcclusionDetector
        occ_detector = OcclusionDetector(self._world, num_rays=8, hit_threshold=0.5)
        target_actor = (getattr(self._scene._target, "_vehicle", None)
                        or getattr(self._scene._target, "_walker", None))

        start_time = time.time()
        off_lost = 0     # consecutive frames the target is OFF-screen (truly lost)
        occ_lost = 0     # consecutive frames the target is in-frame but occluded
        env_cfg = self._scene._config.get("environment", {})
        max_lost = int(env_cfg.get("max_lost_seconds", 5.0) / self._dt)
        max_occluded = int(env_cfg.get("max_occluded_seconds", 10.0) / self._dt)

        try:
            for step in range(max_steps):
                elapsed = step * self._dt

                # Step target vehicle (route following)
                self._scene.step_target(self._dt)

                # Possibly inject a target maneuver (sudden stop / sharp turn / ...)
                man_event = self._scene.check_maneuver(elapsed)

                # Recycle off-screen look-alikes back near the target (co-visibility).
                drone_state = self._scene.drone.get_state()
                self._scene.recycle_lost_lookalikes()

                # Get current states (reflect any recycling)
                tpos, tvel, tyaw, tspeed = self._scene.get_target_state()
                dstates = self._scene.get_distractor_states()  # (D, 6)

                # §11 deliberate loss event: at onset kick a strong evade; DURING the window
                # keep the target sprinting/swerving so it stays off-frame while the expert
                # flies to INTERCEPT the true (privileged) target (recovery mode) — a genuine
                # fly-to-intercept demonstration, NOT hover. At window end, force >=2 look-alikes
                # co-visible so the re-acquisition must use language.
                win_dur = self._scene.loss_window_onset(elapsed)
                if win_dur is not None:
                    self._scene.force_target_evade(elapsed, win_dur)
                if self._scene.in_loss_window(elapsed):
                    self._scene.sustain_target_evade(elapsed)
                elif self._scene.loss_window_offset(elapsed):
                    self._scene.restore_target_speed()
                    self._scene.force_covisible_lookalikes(min_n=2)
                target_visible = True   # §11: expert always servos the true target → intercepts when lost

                # Expert action with perturbation injection
                action, pert_event = self._scene.expert_action(
                    tpos, drone_state.position, drone_state.yaw,
                    target_visible, elapsed,
                )

                # Obstacle avoidance override
                action, obs_event = self._scene.check_obstacles(
                    action, drone_state.position, drone_state.yaw, elapsed,
                )

                # §11: NO hover override — during a loss window the expert flies to intercept
                # the true target (above, via recovery mode). The target's sustained sprint/
                # swerve keeps it off-frame; the recorded (off-screen, fly-toward-target) pairs
                # are the intercept demonstration BC needs (supervise_intercept, spec §3.2/§11).

                # Apply action and move camera BEFORE the tick
                self._scene.step_drone(action)
                cam_t = self._scene.drone.get_camera_transform()
                rgb_sensor._sensor.set_transform(cam_t)
                cam_pose = [
                    cam_t.location.x, cam_t.location.y, cam_t.location.z,
                    cam_t.rotation.pitch, cam_t.rotation.yaw,
                ]

                # Single tick — advances simulation AND captures sensor frame
                self._world.tick()
                rgb_frame = rgb_sensor.get_frame(timeout=0.5)

                # Graded structural occlusion [0,1] (fraction of target→camera rays
                # blocked by geometry — building / tunnel roof).
                occ_raycast = 0.0
                if target_actor is not None:
                    try:
                        occ_raycast = float(occ_detector.check(
                            target_actor,
                            np.array([cam_t.location.x, cam_t.location.y,
                                      cam_t.location.z]))[1])
                    except RuntimeError:
                        occ_raycast = 0.0

                # Record
                self._record_frame(
                    rgb_frame, action, drone_state, tpos, tvel, tspeed, tyaw,
                    target_visible, elapsed,
                    is_turning=self._scene.is_target_turning(),
                    pert_event=pert_event,
                    obs_event=obs_event,
                    distractor_states=dstates,
                    cam_pose=cam_pose,
                    man_event=man_event,
                    occ_raycast=occ_raycast,
                )

                # A1c③ — separate truncation. A target driven OFF-screen (truly lost,
                # e.g. off-map / expert can't follow) truncates fast (max_lost). A target
                # IN frame but structurally occluded (bridge/overpass/tunnel) is tolerated
                # much longer (max_occluded) so the occlusion event SURVIVES as recoverable
                # data — that is exactly the signal §6.2/§6.4/H2 need. Both counters reset
                # on a clean re-acquire.
                in_frame = self._point_in_frame(cam_t, tpos)
                # Expose visibility so the expert can switch to target-recovery framing
                # (fly toward the true target to re-acquire) once it drifts off-screen,
                # instead of staying with the group centroid and losing it.
                self._scene._target_in_frame = bool(in_frame)
                if not in_frame:
                    off_lost += 1
                    occ_lost = 0
                elif occ_raycast >= 0.6:
                    occ_lost += 1
                    off_lost = 0
                else:
                    off_lost = 0
                    occ_lost = 0
                if off_lost >= max_lost or occ_lost >= max_occluded:
                    why = "off-screen" if off_lost >= max_lost else "structural occlusion"
                    print(f"[Episode {self._episode_id}] target {why} too long "
                          f"(off={off_lost}/{max_lost} occ={occ_lost}/{max_occluded}) "
                          f"— truncating at step {step}")
                    break

                # Flush buffer periodically
                if len(self._frames) >= self._buffer_size:
                    self._flush_buffer()

        except Exception as e:
            print(f"[Episode {self._episode_id}] Error at step {step}: {e}")
            metadata["error"] = str(e)
        finally:
            # Destroy actors while STILL in synchronous mode, then tick once so the
            # server finalizes destruction before we switch back / start the next
            # episode (prevents 'operate on a destroyed actor' aborts from the TM).
            try:
                rgb_sensor.destroy()
            except RuntimeError:
                pass
            self._scene.cleanup()
            try:
                self._world.tick()
            except RuntimeError:
                pass
            self._world.apply_settings(original_settings)

            # Final flush
            if self._frames:
                self._flush_buffer()

        duration = time.time() - start_time
        return {
            "episode_id": self._episode_id,
            "steps": step,
            "duration_seconds": duration,
            "fps_actual": step / max(duration, 0.001),
            "metadata": metadata,
            "output_file": str(self._output_dir / f"episode_{self._episode_id:06d}.h5"),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_attached_sensor(self) -> RGBSensor:
        """Create an RGB sensor attached to the world spectator."""
        bp = self._world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self._resolution[0]))
        bp.set_attribute("image_size_y", str(self._resolution[1]))
        bp.set_attribute("fov", str(self._fov))

        cam_transform = self._scene.drone.get_camera_transform()
        sensor = self._world.spawn_actor(bp, cam_transform)
        rgb = RGBSensor.__new__(RGBSensor)
        rgb._sensor = sensor
        rgb._resolution = self._resolution

        import queue
        rgb._queue = queue.Queue()
        sensor.listen(rgb._queue.put)
        return rgb

    def _target_occluded(self, cam_t, tpos) -> bool:
        """Whether the line of sight from the camera to the target is blocked by
        geometry (building / tunnel roof) — i.e. the target is hidden even if it
        still projects inside the frame. One ray per call."""
        try:
            end = carla.Location(float(tpos[0]), float(tpos[1]), float(tpos[2]) + 0.8)
            d_target = cam_t.location.distance(end)
            for h in self._world.cast_ray(cam_t.location, end):
                if cam_t.location.distance(h.location) < d_target - 3.0:
                    return True   # something substantial between camera and target
        except (RuntimeError, AttributeError):
            return False
        return False

    def _point_in_frame(self, cam_t, wp) -> bool:
        """Whether a world point projects inside the RGB frame (90° FOV camera)."""
        M = np.array(cam_t.get_inverse_matrix())
        q = M @ np.array([float(wp[0]), float(wp[1]), float(wp[2]), 1.0])
        if q[0] <= 0.1:
            return False
        W, H = self._resolution
        f = W / (2.0 * np.tan(np.radians(self._fov) / 2.0))
        u = f * (q[1] / q[0]) + W / 2.0
        v = f * (-q[2] / q[0]) + H / 2.0
        return 0 <= u < W and 0 <= v < H

    def _record_frame(
        self,
        rgb: np.ndarray,
        action: tuple[float, float, float, float],
        drone_state: Any,
        tpos: np.ndarray,
        tvel: np.ndarray,
        tspeed: float,
        tyaw: float,
        visible: bool,
        elapsed: float,
        is_turning: bool = False,
        pert_event: Any = None,
        obs_event: Any = None,
        distractor_states: Any = None,
        cam_pose: Any = None,
        man_event: Any = None,
        occ_raycast: float = 0.0,
    ) -> None:
        self._frames.append(rgb)
        self._actions.append(list(action))
        cam_pose = cam_pose or [0.0, 0.0, 0.0, 0.0, 0.0]
        self._states.append({
            "t": elapsed,
            "occ_raycast": float(occ_raycast),
            "uav_x": float(drone_state.position[0]),
            "uav_y": float(drone_state.position[1]),
            "uav_z": float(drone_state.position[2]),
            "uav_vx": float(drone_state.velocity[0]),
            "uav_vy": float(drone_state.velocity[1]),
            "uav_vz": float(drone_state.velocity[2]),
            "uav_yaw": float(drone_state.yaw),
            "uav_energy": float(drone_state.energy),
            "is_turning": int(is_turning),
            "perturbation": int(pert_event is not None),
            "obstacle_avoid": int(obs_event is not None),
            "maneuver": int(man_event is not None),
            "cam_x": float(cam_pose[0]), "cam_y": float(cam_pose[1]),
            "cam_z": float(cam_pose[2]), "cam_pitch": float(cam_pose[3]),
            "cam_yaw": float(cam_pose[4]),
        })
        self._target_positions.append([
            float(tpos[0]), float(tpos[1]), float(tpos[2]),
            float(tvel[0]), float(tvel[1]), float(tvel[2]),
            float(tspeed), float(tyaw),
        ])
        if distractor_states is None:
            distractor_states = np.zeros((0, 6), dtype=np.float32)
        self._distractor_positions.append(np.asarray(distractor_states, dtype=np.float32))
        self._occlusions.append(0.0)  # placeholder, filled in postprocess

    def _flush_buffer(self) -> None:
        """Write buffered frames to HDF5 file."""
        if not self._frames:
            return

        output_path = self._output_dir / f"episode_{self._episode_id:06d}.h5"
        mode = "a" if output_path.exists() else "w"

        with h5py.File(output_path, mode) as f:
            # Determine offset for appending
            n_existing = f["rgb"].shape[0] if "rgb" in f else 0
            n_new = len(self._frames)
            total = n_existing + n_new

            # Create or resize datasets
            if "rgb" not in f:
                f.create_dataset(
                    "rgb", shape=(n_new, *self._resolution, 3),
                    maxshape=(None, *self._resolution, 3),
                    dtype=np.uint8, chunks=(1, *self._resolution, 3),
                    compression="gzip", compression_opts=9,
                )
                # Episode-level attributes (language, target/distractor meta, seed...)
                for k, v in self._episode_attrs.items():
                    f.attrs[k] = v
            else:
                f["rgb"].resize(total, axis=0)

            # Write RGB frames
            for i, frame in enumerate(self._frames):
                f["rgb"][n_existing + i] = frame

            # Write state/action data as JSON string attribute (updated each flush)
            # For large datasets, use separate HDF5 datasets instead
            state_group = f.require_group("state")
            for key in ["t", "uav_x", "uav_y", "uav_z", "uav_vx", "uav_vy",
                         "uav_vz", "uav_yaw", "uav_energy",
                         "is_turning", "perturbation", "obstacle_avoid", "maneuver",
                         "cam_x", "cam_y", "cam_z", "cam_pitch", "cam_yaw", "occ_raycast"]:
                vals = [s[key] for s in self._states]
                if key not in state_group:
                    state_group.create_dataset(
                        key, data=np.array(vals, dtype=np.float32),
                        maxshape=(None,), chunks=True,
                    )
                else:
                    ds = state_group[key]
                    ds.resize(total, axis=0)
                    ds[n_existing:] = np.array(vals, dtype=np.float32)

            # Actions
            act_group = f.require_group("action")
            act_arr = np.array(self._actions, dtype=np.float32)
            for j, key in enumerate(["dx", "dy", "dz", "dyaw"]):
                if key not in act_group:
                    act_group.create_dataset(
                        key, data=act_arr[:, j],
                        maxshape=(None,), chunks=True,
                    )
                else:
                    ds = act_group[key]
                    ds.resize(total, axis=0)
                    ds[n_existing:] = act_arr[:, j]

            # Target positions (for waypoint GT)
            tgt_arr = np.array(self._target_positions, dtype=np.float32)
            tgt_group = f.require_group("target")
            tgt_keys = ["tx", "ty", "tz", "tvx", "tvy", "tvz", "tspeed", "tyaw"]
            for j, key in enumerate(tgt_keys):
                if key not in tgt_group:
                    tgt_group.create_dataset(
                        key, data=tgt_arr[:, j],
                        maxshape=(None,), chunks=True,
                    )
                else:
                    ds = tgt_group[key]
                    ds.resize(total, axis=0)
                    ds[n_existing:] = tgt_arr[:, j]

            # Distractor per-frame tracks (T, D, 6): [x, y, z, vx, vy, vz]  (P1-a)
            if self._distractor_positions and self._distractor_positions[0].shape[0] > 0:
                darr = np.stack(self._distractor_positions, axis=0)
                D = darr.shape[1]
                dgrp = f.require_group("distractors")
                if "positions" not in dgrp:
                    dgrp.create_dataset(
                        "positions", data=darr, maxshape=(None, D, 6),
                        chunks=(min(64, n_new), D, 6),
                        compression="gzip", compression_opts=4,
                    )
                    # (D,) look-alike flags aligned with the D axis (§4.1). Static per
                    # episode → written once with positions.
                    try:
                        sim = self._scene.get_distractor_similar()
                        if sim is not None and len(sim) == D:
                            dgrp.create_dataset("similar", data=np.asarray(sim, dtype=np.uint8))
                    except (AttributeError, RuntimeError):
                        pass
                else:
                    ds = dgrp["positions"]
                    ds.resize(total, axis=0)
                    ds[n_existing:] = darr

        # Clear buffers
        self._frames.clear()
        self._actions.clear()
        self._states.clear()
        self._target_positions.clear()
        self._distractor_positions.clear()
        self._occlusions.clear()
