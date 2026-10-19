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
        extra_attrs: dict | None = None,
        out_name: str | None = None,
    ):
        self._scene = scene
        self._world = world
        self._output_dir = output_dir
        self._episode_id = episode_id
        self._fps = fps
        self._resolution = resolution
        # Horizontal FOV (deg) from output.fov — drives the sensor AND every projection; written to attrs.
        # v6 / Stage-2 standard is 90° (acot_note/1003_data-check.md §5.1; the 09-04 70° regen was a mismatch).
        self._fov = float(scene._config.get("output", {}).get("fov", 90.0))
        self._dt = 1.0 / fps
        # v6 (acot_note/1003_data-regen-v6-spec.md): provenance attrs + per-actor labels from an
        # instance-segmentation camera rendered from the SAME pose/fov as the RGB camera.
        self._extra_attrs = dict(extra_attrs or {})
        self._inst_enabled = bool(scene._config.get("output", {}).get("instance_seg", False))
        self._v6_buf: list[dict] = []
        le = scene._config.get("environment", {}).get("loss_events", {}) or {}
        # expert behaviour inside a deliberate loss window: intercept (default, privileged servo) |
        # blind (dead-reckon) | hover (hold position → the evading target really leaves the frame)
        self._loss_mode = str(le.get("expert_mode", "blind" if le.get("blind_expert") else "intercept"))
        self._out_path = output_dir / (out_name or f"episode_{episode_id:06d}.h5")
        self._v6_prev_iid = None
        self._target_track: list[tuple[float, float]] = []   # whole-episode target xy (stuck detector)
        self._hover_t0, self._hover_off = None, 0              # adaptive hover state (reacq)
        self._unstick = bool(scene._config.get("scene", {}).get("unstick", False))
        self._unstick_events = 0
        # 2026-10-09 (scene.kinematics.unjam.truncate_stuck_s, default off): the v6 "target stuck → truncate" rule
        # for v5-format runs too (last resort after the scene's unjam; the truncated episode fails QC → topped up)
        _kin = (scene._config.get("scene", {}) or {}).get("kinematics") or {}
        self._kin_trunc_steps = int(10 * float((_kin.get("unjam") or {}).get("truncate_stuck_s", 0) or 0))

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
        cfg_ = self._scene._config
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
            # camera / label provenance (v6): every downstream projection must read these
            "fov": float(self._fov),
            "K": json.dumps([[self._resolution[0] / (2.0 * np.tan(np.radians(self._fov) / 2.0)), 0.0, self._resolution[0] / 2.0],
                             [0.0, self._resolution[0] / (2.0 * np.tan(np.radians(self._fov) / 2.0)), self._resolution[1] / 2.0],
                             [0.0, 0.0, 1.0]]),
            "rgb_resolution": json.dumps([int(self._resolution[0]), int(self._resolution[1])]),
            "fps": int(self._fps),
            "label_version": "v6" if self._inst_enabled else "v5",
            # v6.1 distribution switches as direct attrs (Codex R2): filterable without parsing config_snapshot
            "ignore_lights_pct": float(cfg_.get("scene", {}).get("scene_actors_ignore_lights", 0.0)),
            "lane_change": bool(cfg_.get("scene", {}).get("scene_actors_lane_change", False)),
            "ambient_vehicles": int(cfg_.get("scene", {}).get("ambient", {}).get("vehicles", 0)),
            "ambient_spawn_mode": str(cfg_.get("scene", {}).get("ambient", {}).get("spawn_mode", "nearest")),
            "traffic_lights": json.dumps(cfg_.get("scene", {}).get("traffic_lights") or {}),
            "loss_expert_mode": self._loss_mode,
            "geometry_avoidance": bool(cfg_.get("uav", {}).get("geometry_avoidance", False)),
            "target_class_policy": json.dumps(cfg_.get("scene", {}).get("target_class_weights", {})),
            "spawn_filter": json.dumps(cfg_.get("scene", {}).get("spawn_filter") or {}),
            "spawn_max_roof_m": float(getattr(self._scene, "_spawn_roof_m", float("nan"))),
            "uav_start_altitude": float(metadata.get("uav_start_altitude") or float("nan")),
        }
        self._episode_attrs.update(self._extra_attrs)

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
            # v6: geometry avoidance also during warm-up (the UAV may spawn beside a high structure)
            action, _ = self._scene.check_obstacles(action, drone_state.position, drone_state.yaw, 0.0)
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
        inst_sensor = inst_q = None
        if self._inst_enabled:
            inst_sensor, inst_q = self._create_raw_sensor("sensor.camera.instance_segmentation")
            from recording.labels_v6 import actor_handle
            self._v6_handles = [target_actor] + [actor_handle(o) for o in self._scene._distractors]
            sim = list(self._scene.get_distractor_similar()) if self._scene._distractors else []
            self._v6_role = np.array([0] + [1 if s_ else 2 for s_ in sim], np.int8)
            self._v6_ids = np.array([h.id if h is not None else -1 for h in self._v6_handles], np.int32)

        start_time = time.time()
        blocked_run = 0  # v6: consecutive frames with the view blocked by geometry
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
                # v6 reacq: optionally fly the expert BLIND inside loss windows (dead-reckon) so the
                # evading target genuinely leaves frame; it intercepts again once the window ends.
                in_loss = bool(self._scene.in_loss_window(elapsed))
                if self._loss_mode == "hover":
                    # v6.1 adaptive hover: keep holding after the scheduled window until the target has been
                    # out of view for >= 3 s (max 15 s after onset) so every window yields a real loss.
                    if in_loss and self._hover_t0 is None:
                        self._hover_t0, self._hover_off = elapsed, 0
                    if self._hover_t0 is not None:
                        if not in_loss and (self._hover_off >= 30 or elapsed - self._hover_t0 > 15.0):
                            self._hover_t0 = None
                        else:
                            in_loss = True
                if self._loss_mode == "blind" and in_loss:
                    target_visible = False

                # Expert action with perturbation injection
                action, pert_event = self._scene.expert_action(
                    tpos, drone_state.position, drone_state.yaw,
                    target_visible, elapsed,
                )

                # Obstacle avoidance override
                action, obs_event = self._scene.check_obstacles(
                    action, drone_state.position, drone_state.yaw, elapsed,
                )
                if self._loss_mode == "hover" and in_loss:
                    action = (0.0, 0.0, 0.0, 0.0)
                action_valid = not (in_loss and self._loss_mode in ("blind", "hover"))

                # §11: NO hover override — during a loss window the expert flies to intercept
                # the true target (above, via recovery mode). The target's sustained sprint/
                # swerve keeps it off-frame; the recorded (off-screen, fly-toward-target) pairs
                # are the intercept demonstration BC needs (supervise_intercept, spec §3.2/§11).

                # Apply action and move camera BEFORE the tick
                self._scene.step_drone(action)
                cam_t = self._scene.drone.get_camera_transform()
                rgb_sensor._sensor.set_transform(cam_t)
                if inst_sensor is not None:
                    inst_sensor.set_transform(cam_t)
                cam_pose = [
                    cam_t.location.x, cam_t.location.y, cam_t.location.z,
                    cam_t.rotation.pitch, cam_t.rotation.yaw,
                ]

                # Single tick — advances simulation AND captures sensor frame
                frame_id = self._world.tick()
                if inst_sensor is None:
                    rgb_frame = rgb_sensor.get_frame(timeout=0.5)
                else:   # v6: RGB and instance image must come from the SAME simulation frame
                    rgb_data = self._get_synced(rgb_sensor._queue, frame_id)
                    a = np.frombuffer(rgb_data.raw_data, dtype=np.uint8).reshape(
                        (self._resolution[1], self._resolution[0], 4))
                    rgb_frame = a[:, :, [2, 1, 0]]
                    inst_data = self._get_synced(inst_q, frame_id)

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

                v6 = None
                if inst_sensor is not None:
                    v6 = self._v6_labels(cam_t, inst_data, occ_detector, occ_raycast)

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
                    v6=v6,
                    loss_window=in_loss,
                    action_valid=action_valid,
                )
                # v6.1 unstick: target still for 8 s outside a loss window → ask TM for a lane change (once per 8 s)
                if v6 is not None and self._unstick and not in_loss and step % 20 == 0 and len(self._target_track) >= 80:
                    tp8 = np.array(self._target_track[-80:])
                    if float(np.linalg.norm(tp8[-1] - tp8[0])) < 1.0:
                        veh, tm = self._scene._target_tm()
                        if veh is not None and tm is not None:
                            try:
                                tm.force_lane_change(veh, bool(step // 20 % 2))
                                self._unstick_events += 1
                            except RuntimeError:
                                pass
                # v6: gridlocked target (moved < 5 m in the last 40 s; a red-light wait is shorter) → stop
                # early; the episode fails QC (G8/G9) and is topped up instead of burning a full episode.
                _win = self._kin_trunc_steps or 400
                if (v6 is not None or self._kin_trunc_steps) and step >= _win and step % 50 == 0:
                    tp = np.array([p_[:2] for p_ in self._target_track[-_win:]])
                    if len(tp) >= _win and float(np.linalg.norm(tp[-1] - tp[0])) < 5.0 and \
                            float(np.abs(np.diff(tp, axis=0)).sum()) < 10.0:
                        print(f"[Episode {self._episode_id}] target stuck (<5 m in {_win / 10:.0f} s) — truncating at step {step}")
                        break
                # v6: camera inside / pressed against geometry (garbage frames) → truncate after 1 s. A building
                # merely hiding the target from a normal distance is realistic and kept; "pressed against" =
                # the frame is ≥80 % building AND the optical axis hits a surface within 8 m.
                if v6 is not None:
                    blocked_run = blocked_run + 1 if (v6["view_block_frac"] > 0.8 and v6["axis_hit_m"] < 8.0) else 0
                    if blocked_run >= 10:
                        print(f"[Episode {self._episode_id}] camera view blocked by geometry for "
                              f"{blocked_run} frames — truncating at step {step}")
                        break

                # A1c③ — separate truncation. A target driven OFF-screen (truly lost,
                # e.g. off-map / expert can't follow) truncates fast (max_lost). A target
                # IN frame but structurally occluded (bridge/overpass/tunnel) is tolerated
                # much longer (max_occluded) so the occlusion event SURVIVES as recoverable
                # data — that is exactly the signal §6.2/§6.4/H2 need. Both counters reset
                # on a clean re-acquire.
                in_frame = self._point_in_frame(cam_t, tpos)
                if self._hover_t0 is not None:
                    self._hover_off = self._hover_off + 1 if not in_frame else 0
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
            print(f"[Episode {self._episode_id}] Error at step {step}: {e!r}")
            metadata["error"] = repr(e)
        finally:
            # Destroy actors while STILL in synchronous mode, then tick once so the
            # server finalizes destruction before we switch back / start the next
            # episode (prevents 'operate on a destroyed actor' aborts from the TM).
            try:
                rgb_sensor.destroy()
            except RuntimeError:
                pass
            if inst_sensor is not None:
                try:
                    inst_sensor.stop(); inst_sensor.destroy()
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
            if self._kin_trunc_steps and self._out_path.exists():   # 2026-10-09: unjam audit → QC / filtering
                try:
                    with h5py.File(self._out_path, "a") as f:
                        f.attrs["kin_unjam"] = json.dumps({
                            "relocated": int(getattr(self._scene, "_kin_relocated", 0)),
                            "lane_kicks": int(getattr(self._scene, "_kin_lane_kicks", 0)),
                            "target_speed_mps": getattr(self._scene, "_kin_target_speed", None),
                            "events": list(getattr(self._scene, "_kin_events", []))})
                except OSError:
                    pass

        duration = time.time() - start_time
        return {
            "episode_id": self._episode_id,
            "steps": step,
            "duration_seconds": duration,
            "fps_actual": step / max(duration, 0.001),
            "metadata": metadata,
            "output_file": str(self._out_path),
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

    def _flush_v6(self, f, n0: int, total: int) -> None:
        """Append buffered v6 labels: actors/*, traffic/blobs, instance_mask, seg/{iid,semantic}, view/*."""
        g = f.require_group("actors")
        if "ids" not in g:
            g.create_dataset("ids", data=self._v6_ids)       # (A,) carla actor id; index 0 = target
            g.create_dataset("role", data=self._v6_role)     # (A,) 0 target, 1 look-alike, 2 other distractor
            ext = np.full((len(self._v6_ids), 3), np.nan, np.float32)
            loc = np.full((len(self._v6_ids), 3), np.nan, np.float32)
            for a, h in enumerate(self._v6_handles):
                try:
                    bb = h.bounding_box
                    ext[a] = (bb.extent.x, bb.extent.y, bb.extent.z)
                    loc[a] = (bb.location.x, bb.location.y, bb.location.z)
                except (RuntimeError, AttributeError):
                    pass
            g.create_dataset("extent", data=ext)              # (A,3) half-sizes (m) of the 3D bbox
            g.create_dataset("bbox_loc", data=loc)            # (A,3) bbox centre offset in the actor frame

        def app(grp, name, arr, comp=4, chunks=None):
            if name not in grp:
                grp.create_dataset(name, data=arr, maxshape=(None, *arr.shape[1:]),
                                   chunks=chunks or (min(64, len(arr)), *arr.shape[1:]),
                                   compression="gzip", compression_opts=comp)
            else:
                ds = grp[name]
                ds.resize(total, axis=0)
                ds[n0:total] = arr

        st = lambda k: np.stack([b[k] for b in self._v6_buf], 0)
        for k in ("in_frame", "amodal_box", "depth", "occ_raycast", "inst_id", "vis_px", "modal_box", "yaw", "pos"):
            app(g, k, st(k))
        app(f.require_group("traffic"), "blobs", st("traffic"))
        img_chunks = (1, self._resolution[1], self._resolution[0])
        app(f, "instance_mask", st("mask"), chunks=img_chunks)
        sg = f.require_group("seg")
        app(sg, "iid", st("seg_iid"), chunks=img_chunks)            # raw instance id on vehicle/pedestrian px
        app(sg, "semantic", st("seg_semantic"), chunks=img_chunks)  # CARLA semantic tag per pixel
        vg = f.require_group("view")
        app(vg, "block_frac", np.array([b["view_block_frac"] for b in self._v6_buf], np.float32))
        app(vg, "axis_hit_m", np.array([min(b["axis_hit_m"], 1e3) for b in self._v6_buf], np.float32))
        app(vg, "mean_rgb", np.array([fr.reshape(-1, 3).mean(0) for fr in self._frames], np.float32))

    def _axis_hit_distance(self, cam_t, max_d: float = 60.0) -> float:
        """Distance (m) along the camera optical axis to the first non-ground surface (inf if none ≤ max_d)."""
        f = cam_t.get_forward_vector()
        a = cam_t.location
        b = carla.Location(a.x + max_d * f.x, a.y + max_d * f.y, a.z + max_d * f.z)
        try:
            hits = self._world.cast_ray(a, b)
        except RuntimeError:
            return float("inf")
        L = carla.CityObjectLabel
        ground = {getattr(L, n) for n in ("Roads", "Sidewalks", "Ground", "Terrain", "RoadLines", "Water") if hasattr(L, n)}
        ds = [a.distance(h.location) for h in hits if h.label not in ground]
        return float(min(ds)) if ds else float("inf")

    def _create_raw_sensor(self, bp_name: str):
        """Extra camera with the RGB camera's resolution/fov, repositioned every tick → (sensor, queue)."""
        import queue
        bp = self._world.get_blueprint_library().find(bp_name)
        bp.set_attribute("image_size_x", str(self._resolution[0]))
        bp.set_attribute("image_size_y", str(self._resolution[1]))
        bp.set_attribute("fov", str(self._fov))
        sensor = self._world.spawn_actor(bp, self._scene.drone.get_camera_transform())
        q = queue.Queue()
        sensor.listen(q.put)
        return sensor, q

    @staticmethod
    def _get_synced(q, frame_id: int, timeout: float = 10.0):
        """Pop sensor data until the one rendered for frame_id (drops stale frames)."""
        while True:
            d = q.get(timeout=timeout)
            if d.frame >= frame_id:
                if d.frame > frame_id:
                    raise RuntimeError(f"sensor frame {d.frame} ahead of world frame {frame_id}")
                return d

    def _v6_labels(self, cam_t, inst_data, occ_detector, target_occ: float) -> dict:
        """Per-actor (target + distractors) labels for one frame — see recording/labels_v6.py."""
        from recording.labels_v6 import (PEDESTRIAN_TAG, VEHICLE_TAGS, amodal_box_from_vertices,
                                         associate, decode_instance, focal, view_block_frac)
        W, H = self._resolution
        fpx = focal(W, self._fov)
        cam_inv = np.array(cam_t.get_inverse_matrix())
        cam_xyz = np.array([cam_t.location.x, cam_t.location.y, cam_t.location.z])
        A = len(self._v6_handles)
        boxes, inframe = [None] * A, np.zeros(A, np.uint8)
        depth, occ = np.full(A, np.nan, np.float32), np.full(A, np.nan, np.float32)
        amodal = np.full((A, 4), -1.0, np.float32)
        yaw, pos = np.full(A, np.nan, np.float32), np.full((A, 3), np.nan, np.float32)
        for a, h in enumerate(self._v6_handles):
            if h is None or not h.is_alive:
                continue
            try:
                tf = h.get_transform()
                verts = h.bounding_box.get_world_vertices(tf)
            except RuntimeError:
                continue
            yaw[a] = tf.rotation.yaw
            pos[a] = (tf.location.x, tf.location.y, tf.location.z)
            pts = np.array([[v.x, v.y, v.z] for v in verts], np.float64)
            b, cin, dd = amodal_box_from_vertices(pts, cam_inv, W, H, fpx)
            boxes[a], inframe[a], depth[a] = b, int(cin), dd
            if b is not None:
                amodal[a] = b
                if a == 0:
                    occ[a] = target_occ
                else:
                    try:
                        occ[a] = float(occ_detector.check(h, cam_xyz)[1])
                    except RuntimeError:
                        pass
        tag, iid = decode_instance(inst_data.raw_data, W, H)
        axis_hit = self._axis_hit_distance(cam_t)
        inst_id, vis_px, modal, mask, traffic = associate(boxes, tag, iid, prev=self._v6_prev_iid)
        if self._v6_prev_iid is None:
            self._v6_prev_iid = np.full(A, -1, np.int32)
        self._v6_prev_iid = np.where(inst_id >= 0, inst_id, self._v6_prev_iid)
        dyn = np.isin(tag, VEHICLE_TAGS + (PEDESTRIAN_TAG,))
        return {"in_frame": inframe, "amodal_box": amodal, "depth": depth, "occ_raycast": occ,
                "inst_id": inst_id, "vis_px": vis_px, "modal_box": modal, "yaw": yaw, "pos": pos,
                "mask": mask, "traffic": traffic,
                "seg_iid": np.where(dyn, iid, 0).astype(np.uint16), "seg_semantic": tag.astype(np.uint8),
                "view_block_frac": view_block_frac(tag), "axis_hit_m": axis_hit}

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
        v6: dict | None = None,
        loss_window: bool = False,
        action_valid: bool = True,
    ) -> None:
        if v6 is not None:
            self._v6_buf.append(v6)
        if v6 is not None or self._kin_trunc_steps:          # 2026-10-09: stuck detector also for kinematics runs
            self._target_track.append((float(tpos[0]), float(tpos[1])))
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
            "loss_window": int(loss_window),
            "action_valid": int(action_valid),
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

        output_path = self._out_path
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
                         "cam_x", "cam_y", "cam_z", "cam_pitch", "cam_yaw", "occ_raycast",
                         "loss_window", "action_valid"]:
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

            if self._v6_buf:
                assert len(self._v6_buf) == n_new, (len(self._v6_buf), n_new)
                self._flush_v6(f, n_existing, total)

        # Clear buffers
        self._v6_buf.clear()
        self._frames.clear()
        self._actions.clear()
        self._states.clear()
        self._target_positions.clear()
        self._distractor_positions.clear()
        self._occlusions.clear()
