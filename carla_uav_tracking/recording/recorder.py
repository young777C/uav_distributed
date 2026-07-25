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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_steps: int | None = None) -> dict[str, Any]:
        """Run one episode. Returns summary dict."""
        # Setup scene first
        metadata = self._scene.setup_episode()
        assert self._scene.drone is not None

        max_duration = self._scene._config.get("max_duration_seconds", 180)
        max_steps = max_steps or int(max_duration * self._fps)

        # Setup synchronous mode
        settings = self._world.get_settings()
        original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._dt
        self._world.apply_settings(settings)

        # Validate target is moving AND let UAV catch up (tick 50x = 5 seconds)
        max_speed = 0.0
        for _ in range(50):
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
        if max_speed < 2.0:
            self._world.apply_settings(original_settings)
            self._scene.cleanup()
            return {"episode_id": self._episode_id, "steps": 0,
                    "duration_seconds": 0, "fps_actual": 0,
                    "metadata": metadata, "skipped": True,
                    "reason": f"target stationary (max_speed={max_speed:.1f}m/s)"}

        # Create RGB sensor (manually positioned each tick)
        rgb_sensor = self._create_attached_sensor()

        start_time = time.time()

        try:
            for step in range(max_steps):
                elapsed = step * self._dt

                # Step target vehicle (route following)
                self._scene.step_target(self._dt)

                # Get current states
                tpos, tvel, tyaw, tspeed = self._scene.get_target_state()
                drone_state = self._scene.drone.get_state()
                target_visible = True

                # Expert action with perturbation injection
                action, pert_event = self._scene.expert_action(
                    tpos, drone_state.position, drone_state.yaw,
                    target_visible, elapsed,
                )

                # Obstacle avoidance override
                action, obs_event = self._scene.check_obstacles(
                    action, drone_state.position, drone_state.yaw, elapsed,
                )

                # Apply action and move camera BEFORE the tick
                self._scene.step_drone(action)
                cam_t = self._scene.drone.get_camera_transform()
                rgb_sensor._sensor.set_transform(cam_t)

                # Single tick — advances simulation AND captures sensor frame
                self._world.tick()
                rgb_frame = rgb_sensor.get_frame(timeout=0.5)

                # Record
                self._record_frame(
                    rgb_frame, action, drone_state, tpos, tvel, tspeed, tyaw,
                    target_visible, elapsed,
                    is_turning=self._scene.is_target_turning(),
                    pert_event=pert_event,
                    obs_event=obs_event,
                )

                # Flush buffer periodically
                if len(self._frames) >= self._buffer_size:
                    self._flush_buffer()

        except Exception as e:
            print(f"[Episode {self._episode_id}] Error at step {step}: {e}")
            metadata["error"] = str(e)
        finally:
            # Restore settings
            self._world.apply_settings(original_settings)
            rgb_sensor.destroy()
            self._scene.cleanup()

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
        bp.set_attribute("fov", "90.0")

        cam_transform = self._scene.drone.get_camera_transform()
        sensor = self._world.spawn_actor(bp, cam_transform)
        rgb = RGBSensor.__new__(RGBSensor)
        rgb._sensor = sensor
        rgb._resolution = self._resolution

        import queue
        rgb._queue = queue.Queue()
        sensor.listen(rgb._queue.put)
        return rgb

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
    ) -> None:
        self._frames.append(rgb)
        self._actions.append(list(action))
        self._states.append({
            "t": elapsed,
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
        })
        self._target_positions.append([
            float(tpos[0]), float(tpos[1]), float(tpos[2]),
            float(tvel[0]), float(tvel[1]), float(tvel[2]),
            float(tspeed), float(tyaw),
        ])
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
                         "is_turning", "perturbation", "obstacle_avoid"]:
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

        # Clear buffers
        self._frames.clear()
        self._actions.clear()
        self._states.clear()
        self._target_positions.clear()
        self._occlusions.clear()
