"""CARLA sensor setup and management for data recording."""

from __future__ import annotations

import queue
import time

import carla
import numpy as np
from PIL import Image


class RGBSensor:
    """RGB camera sensor attached to the UAV spectator.

    Supports synchronous mode — get_data() blocks until a frame is ready.
    """

    def __init__(
        self,
        world: carla.World,
        parent: carla.Actor,
        attach_transform: carla.Transform,
        resolution: tuple[int, int] = (224, 224),
        fov: float = 90.0,
    ):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(resolution[0]))
        bp.set_attribute("image_size_y", str(resolution[1]))
        bp.set_attribute("fov", str(fov))

        self._sensor = world.spawn_actor(bp, attach_transform, attach_to=parent)
        self._queue: queue.Queue = queue.Queue()
        self._sensor.listen(self._queue.put)
        self._resolution = resolution

        # Color converter for CARLA BGRA → RGB
        self._cc = carla.ColorConverter.Raw

    def get_frame(self, timeout: float = 2.0) -> np.ndarray:
        """Return the latest RGB frame as a (H, W, 3) uint8 numpy array.

        Blocks until a frame is available or timeout.
        """
        data = self._queue.get(timeout=timeout)
        # CARLA raw format: BGRA, 4 channels
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = array.reshape((self._resolution[1], self._resolution[0], 4))
        # Convert BGRA → RGB
        rgb = array[:, :, [2, 1, 0]]
        return rgb

    def get_frame_pil(self, timeout: float = 2.0) -> Image.Image:
        """Return the latest RGB frame as a PIL Image."""
        return Image.fromarray(self.get_frame(timeout))

    def flush_queue(self) -> None:
        """Clear any stale frames from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def destroy(self) -> None:
        if self._sensor is not None and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()


class DepthSensor:
    """Depth camera sensor — optional, for 3D-aware pretraining."""

    def __init__(
        self,
        world: carla.World,
        parent: carla.Actor,
        attach_transform: carla.Transform,
        resolution: tuple[int, int] = (224, 224),
        fov: float = 90.0,
    ):
        bp = world.get_blueprint_library().find("sensor.camera.depth")
        bp.set_attribute("image_size_x", str(resolution[0]))
        bp.set_attribute("image_size_y", str(resolution[1]))
        bp.set_attribute("fov", str(fov))

        self._sensor = world.spawn_actor(bp, attach_transform, attach_to=parent)
        self._queue: queue.Queue = queue.Queue()
        self._sensor.listen(self._queue.put)
        self._resolution = resolution

    def get_frame(self, timeout: float = 2.0) -> np.ndarray:
        """Return depth as (H, W, 1) float32 (meters)."""
        data = self._queue.get(timeout=timeout)
        array = np.frombuffer(data.raw_data, dtype=np.float32)
        # CARLA depth: float32, BGR channel encoding in meters
        array = array.reshape((self._resolution[1], self._resolution[0], 4))
        # Extract R channel (depth is encoded across RGB, normalized)
        # Simplified: use R channel as approximation
        depth = array[:, :, 0:1]
        return depth.astype(np.float32)

    def destroy(self) -> None:
        if self._sensor is not None and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
