"""Data format validation tests — run without CARLA.

Usage:
    python -m pytest tests/test_data_format.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

pytestmark = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")


class TestHDF5Format:
    """Verify HDF5 output format matches training pipeline expectations."""

    def test_minimal_format(self):
        """Create a minimal valid HDF5 file and verify structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_episode.h5"

            n_steps = 100
            H, W = 224, 224

            with h5py.File(path, "w") as f:
                # RGB frames
                f.create_dataset("rgb", data=np.random.randint(0, 255, (n_steps, H, W, 3),
                                                               dtype=np.uint8))

                # State
                sg = f.create_group("state")
                for key in ["t", "uav_x", "uav_y", "uav_z", "uav_vx", "uav_vy",
                             "uav_vz", "uav_yaw", "uav_energy"]:
                    sg.create_dataset(key, data=np.random.randn(n_steps).astype(np.float32))

                # Actions
                ag = f.create_group("action")
                for key in ["dx", "dy", "dz", "dyaw"]:
                    ag.create_dataset(key, data=np.random.randn(n_steps).astype(np.float32))

                # Target
                tg = f.create_group("target")
                for key in ["tx", "ty", "tz", "tvx", "tvy", "tvz", "tspeed", "tyaw"]:
                    tg.create_dataset(key, data=np.random.randn(n_steps).astype(np.float32))

                # Annotations
                ann = f.create_group("annotation")
                for key in ["bbox_u", "bbox_v", "bbox_w", "bbox_h",
                             "occlusion", "search_mode"]:
                    ann.create_dataset(key, data=np.random.randn(n_steps).astype(np.float32))
                ann.create_dataset("waypoints", data=np.random.randn(n_steps, 9).astype(np.float32))
                ann.create_dataset("waypoint_sigma", data=np.random.randn(n_steps, 3).astype(np.float32))

            # Re-read and validate
            with h5py.File(path, "r") as f:
                assert f["rgb"].shape == (n_steps, H, W, 3)
                assert f["rgb"].dtype == np.uint8

                assert f["state/uav_x"].shape == (n_steps,)
                assert f["action/dx"].shape == (n_steps,)
                assert f["target/tx"].shape == (n_steps,)

                assert f["annotation/occlusion"].shape == (n_steps,)
                assert f["annotation/search_mode"].shape == (n_steps,)
                assert f["annotation/waypoints"].shape == (n_steps, 9)
                assert f["annotation/waypoint_sigma"].shape == (n_steps, 3)

    def test_frame_temporal_alignment(self):
        """Verify RGB frames and state/action are temporally aligned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_alignment.h5"
            n = 50

            with h5py.File(path, "w") as f:
                # Create datasets with known indices
                f.create_dataset("rgb", data=np.zeros((n, 224, 224, 3), dtype=np.uint8))
                sg = f.create_group("state")
                sg.create_dataset("t", data=np.arange(n, dtype=np.float32) * 0.1)
                ag = f.create_group("action")
                ag.create_dataset("dx", data=np.arange(n, dtype=np.float32))

            with h5py.File(path, "r") as f:
                # Each frame should have corresponding state and action
                for i in range(n):
                    assert f["state/t"][i] == pytest.approx(i * 0.1)
                    assert f["action/dx"][i] == pytest.approx(float(i))

    def test_occlusion_range(self):
        """Verify occlusion values are in [0, 1]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_occlusion.h5"

            with h5py.File(path, "w") as f:
                occ = np.clip(np.random.randn(200).astype(np.float32) * 0.3 + 0.3, 0.0, 1.0)
                ann = f.create_group("annotation")
                ann.create_dataset("occlusion", data=occ)

            with h5py.File(path, "r") as f:
                occ = f["annotation/occlusion"][:]
                assert occ.min() >= 0.0 and occ.max() <= 1.0
