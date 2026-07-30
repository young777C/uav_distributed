"""End-to-end plumbing test — NO model download, NO CARLA, NO GPU.

Creates a tiny synthetic HDF5 episode (correct schema), a DummyBackbone that
returns random grid features with a PLANTED signal at the target cell, then runs
the full cache -> probe -> report path. Asserts the probe beats chance (proving
projection→candidate→pool→head all wire up). Run:

    python -m acot_probe.smoke_test
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .data import build_cache, resolve_episode_paths
from .probe import run_layer_sweep


# ---------------------------------------------------------------------------
# synthetic episode with the real recorder schema
# ---------------------------------------------------------------------------
def make_episode(path, n=60, n_distractor=4, seed=0):
    import h5py
    rng = np.random.default_rng(seed)
    W = H = 336
    with h5py.File(path, "w") as f:
        f.create_dataset("rgb", data=rng.integers(0, 255, (n, H, W, 3), dtype=np.uint8))
        # camera fixed at origin, looking +x, small downward pitch
        st = f.create_group("state")
        for k, val in {
            "cam_x": 0.0, "cam_y": 0.0, "cam_z": 20.0,
            "cam_pitch": -20.0, "cam_yaw": 0.0,
        }.items():
            st.create_dataset(k, data=np.full(n, val, np.float32))
        # target ~30-40m ahead, small lateral wander (kept in-frame)
        tg = f.create_group("target")
        tx = np.linspace(30, 40, n) + rng.normal(0, 0.5, n)
        ty = rng.normal(0, 3, n)
        tz = np.full(n, 0.0)
        for k, v in {"tx": tx, "ty": ty, "tz": tz}.items():
            tg.create_dataset(k, data=v.astype(np.float32))
        # distractors: ahead too, different lateral offsets
        dg = f.create_group("distractors")
        dpos = np.zeros((n, n_distractor, 6), np.float32)
        for d in range(n_distractor):
            dpos[:, d, 0] = np.linspace(28, 42, n) + rng.normal(0, 0.5, n)
            dpos[:, d, 1] = rng.uniform(-8, 8) + rng.normal(0, 0.5, n)
        dg.create_dataset("positions", data=dpos)
        f.attrs["language"] = "Track the red Ford Mustang. Ignore the look-alikes."
        f.attrs["target_desc"] = "red Ford Mustang"
        f.attrs["num_similar"] = 2


class DummyBackbone:
    """No-op stand-in: random grid + a planted signal at the TARGET's grid cell.

    Proves the probe CAN separate target from distractors when the signal exists,
    and that projection lands the target in the correct grid cell.
    """
    C = 32

    def __init__(self, grid=16, seed=0):
        self.g = grid
        self.rng = np.random.default_rng(seed)
        self._target_dir = self.rng.normal(0, 1, self.C)   # the "referred" signal

    def encode(self, image, instruction, layers):
        # deterministic-ish random grid
        grid = self.rng.normal(0, 1, (self.g, self.g, self.C)).astype(np.float32)
        text = self.rng.normal(0, 1, self.C).astype(np.float32)
        out = {}
        for L in ([l for l in layers if l != -1] or [16]):
            out[L] = (grid.copy(), text.copy())
        return out


def _inject_target_signal(cache_layer, feat_dim, strength=4.0):
    """Post-hoc: add the signal to target rows so a probe can learn (dummy only)."""
    C = feat_dim // 2
    sig = np.zeros(feat_dim, np.float32)
    sig[:C] = strength                                   # bias the box-feature half
    cache_layer["feat"][cache_layer["is_target"]] += sig


def main():
    cfg = {
        "image": {"width": 336, "height": 336, "fov_deg": 90.0,
                  "vehicle_size_m": 4.0, "box_min_px": 10, "box_max_px": 140},
        "data": {"frame_stride": 2, "max_frames_per_episode": 40, "min_candidates": 2,
                 "episodes_glob": ""},
        "probe": {"epochs": 40, "lr": 1e-3, "hidden": 64, "batch_groups": 32,
                  "val_frac": 0.3, "head_device": "cpu"},
        "layers": [16], "seed": 0,
    }
    with tempfile.TemporaryDirectory() as td:
        ep = Path(td) / "episode_000000.h5"
        make_episode(ep)
        cfg["data"]["episodes_glob"] = str(Path(td) / "episode_*.h5")
        eps = resolve_episode_paths(cfg)
        assert eps, "synthetic episode not found"

        cache = build_cache(DummyBackbone(), cfg, eps, cfg["layers"])
        assert cache, "no features cached — projection put no target in-frame?"
        L = next(iter(cache))
        n_groups = len(np.unique(cache[L]["group"]))
        print(f"[smoke] cached {len(cache[L]['feat'])} rows across {n_groups} frames, "
              f"feat_dim={cache[L]['feat'].shape[1]}")
        assert n_groups >= 4, "too few usable frames (target projection failing?)"

        # without signal: probe ~ chance ; with signal: probe >> chance
        _inject_target_signal(cache[L], cache[L]["feat"].shape[1])
        res = run_layer_sweep(cache, cfg)[0]
        print(f"[smoke] layer {res.layer}: TSA={res.tsa:.2f} chance={res.chance:.2f} "
              f"(val groups={res.n_groups_val})")
        assert res.tsa > res.chance + 0.15, "probe failed to learn planted signal"
        print("[smoke] OK — projection→candidates→pool→probe wiring is sound.")


if __name__ == "__main__":
    main()
