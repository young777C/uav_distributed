"""Datasets for Stage-1 EAR warmup.

RealEARDataset  — reads episode H5 (waypoint GT + UAV pose) + a precomputed VLM
                  context cache (per-episode .npz from backbone_kv.precompute).
SyntheticEARDataset — dependency-free; a LEARNABLE cond->waypoint mapping so the
                  Stage-1 loop can be smoke-tested with no VLM / no CARLA data.

Waypoint GT (`annotation/waypoints`, shape (T,9) -> (T,3,3) = [+2s,+4s,+6s]×xyz) is
world-frame; we predict it RELATIVE to the current UAV position and /scale so the
target is translation-invariant and O(1).
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from acot_probe.projection import inverse_pose_matrix


def _select_episodes(cfg, split):
    """Episodes for a split. Prefer the A2 manifest; else fall back to a naive slice."""
    import json
    all_eps = sorted(glob.glob(cfg["data"]["episodes_glob"]))
    manifest_path = cfg["data"].get("split_manifest")
    if manifest_path and Path(manifest_path).exists():
        man = json.loads(Path(manifest_path).read_text())
        names = set(man.get(split, []))
        return [e for e in all_eps if Path(e).name in names]
    # fallback (no manifest): naive episode slice, train/val only
    vf = cfg["data"].get("val_frac", 0.15)
    n_val = max(1, int(len(all_eps) * vf))
    if split == "test":
        return []
    return all_eps[n_val:] if split == "train" else all_eps[:n_val]


class RealEARDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.scale = cfg["waypoint"].get("scale", 50.0)
        fps = cfg.get("fps", 10)
        self.offsets = [int(o * fps) for o in cfg["waypoint"].get("offsets_s", [2, 4, 6])]
        self.k = len(self.offsets)
        self.rel = cfg["waypoint"].get("relative_to_uav", True)
        self.cam_frame = cfg["waypoint"].get("camera_frame", False)  # predict in camera frame
        self.ctx_dir = Path(cfg["data"]["context_cache"])
        self._zc = {}          # ctx npz cache (path -> (frames, ctx)); load each file ONCE
        self._hc = {}          # h5 field cache (ep -> (waypoints, uav_xyz, occ)); read ONCE
        eps = _select_episodes(cfg, split)
        self.index = self._build_index(eps, cfg)
        self.cond_dim = self._peek_cond_dim()

    def _build_index(self, eps, cfg):
        import h5py
        stride = int(cfg["data"].get("frame_stride", 3))
        cap = int(cfg["data"].get("max_frames_per_episode", 200))
        # Drop each episode's last `horizon` frames: their +6s waypoint GT is clamped
        # to the final frame (degenerate) — see postprocess._compute_waypoints. Also
        # drops episodes too short to yield any valid-horizon frame.
        horizon = int(cfg["waypoint"].get("max_horizon_s", 6) * cfg.get("fps", 10))
        idx = []
        for ep in eps:
            ctx = self.ctx_dir / (Path(ep).stem + ".npz")
            if not ctx.exists():
                continue
            with h5py.File(ep, "r") as f:
                n = f["rgb"].shape[0]
            valid_n = n - horizon
            if valid_n <= 0:
                continue
            for i in list(range(0, valid_n, stride))[:cap]:
                idx.append((ep, str(ctx), i))
        return idx

    def _peek_cond_dim(self):
        if not self.index:
            return self.cfg["ear"].get("cond_dim", 2048)
        z = np.load(self.index[0][1])
        return int(z["ctx"].shape[-1])

    def __len__(self):
        return len(self.index)

    def _ctx(self, path):
        z = self._zc.get(path)
        if z is None:
            d = np.load(path)
            z = (np.asarray(d["frames"]), np.asarray(d["ctx"], dtype=np.float32))
            self._zc[path] = z
        return z

    def _h5(self, ep):
        h = self._hc.get(ep)
        if h is None:
            import h5py
            with h5py.File(ep, "r") as f:
                occ = np.asarray(f["annotation/occlusion"][:], np.float32) \
                    if "annotation/occlusion" in f else None
                tgt = np.stack([f["target/tx"][:], f["target/ty"][:],
                                f["target/tz"][:]], -1).astype(np.float32)
                uav = np.stack([f["state/uav_x"][:], f["state/uav_y"][:],
                                f["state/uav_z"][:]], -1).astype(np.float32)
                cam = np.stack([f["state/cam_x"][:], f["state/cam_y"][:], f["state/cam_z"][:],
                                f["state/cam_pitch"][:], f["state/cam_yaw"][:]], -1).astype(np.float32)
                vel = np.stack([f["state/uav_vx"][:], f["state/uav_vy"][:],
                                f["state/uav_vz"][:]], -1).astype(np.float32)
                h = (tgt, uav, cam, vel, occ)   # waypoints computed on-the-fly at self.offsets
            self._hc[ep] = h
        return h

    def __getitem__(self, j):
        # in-memory caches: each ctx npz / h5 file is read ONCE, then indexed (was
        # reloading the whole npz + reopening h5 per sample -> GPU-starved training).
        ep, ctx_path, i = self.index[j]
        frames, ctx = self._ctx(ctx_path)
        row = min(int(np.searchsorted(frames, i)), len(frames) - 1)
        cond = torch.from_numpy(ctx[row])                       # (M, C)
        tgt, uav_all, cam, vel, occ = self._h5(ep)
        n = len(tgt)
        fut = np.stack([tgt[min(i + off, n - 1)] for off in self.offsets])  # (K,3) future target world
        if self.cam_frame:
            # transform future target into the CURRENT camera frame (fwd, right, up) ->
            # removes UAV-yaw ambiguity of a world-frame target (root cause #1)
            M = inverse_pose_matrix(*cam[i])                    # world -> camera-local (4x4)
            homog = np.concatenate([fut, np.ones((self.k, 1), np.float32)], -1)
            wp = (homog @ M.T)[:, :3].astype(np.float32)
        else:
            wp = fut - uav_all[i][None, :]
        wp = (wp / self.scale).astype(np.float32)
        # proprio: altitude + camera pitch (depth via ground plane) + velocity (motion)
        proprio = np.array([cam[i][2] / 30.0, cam[i][3] / 90.0,
                            vel[i][0] / 15.0, vel[i][1] / 15.0, vel[i][2] / 15.0], np.float32)
        vis = float(occ[i] < 0.8) if occ is not None else 1.0
        mask = torch.ones(cond.shape[0], dtype=torch.bool)
        return {"cond": cond, "cond_mask": mask, "wp": torch.from_numpy(wp),
                "proprio": torch.from_numpy(proprio), "vis": torch.tensor(vis)}


class SyntheticEARDataset(Dataset):
    """cond ~ N(0,1) (M,C); wp = pooled(cond) @ W + b + noise -> learnable by EAR."""

    def __init__(self, n=512, m=16, c=64, k=3, seed=0, map_seed=42):
        rng = np.random.default_rng(seed)
        self.k = k
        self.cond_dim = c
        self._cond = rng.normal(0, 1, (n, m, c)).astype(np.float32)
        # SHARED mapping across train/val (fixed map_seed) — like the real physics;
        # train/val differ only in sampled cond, not in the cond->waypoint function.
        W = np.random.default_rng(map_seed).normal(0, 1, (c, k * 3)).astype(np.float32) / np.sqrt(c)
        pooled = self._cond.mean(1)                              # (n, c)
        wp = pooled @ W
        wp = wp / (wp.std() + 1e-6)                              # unit-scale (like real /scale)
        wp = wp + 0.03 * rng.normal(0, 1, (n, k * 3)).astype(np.float32)
        self._wp = wp.reshape(n, k, 3).astype(np.float32)

    def __len__(self):
        return len(self._cond)

    def __getitem__(self, j):
        cond = torch.from_numpy(self._cond[j])
        return {"cond": cond, "cond_mask": torch.ones(cond.shape[0], dtype=torch.bool),
                "wp": torch.from_numpy(self._wp[j]), "proprio": torch.zeros(5),
                "vis": torch.tensor(1.0)}


def collate(batch):
    """Pad variable-length cond token sets to the batch max M."""
    M = max(b["cond"].shape[0] for b in batch)
    C = batch[0]["cond"].shape[1]
    cond = torch.zeros(len(batch), M, C)
    mask = torch.zeros(len(batch), M, dtype=torch.bool)
    for i, b in enumerate(batch):
        m = b["cond"].shape[0]
        cond[i, :m] = b["cond"]
        mask[i, :m] = b["cond_mask"]
    wp = torch.stack([b["wp"] for b in batch])
    vis = torch.stack([b["vis"] for b in batch])
    proprio = torch.stack([b["proprio"] for b in batch])
    return {"cond": cond, "cond_mask": mask, "wp": wp, "proprio": proprio, "vis": vis}
