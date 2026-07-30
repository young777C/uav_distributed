"""Stage-2 datasets.

RealStage2Dataset  — reads episodes + a MULTI-LAYER VLM context cache
                     (backbone_kv --layers ...) + labels.py-derived targets.
SyntheticStage2Dataset — dependency-free, learnable mappings so the full Stage-2
                     loss/train loop can be smoke-tested with no VLM/CARLA/GPU.

Sample dict contract (both datasets + collate_stage2):
  vlm_ctx_layers : list[L] of (M, C)     # per-layer context, for IAR
  vlm_ctx        : (M, C)                 # single layer for EAR/DiT cross-attn
  proprio        : (8,)
  action         : (H, 5)                # expert action chunk GT
  waypoint       : (K, 3)                # EAR waypoint GT
  occ_structural : ()                    # IAR occlusion-head target (L_visibility)
  maneuver       : ()                    # IAR maneuver-head target (L_maneuver)
  cand_feats     : (N, Ccand)            # candidate box features (target + distractors)
  target_idx     : int                   # which candidate is the referred target
  visible        : ()                    # 1 = target genuinely observable (mask L_action)
  tier           : 'a' | 'b' | 'c'
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from acot_probe.projection import inverse_pose_matrix


def _unit(x):
    return (x / (x.std() + 1e-6)).astype(np.float32)


class SyntheticStage2Dataset(Dataset):
    def __init__(self, n=1024, L=4, M=16, C=48, H=16, K=3, N=5,
                 proprio=8, seed=0, map_seed=42):
        rng = np.random.default_rng(seed)
        mrng = np.random.default_rng(map_seed)          # SHARED mapping across train/val
        self.L, self.H, self.K, self.N, self.C = L, H, K, N, C
        Wa = mrng.normal(0, 1, (C + proprio, H * 5)).astype(np.float32) / np.sqrt(C + proprio)
        Ww = mrng.normal(0, 1, (C, K * 3)).astype(np.float32) / np.sqrt(C)
        Wt = mrng.normal(0, 1, (C, C)).astype(np.float32) / np.sqrt(C)
        wocc = mrng.normal(0, 1, C).astype(np.float32)
        wman = mrng.normal(0, 1, C).astype(np.float32)
        tiers = np.array(list("abc"))

        # pass 1: raw targets (linear maps of context/proprio)
        raw = []
        for _ in range(n):
            ctx = rng.normal(0, 1, (L, M, C)).astype(np.float32)
            pooled = ctx.mean((0, 1))                    # (C,)
            prop = rng.normal(0, 1, proprio).astype(np.float32)
            a_raw = (np.concatenate([pooled, prop]) @ Wa).astype(np.float32)   # (H*5,)
            w_raw = (pooled @ Ww).astype(np.float32)                            # (K*3,)
            cand = rng.normal(0, 1, (N, C)).astype(np.float32)
            tgt = int(rng.integers(N))
            cand[tgt] = (Wt @ pooled) + 0.3 * rng.normal(0, 1, C).astype(np.float32)
            occ = float(1 / (1 + np.exp(-(pooled @ wocc) / np.sqrt(C))))
            man = float((pooled @ wman) > 0)
            raw.append((ctx, prop, a_raw, w_raw, cand, tgt, occ, man, str(rng.choice(tiers))))
        # pass 2: GLOBAL normalization (keeps the map linear -> learnable, cf. Stage-1)
        A_all = np.stack([r[2] for r in raw]); A_all = A_all / (A_all.std() + 1e-6)
        W_all = np.stack([r[3] for r in raw]); W_all = W_all / (W_all.std() + 1e-6)
        self.samples = []
        for i, (ctx, prop, _, _, cand, tgt, occ, man, tier) in enumerate(raw):
            self.samples.append({
                "vlm_ctx_layers": [torch.from_numpy(ctx[l]) for l in range(L)],
                "vlm_ctx": torch.from_numpy(ctx[-1]),
                "proprio": torch.from_numpy(prop),
                "action": torch.from_numpy(A_all[i].reshape(H, 5).astype(np.float32)),
                "waypoint": torch.from_numpy(W_all[i].reshape(K, 3).astype(np.float32)),
                "occ_structural": torch.tensor(occ, dtype=torch.float32),
                "maneuver": torch.tensor(man, dtype=torch.float32),
                "cand_feats": torch.from_numpy(cand),
                "target_idx": tgt,
                "visible": torch.tensor(1.0),
                "act_supervise": torch.tensor(1.0),
                "tier": tier,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class RealStage2Dataset(Dataset):
    """Reads episodes + a MULTI-LAYER context cache (backbone_kv --layers ...) and
    labels.py-derived targets. Candidate features are pooled from the cached grid at
    each vehicle's projected box. Requires the Stage-2 context cache to exist."""

    def __init__(self, cfg, split="train"):
        from .dataset import _select_episodes
        from . import curriculum
        self.cfg = cfg
        self.ctx_dir = Path(cfg["data"]["stage2_context_cache"])
        self.H = cfg["action"]["horizon"]
        fps = cfg.get("fps", 10)
        self.offsets = [int(o * fps) for o in cfg["waypoint"].get("offsets_s", [2, 4, 6])]
        self.K = len(self.offsets)
        self.wp_scale = cfg["waypoint"]["scale"]
        self.a_scale = cfg["action"].get("scale", 5.0)
        self.cam_frame = cfg["waypoint"].get("camera_frame", False)  # align with Stage-1 EAR
        # §3.2 mask reversal: supervise action on temporary-loss frames where the target
        # reappears within the EAR horizon (expert demonstrates intercept, not blind-follow).
        self.supervise_intercept = bool(cfg["train"].get("supervise_intercept", False)) \
            if "train" in cfg else False
        self.intercept_hz = int(cfg["waypoint"].get("max_horizon_s", 6) * fps)
        self.img = cfg["image"]
        self._tier_of = curriculum.tier_of
        self._zc = {}          # multi-layer ctx npz cache (path -> (frames, grid, ctx))
        self._hc = {}          # h5 field cache (ep -> dict of arrays)
        self.index = self._build_index(_select_episodes(cfg, split), cfg)
        self.cond_dim = self._peek()

    def _build_index(self, eps, cfg):
        import h5py
        stride = int(cfg["data"].get("frame_stride", 3))
        cap = int(cfg["data"].get("max_frames_per_episode", 200))
        horizon = int(cfg["waypoint"].get("max_horizon_s", 6) * cfg.get("fps", 10))
        idx = []
        for ep in eps:
            ctx = self.ctx_dir / (Path(ep).stem + ".npz")
            if not ctx.exists():
                continue
            with h5py.File(ep, "r") as f:
                n = f["rgb"].shape[0]
                a = f.attrs
                tier = self._tier_of(str(a.get("strategy", "")), int(a.get("num_similar", 0)))
            valid = n - horizon
            for i in list(range(0, max(0, valid), stride))[:cap]:
                idx.append((ep, str(ctx), i, tier))
        return idx

    def _peek(self):
        if not self.index:
            return self.cfg["model"].get("cond_dim", 2048)
        return int(self._ctx(self.index[0][1])[2].shape[-1])   # works for mmap & fat-npz

    def __len__(self):
        return len(self.index)

    def _ctx(self, path):                          # meta from npz; big ctx via mmap (shared)
        z = self._zc.get(path)
        if z is None:
            d = np.load(path)
            frames = np.asarray(d["frames"]); grid = tuple(int(x) for x in d["grid"])
            ctx_npy = Path(path).with_name(Path(path).stem + ".ctx.npy")
            if ctx_npy.exists():
                ctx = np.load(ctx_npy, mmap_mode="r")      # memmap -> OS page cache, shared
            else:
                ctx = np.asarray(d["ctx"], dtype=np.float32)  # fallback: old fat-npz format
            z = (frames, grid, ctx)
            self._zc[path] = z
        return z

    def _h5(self, ep):                             # h5 field arrays, read ONCE
        d = self._hc.get(ep)
        if d is None:
            import h5py
            from . import labels
            with h5py.File(ep, "r") as f:
                g = lambda k: np.asarray(f[k][:], np.float32)
                d = {
                    "tgt": np.stack([g("target/tx"), g("target/ty"), g("target/tz")], -1),
                    "cam": np.stack([g("state/cam_x"), g("state/cam_y"), g("state/cam_z"),
                                     g("state/cam_pitch"), g("state/cam_yaw")], -1),
                    "vel": np.stack([g("state/uav_vx"), g("state/uav_vy"), g("state/uav_vz")], -1),
                    "act": np.stack([g("action/dx"), g("action/dy"), g("action/dz"), g("action/dyaw")], -1),
                    "sm": g("annotation/search_mode"),
                    "occ_s": g("annotation/occ_structural") if "annotation/occ_structural" in f else None,
                    "man": g("state/maneuver") if "state/maneuver" in f else None,
                    "vis": (labels.combined_occlusion(f) < 0.8).astype(np.float32),
                    "distr": g("distractors/positions") if "distractors/positions" in f else None,
                }
            self._hc[ep] = d
        return d

    def _candidates(self, h, i):                   # target + in-frame distractor boxes (L_target_id)
        import math
        from acot_probe.projection import project_point, Candidate
        img = self.img; W, H, fov = img["width"], img["height"], img["fov_deg"]
        cam = tuple(h["cam"][i]); f = W / (2.0 * math.tan(math.radians(fov) / 2.0))
        def mk(xyz, is_t, idx):
            pr = project_point(xyz, cam, W, H, fov)
            if pr is None:
                return None
            u, v, depth = pr
            if not (0 <= u < W and 0 <= v < H):
                return None
            s = float(np.clip(f * img.get("vehicle_size_m", 4.0) / max(depth, 0.1),
                              img.get("box_min_px", 10), img.get("box_max_px", 140)))
            return Candidate(u, v, s, s, depth, is_t, idx)
        t = mk(h["tgt"][i], True, -1)
        if t is None:
            return []
        out = [t]
        if h["distr"] is not None:
            for d in range(h["distr"].shape[1]):
                c = mk(h["distr"][i, d, :3], False, d)
                if c is not None:
                    out.append(c)
        return out

    def __getitem__(self, j):
        from acot_probe.backbones import pool_box
        ep, ctx_path, i, tier = self.index[j]
        frames, (gh, gw), ctx_all = self._ctx(ctx_path)
        row = min(int(np.searchsorted(frames, i)), len(frames) - 1)
        ctx = np.ascontiguousarray(ctx_all[row])   # (L, gh*gw, C); page-in mmap row -> owned array
        L, _, C = ctx.shape
        layers = [torch.from_numpy(ctx[l]) for l in range(L)]
        grid_last = ctx[-1].reshape(gh, gw, C)
        h = self._h5(ep); n = len(h["tgt"])
        W, H = self.img["width"], self.img["height"]
        # action chunk: [dx,dy,dz,dyaw]/scale + search_mode
        action = np.concatenate([h["act"][i:i + self.H], h["sm"][i:i + self.H][:, None]], -1).astype(np.float32)
        if action.shape[0] < self.H:
            action = np.pad(action, ((0, self.H - action.shape[0]), (0, 0)))
        action[:, :4] = action[:, :4] / self.a_scale
        # camera-frame waypoints + 5d proprio (SAME as Stage-1 EAR -> warmstart-compatible)
        fut = np.stack([h["tgt"][min(i + off, n - 1)] for off in self.offsets])
        if self.cam_frame:
            M = inverse_pose_matrix(*h["cam"][i])
            homog = np.concatenate([fut, np.ones((self.K, 1), np.float32)], -1)
            wp = ((homog @ M.T)[:, :3] / self.wp_scale).astype(np.float32)
        else:
            wp = ((fut - h["cam"][i][:3][None, :]) / self.wp_scale).astype(np.float32)
        proprio = np.array([h["cam"][i][2] / 30.0, h["cam"][i][3] / 90.0,
                            h["vel"][i][0] / 15.0, h["vel"][i][1] / 15.0, h["vel"][i][2] / 15.0], np.float32)
        occ_s = float(h["occ_s"][i]) if h["occ_s"] is not None else 0.0
        man = float(h["man"][i]) if h["man"] is not None else 0.0
        vis = float(h["vis"][i])
        # §3.2 mask reversal: on temporary-loss frames (target occluded now but
        # reappears within the EAR horizon) supervise the intercept action instead of
        # masking it. act_supervise == visible when the flag is off (current behavior).
        act_sup = vis
        if self.supervise_intercept and vis < 0.5:
            fut = h["vis"][i:i + self.intercept_hz]
            if fut.size and float(fut.max()) >= 0.5:     # temporary loss -> intercept-eligible
                act_sup = 1.0
        cands = self._candidates(h, i)
        if not cands:                              # target off-screen -> pick another frame
            return self.__getitem__((j + 1) % len(self.index))
        cand_feats = np.stack([pool_box(grid_last, c.frac_box(W, H)) for c in cands]).astype(np.float32)
        tgt_idx = next(k for k, c in enumerate(cands) if c.is_target)
        return {
            "vlm_ctx_layers": layers, "vlm_ctx": layers[-1],
            "proprio": torch.from_numpy(proprio), "action": torch.from_numpy(action),
            "waypoint": torch.from_numpy(wp),
            "occ_structural": torch.tensor(occ_s, dtype=torch.float32),
            "maneuver": torch.tensor(man, dtype=torch.float32),
            "cand_feats": torch.from_numpy(cand_feats), "target_idx": tgt_idx,
            "visible": torch.tensor(vis), "act_supervise": torch.tensor(act_sup),
            "tier": tier,
        }


def collate_stage2(batch):
    """Pad per-layer context tokens (M) and candidates (N) to batch max; build masks."""
    L = len(batch[0]["vlm_ctx_layers"])
    C = batch[0]["vlm_ctx"].shape[-1]
    Ccand = batch[0]["cand_feats"].shape[-1]
    Mmax = max(b["vlm_ctx"].shape[0] for b in batch)
    Nmax = max(b["cand_feats"].shape[0] for b in batch)
    B = len(batch)

    ctx_layers = [torch.zeros(B, Mmax, C) for _ in range(L)]
    ctx = torch.zeros(B, Mmax, C)
    ctx_mask = torch.zeros(B, Mmax, dtype=torch.bool)
    cand = torch.zeros(B, Nmax, Ccand)
    cand_mask = torch.zeros(B, Nmax, dtype=torch.bool)
    tgt = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        m = b["vlm_ctx"].shape[0]
        ctx[i, :m] = b["vlm_ctx"]; ctx_mask[i, :m] = True
        for l in range(L):
            ctx_layers[l][i, :m] = b["vlm_ctx_layers"][l]
        nc = b["cand_feats"].shape[0]
        cand[i, :nc] = b["cand_feats"]; cand_mask[i, :nc] = True
        tgt[i] = b["target_idx"]

    stack = lambda k: torch.stack([b[k] for b in batch])
    return {
        "vlm_ctx_layers": ctx_layers, "vlm_ctx": ctx, "ctx_mask": ctx_mask,
        "proprio": stack("proprio"), "action": stack("action"),
        "waypoint": stack("waypoint"), "occ_structural": stack("occ_structural"),
        "maneuver": stack("maneuver"), "cand_feats": cand, "cand_mask": cand_mask,
        "target_idx": tgt, "visible": stack("visible"), "act_supervise": stack("act_supervise"),
        "tiers": [b["tier"] for b in batch],
    }
