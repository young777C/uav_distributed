"""Cache per-episode candidate SEQUENCES for the Temporal Association Head (TAH, path-B: replace the
heuristic WHICH stack with ONE learned module). Per frame: crop-DINOv2 appearance + image position +
GT identity, in temporal order (stable actor idx). Offline from mvp_full_v5. Runs in container (GPU)."""
from __future__ import annotations
import argparse, os, sys, glob
import numpy as np, yaml, h5py

_CARLA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "carla_uav_tracking")
for _p in (os.path.dirname(_CARLA), _CARLA):
    if _p not in sys.path: sys.path.insert(0, _p)
from acot_probe.projection import project_point
from rollout.candidates import Candidate
from train.dataset import _select_episodes


def _mk(world, is_t, idx, campose, W, H, fov, vsz, bmin, bmax):
    pr = project_point((float(world[0]), float(world[1]), float(world[2])), tuple(campose), W, H, fov)
    if pr is None: return None
    u, v, depth = pr
    if not (0 <= u < W and 0 <= v < H): return None
    f = W / (2.0 * np.tan(np.radians(fov) / 2.0)); s = float(np.clip(f * vsz / max(depth, 0.1), bmin, bmax))
    return Candidate(u, v, s, s, depth, is_t, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--out", default="runs/tah_cache.pt")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import torch
    from train.reid_resolution_probe import _load_dino, _dino_feats
    cfg = yaml.safe_load(open(a.config)); ic = cfg["image"]
    W, H, fov = ic["width"], ic["height"], ic["fov_deg"]
    vsz, bmin, bmax = float(ic["vehicle_size_m"]), float(ic["box_min_px"]), float(ic["box_max_px"])
    dino = _load_dino("vit_small_patch14_dinov2.lvd142m", a.device)
    eps = _select_episodes(cfg, "train") + _select_episodes(cfg, "val")
    episodes = []
    for ei, ep in enumerate(eps):
        with h5py.File(ep, "r") as fh:
            n = fh["rgb"].shape[0]
            tgt = np.stack([fh["target/tx"][:], fh["target/ty"][:], fh["target/tz"][:]], -1)
            dpos = fh["distractors/positions"][:]                       # (T,D,6)
            cam = np.stack([fh[f"state/cam_{k}"][:] for k in ["x","y","z","pitch","yaw"]], -1)
            frames = []
            for i in range(0, n, a.stride):
                campose = cam[i]
                cands, aidx, tflag = [], [], []
                c = _mk(tgt[i], True, -1, campose, W, H, fov, vsz, bmin, bmax)
                if c is not None: cands.append(c); aidx.append(-1); tflag.append(1)
                for d in range(dpos.shape[1]):
                    c = _mk(dpos[i, d, :3], False, d, campose, W, H, fov, vsz, bmin, bmax)
                    if c is not None: cands.append(c); aidx.append(d); tflag.append(0)
                if not cands: continue
                feats = _dino_feats(dino, np.asarray(fh["rgb"][i]), cands, W, H, 224, 1.3, a.device).astype(np.float32)
                feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
                pos = np.array([[c.u / W, c.v / H, c.depth / 100.0] for c in cands], np.float32)
                tidx = tflag.index(1) if 1 in tflag else -1
                frames.append({"feat": feats.astype(np.float16), "pos": pos,
                               "aidx": np.array(aidx, np.int16), "tidx": tidx})
            if frames: episodes.append(frames)
        print(f"[tah_cache] ep{ei} {os.path.basename(ep)}: {len(frames)} frames (total eps={len(episodes)})", flush=True)
    torch.save({"episodes": episodes, "d_feat": 384}, a.out)
    nf = sum(len(e) for e in episodes)
    print(f"[tah_cache] wrote {a.out}: {len(episodes)} episodes, {nf} frames", flush=True)


if __name__ == "__main__":
    main()
