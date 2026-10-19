"""1006 DINOv3 trial: rebuild the OFFLINE TAH cache (the 64 mvp_full_v5 episodes of runs/tah_cache.pt) with a different
crop-feature backbone. Identical to train/tah_cache.py (same config / projection / stride / candidate construction /
crops: 224 px, 1.3× box, ImageNet norm) except the backbone; train/tah_cache.py itself is not modified. After writing,
every frame is checked against the DINOv2 cache: same frames, same candidate order (aidx), same target index, same
positions — only the features differ.
  python -m acot.prep.tah_cache_backbone --backbone dinov3_vitb16 --out runs/tah_1006/tah_cache_dinov3.pt \\
      --ref runs/tah_cache.pt [--device cuda:0]"""
import argparse
import os
import sys

import numpy as np

from acot import ROOT

V3 = "external/OA-VAT/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov3_vitb16", choices=["dinov3_vitb16"])
    ap.add_argument("--config", default="train/config_v5.yaml")          # = the DINOv2 cache's config
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="runs/tah_cache.pt")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    for p in (ROOT, os.path.join(ROOT, "carla_uav_tracking"), os.path.join(ROOT, "train")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import h5py
    import torch
    import yaml
    from acot.runtime.head_oavat import dinov3_extractor
    from train.reid_resolution_probe import _dino_feats
    from train.tah_cache import _mk, _select_episodes

    cfg = yaml.safe_load(open(os.path.join(ROOT, a.config)))
    ic = cfg["image"]
    W, H, fov = ic["width"], ic["height"], ic["fov_deg"]
    vsz, bmin, bmax = float(ic["vehicle_size_m"]), float(ic["box_min_px"]), float(ic["box_max_px"])
    model = dinov3_extractor(a.backbone, os.path.join(ROOT, V3), a.device).model
    eps = _select_episodes(cfg, "train") + _select_episodes(cfg, "val")
    episodes, order, d_feat = [], [], None
    for ei, ep in enumerate(eps):
        with h5py.File(ep, "r") as fh:
            n = fh["rgb"].shape[0]
            tgt = np.stack([fh["target/tx"][:], fh["target/ty"][:], fh["target/tz"][:]], -1)
            dpos = fh["distractors/positions"][:]
            cam = np.stack([fh[f"state/cam_{k}"][:] for k in ["x", "y", "z", "pitch", "yaw"]], -1)
            frames = []
            for i in range(0, n, a.stride):
                campose = cam[i]
                cands, aidx, tflag = [], [], []
                c = _mk(tgt[i], True, -1, campose, W, H, fov, vsz, bmin, bmax)
                if c is not None:
                    cands.append(c); aidx.append(-1); tflag.append(1)
                for d in range(dpos.shape[1]):
                    c = _mk(dpos[i, d, :3], False, d, campose, W, H, fov, vsz, bmin, bmax)
                    if c is not None:
                        cands.append(c); aidx.append(d); tflag.append(0)
                if not cands:
                    continue
                feats = _dino_feats(model, np.asarray(fh["rgb"][i]), cands, W, H, 224, 1.3, a.device).astype(np.float32)
                feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
                d_feat = feats.shape[1]
                pos = np.array([[c.u / W, c.v / H, c.depth / 100.0] for c in cands], np.float32)
                tidx = tflag.index(1) if 1 in tflag else -1
                frames.append({"feat": feats.astype(np.float16), "pos": pos, "aidx": np.array(aidx, np.int16),
                               "tidx": tidx})
            if frames:
                episodes.append(frames)
                order.append(ei)
        print(f"[tah_cache_backbone] ep{ei} {os.path.basename(ep)}: {len(frames)} frames", flush=True)
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    torch.save({"episodes": episodes, "d_feat": d_feat, "order": order, "fov": fov, "backbone": a.backbone,
                "weights": V3}, os.path.join(ROOT, a.out))
    ref = torch.load(os.path.join(ROOT, a.ref), map_location="cpu", weights_only=False)["episodes"]
    same = len(ref) == len(episodes) and all(
        len(x) == len(y) and all(np.array_equal(fx["aidx"], fy["aidx"]) and fx["tidx"] == fy["tidx"]
                                 and np.allclose(fx["pos"], fy["pos"]) for fx, fy in zip(x, y))
        for x, y in zip(ref, episodes))
    print(f"[tah_cache_backbone] wrote {a.out}: {len(episodes)} episodes, d_feat={d_feat}; "
          f"frames / candidates / targets / positions identical to {a.ref}: {same}", flush=True)
    if not same:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
