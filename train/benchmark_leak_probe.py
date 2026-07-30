"""Benchmark leak probe — quantify how much the target can be picked among
distractors WITHOUT language or appearance, using ONLY candidate geometry
(image position + depth + relative ranks).

Motivation (2026-07-29): the w/o-language ablation showed language is NOT
load-bearing (with-lang 6.16%±1.81 vs no-lang 6.23%±1.12 mis_follow on the hard
multi-candidate subset — indistinguishable). This probe found WHY: an optimal
position-only classifier hits 3.29% on the same subset — a near-perfect POSITIONAL
SHORTCUT. The target's geometry is systematically predictable, so no model needs
language. Fixing the benchmark = decorrelating the target's position/depth/rank
from target-ness; then this probe's learned-classifier mis_follow should rise
toward chance and language should finally matter.

Re-run after a data fix to check the shortcut is closed:

    python -m train.benchmark_leak_probe --config train/config.yaml --split val
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml

from acot_probe.projection import project_point
from train.dataset import _select_episodes


def _proj(xyz, cam, W, H, fov):
    pr = project_point(tuple(xyz), tuple(cam), W, H, fov)
    if pr is None or not (0 <= pr[0] < W and 0 <= pr[1] < H):
        return None
    return pr                                        # (u, v, depth)


def build(cfg, split):
    """Hard-subset (>=2 on-screen candidates) samples -> (per-candidate feat, target_idx=0)."""
    img = cfg["image"]; W, H, fov = img["width"], img["height"], img["fov_deg"]
    stride = int(cfg["data"].get("frame_stride", 3))
    cap = int(cfg["data"].get("max_frames_per_episode", 200))
    horizon = int(cfg["waypoint"].get("max_horizon_s", 6) * cfg.get("fps", 10))
    out = []
    for ep in _select_episodes(cfg, split):
        with h5py.File(ep, "r") as f:
            n = f["rgb"].shape[0]
            tgt = np.stack([f["target/tx"][:], f["target/ty"][:], f["target/tz"][:]], -1)
            cam = np.stack([f["state/cam_x"][:], f["state/cam_y"][:], f["state/cam_z"][:],
                            f["state/cam_pitch"][:], f["state/cam_yaw"][:]], -1)
            distr = f["distractors/positions"][:] if "distractors/positions" in f else None
        for i in list(range(0, max(0, n - horizon), stride))[:cap]:
            t = _proj(tgt[i], cam[i], W, H, fov)
            if t is None:
                continue
            cands = [t]                              # index 0 = target
            if distr is not None:
                for d in range(distr.shape[1]):
                    c = _proj(distr[i, d, :3], cam[i], W, H, fov)
                    if c is not None:
                        cands.append(c)
            if len(cands) < 2:
                continue
            c = np.asarray(cands, np.float32)         # (K, 3) u,v,depth
            u, v, dp = c[:, 0], c[:, 1], c[:, 2]
            un = (u - W / 2) / (W / 2); vn = (v - H / 2) / (H / 2)
            r = np.sqrt(un ** 2 + vn ** 2)
            crank = np.argsort(np.argsort(r)) / (len(c) - 1)      # 0 = most central
            drank = np.argsort(np.argsort(dp)) / (len(c) - 1)     # 0 = nearest
            feat = np.stack([un, vn, np.abs(un), np.abs(vn), r, dp / 50.0,
                             1.0 / np.maximum(dp, 1.0), crank, drank], -1).astype(np.float32)
            out.append((feat, 0))
    return out


def _mf_policy(data, key):
    """mis_follow of a fixed policy that picks argmin of key(feat)->per-candidate score."""
    wrong = 0
    for f, t in data:
        pick = int(np.argmin(key(f)))
        wrong += int(pick != t)
    return 100.0 * wrong / max(len(data), 1)


def train_position_classifier(tr, va, epochs=60, seed=0):
    allf = np.concatenate([f for f, _ in tr]); mu = allf.mean(0); sd = allf.std(0) + 1e-6
    nz = lambda s: [((f - mu) / sd, t) for f, t in s]
    tr, va = nz(tr), nz(va)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(tr[0][0].shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)

    def mf(data):
        w = 0
        with torch.no_grad():
            for f, t in data:
                if int(net(torch.from_numpy(f)).squeeze(-1).argmax()) != t:
                    w += 1
        return 100.0 * w / max(len(data), 1)

    for _ in range(epochs):
        idx = np.random.permutation(len(tr))
        for j in idx:
            f, t = tr[j]
            s = net(torch.from_numpy(f)).squeeze(-1)
            loss = nn.functional.cross_entropy(s.unsqueeze(0), torch.tensor([t]))
            opt.zero_grad(); loss.backward(); opt.step()
    return mf(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config.yaml")
    ap.add_argument("--split", default="val", help="eval split (classifier trains on 'train')")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tr = build(cfg, "train"); va = build(cfg, a.split)
    avg_k = np.mean([len(f) for f, _ in va]) if va else 0
    chance = 100.0 * (1 - np.mean([1.0 / len(f) for f, _ in va])) if va else 0
    print(f"[leak-probe] hard samples: train={len(tr)}  {a.split}={len(va)}  avg_candidates={avg_k:.2f}")
    print(f"  chance (random pick):         {chance:5.2f}%")
    print(f"  pick most-central:            {_mf_policy(va, lambda f: f[:, 4]):5.2f}%")   # r
    print(f"  pick nearest (min depth):     {_mf_policy(va, lambda f: f[:, 5]):5.2f}%")   # dp/50
    floor = train_position_classifier(tr, va)
    print(f"  OPTIMAL position-only MLP:    {floor:5.2f}%   <- positional-shortcut floor")
    print(f"\n  Interpretation: if this floor is LOW (<~15%), geometry alone solves target")
    print(f"  selection -> language is not load-bearing. Fix the data (decorrelate target")
    print(f"  position/depth/rank from target-ness) until this rises toward chance ({chance:.0f}%).")


if __name__ == "__main__":
    main()
