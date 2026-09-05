"""BC-centering: fine-tune the DiT (last layers) to CENTER the target — the Stage-3 de-risk.

Diagnosis (memory acot-uav-reid-reframe): the deployable SR gap is the DiT framing the target
OFF-center (decorrelate design) → reid degrades → cascade. frame-gain (fixed geometric gain,
either sign) fails: too weak to move the DiT + amplifies reid errors. So the fix must be a
LEARNED, in-DiT centering. This script BC-fine-tunes the DiT (flow-matching, last layers only,
VLM/EAR/IAR frozen) toward a CENTERING action target — teaching the DiT to output actions that
point the camera at the (z_ex-indicated) target.

Centering target (validated: corr(off, yaw_err)=+0.677, sign +off=toward-target):
  a1[k] = [recorded dx,dy,dz at i+k ; dyaw_center[k] ; recorded search]
  dyaw_center[k] = clip(yaw_err(target[i+k], cam[i+k]), ±cap) / a_scale
DECOUPLES the decorrelate tension: off-center TRAINING DATA stays (VLM/reid features frozen,
identity load-bearing); only the DiT CONTROL is fine-tuned to center → does NOT re-introduce
the shortcut. z_ex during training = CV of the TRUE target (so the DiT learns "center what z_ex
points at"); at inference z_ex = reid-committed target.

Stages: cache (VLM/IAR/EAR frozen forward + centering target → .pt) | train (flow-matching
fine-tune DiT last layers) | saves a stage2-format ckpt to eval closed-loop.
Runs in the acot-uav-train container (VLM + torch + the regenerated mvp_full_v5 data).
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml
import h5py

_CARLA_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "carla_uav_tracking")
for _p in (os.path.dirname(_CARLA_PKG), _CARLA_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from acot_probe.projection import project_point, inverse_pose_matrix
from train.dataset import _select_episodes


def _norm180(a):
    return (a + 180.0) % 360.0 - 180.0


def _centering_target(tgt, cam, act, i, H, W, fov, a_scale, yaw_cap_deg, blend=1.0):
    """(H, 5) centering action chunk: recorded dx,dy,dz + blended centering dyaw + recorded search.

    blend=1.0 → full geometric centering yaw (aggressive; destabilized tracking in the 4-ep smoke:
    track_s 33.8→2.2 — over-rotates on noisy CV z_ex). blend<1.0 → dyaw = b·center + (1-b)·recorded,
    a GENTLER nudge toward center that preserves the expert's coordinated (yaw,translation) so it
    doesn't swing the target out of frame.
    """
    out = np.zeros((H, 5), np.float32)
    n = len(tgt)
    for k in range(H):
        j = min(i + k, n - 1)
        cx, cy, cz, cp, cyaw = cam[j]
        desired = math.degrees(math.atan2(tgt[j][1] - cy, tgt[j][0] - cx))
        yaw_err = _norm180(desired - cyaw)
        dyaw_c = float(np.clip(yaw_err, -yaw_cap_deg, yaw_cap_deg) / a_scale)
        out[k, 0], out[k, 1], out[k, 2] = act[j, 0], act[j, 1], act[j, 2]   # recorded dx,dy,dz
        out[k, 3] = blend * dyaw_c + (1.0 - blend) * float(act[j, 3])       # blended CENTERING dyaw
        out[k, 4] = act[j, 4] if act.shape[1] > 4 else 0.0                   # recorded search
    return out


def cache_stage(cfg, out_path, device, episodes, stride, cap):
    import torch
    from rollout.online_vlm import OnlineVLM
    from rollout.policy import load_policy_modules
    from rollout.target_state import TargetStateEstimator
    from train import flow_matching

    vlm = OnlineVLM(cfg, device=device, language_mode="neutral")
    ear, iar, dit, tid, d, meta = load_policy_modules(cfg["_ckpt"], cfg, device)
    K, H, A = int(d["K"]), int(d["H"]), int(d["A"])
    W, Himg, fov = cfg["image"]["width"], cfg["image"]["height"], cfg["image"]["fov_deg"]
    a_scale = float(cfg["action"]["scale"]); yaw_cap = float(cfg.get("_yaw_cap_deg", 20.0))
    wp = cfg["waypoint"]; est = TargetStateEstimator(wp["offsets_s"], wp["scale"], cfg.get("fps", 10))
    steps = int(cfg["flow"]["sample_steps"])

    recs = []
    for split_id, split in enumerate(["train", "val"]):
        eps = _select_episodes(cfg, split)[:episodes]
        for ei, ep in enumerate(eps):
            with h5py.File(ep, "r") as f:
                n = f["rgb"].shape[0]
                lang = f.attrs.get("language", ""); lang = lang.decode() if isinstance(lang, bytes) else str(lang)
                tgt = np.stack([f["target/tx"][:], f["target/ty"][:], f["target/tz"][:]], -1)
                tv = np.stack([f["target/tvx"][:], f["target/tvy"][:], f["target/tvz"][:]], -1)
                cam = np.stack([f[f"state/cam_{k}"][:] for k in ["x", "y", "z", "pitch", "yaw"]], -1)
                uavv = np.stack([f["state/uav_vx"][:], f["state/uav_vy"][:], f["state/uav_vz"][:]], -1)
                act = np.stack([f[f"action/{k}"][:] for k in ["dx", "dy", "dz", "dyaw"]], -1)
                act = np.concatenate([act, np.zeros((n, 1), np.float32)], -1)   # +search placeholder
                offs_s = np.array(wp["offsets_s"], np.float64)
                frames = list(range(0, n - H, stride))[:cap]
                for i in frames:
                    import torch as T
                    layers, vlm_ctx, grid = vlm.encode(np.asarray(f["rgb"][i]), lang)
                    ctx_layers = [x[None] for x in layers]
                    with T.no_grad():
                        z_im, _ = iar(ctx_layers, T.ones(1, vlm_ctx.shape[0], dtype=T.bool, device=device))
                    # z_ex = CV of the TRUE target (world → cam frame) — teaches "center what z_ex points at"
                    hor = offs_s
                    pw = tgt[i][None] + tv[i][None] * hor[:, None]              # (K,3)
                    M = inverse_pose_matrix(*tuple(cam[i]))
                    z_ex = ((np.concatenate([pw, np.ones((K, 1))], -1) @ M.T)[:, :3] / wp["scale"]).astype(np.float32)
                    prop = np.array([cam[i][2] / 30.0, cam[i][3] / 90.0,
                                     uavv[i][0] / 15.0, uavv[i][1] / 15.0, uavv[i][2] / 15.0], np.float32)
                    a1 = _centering_target(tgt, cam, act, i, H, W, fov, a_scale, yaw_cap, blend=1.0)
                    rec_dyaw = np.array([act[min(i + k, n - 1), 3] for k in range(H)], np.float32)  # for train-time blend
                    pr = project_point(tuple(float(x) for x in tgt[i]), tuple(cam[i]), W, Himg, fov)
                    visible = pr is not None and 0 <= pr[0] < W
                    recs.append({"vlm_ctx": vlm_ctx.half().cpu().numpy(),
                                 "z_im": z_im[0].half().cpu().numpy(), "z_ex": z_ex,
                                 "proprio": prop, "a1": a1, "rec_dyaw": rec_dyaw, "visible": visible})
            print(f"[cache] {split} ep{ei} {Path(ep).name}: n={len(recs)}", flush=True)
    import torch
    torch.save({"recs": recs, "dims": {"K": K, "H": H, "A": A}}, out_path)
    print(f"[cache] wrote {out_path}: {len(recs)} frames", flush=True)


def train_stage(cfg, cache, device, epochs, lr, last_layers, save_ckpt, blend=1.0):
    import torch
    from rollout.policy import load_policy_modules
    from train.stage2 import action_flow_loss
    ear, iar, dit, tid, d, meta = load_policy_modules(cfg["_ckpt"], cfg, device)
    data = torch.load(cache, weights_only=False)["recs"]
    if blend < 1.0:   # train-time gentler blend: dyaw = b·center + (1-b)·recorded (no re-cache)
        for b in data:
            b["a1"] = b["a1"].copy()
            b["a1"][:, 3] = blend * b["a1"][:, 3] + (1.0 - blend) * b["rec_dyaw"]
        print(f"[train] applied center-blend={blend} (dyaw = {blend}·center + {1-blend:.2f}·recorded)", flush=True)
    # freeze all but the last `last_layers` DiT blocks + the output head (VLM/EAR/IAR already frozen)
    import re
    for p in dit.parameters():
        p.requires_grad_(False)
    names = [n for n, _ in dit.named_parameters()]
    idxs = sorted(set(int(m.group(1)) for n in names for m in [re.search(r'blocks\.(\d+)\.', n)] if m))
    keep = set(idxs[-last_layers:]) if idxs else set()
    trainable = []
    for n, p in dit.named_parameters():
        m = re.search(r'blocks\.(\d+)\.', n)
        unfreeze = (m and int(m.group(1)) in keep) or n.startswith("out.")
        if not idxs:               # fallback: no block naming → DiT-only full fine-tune (still freezes VLM/EAR/IAR)
            unfreeze = True
        if unfreeze:
            p.requires_grad_(True); trainable.append(p)
    print(f"[train] DiT blocks={idxs}, unfreezing last {last_layers} + head → {sum(p.numel() for p in trainable)} params", flush=True)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    B = 32
    for ep in range(epochs):
        tot = 0.0; nb = 0
        order = np.random.permutation(len(data))
        for s0 in range(0, len(data), B):
            bat = [data[o] for o in order[s0:s0 + B]]
            vc = torch.tensor(np.stack([b["vlm_ctx"] for b in bat]), dtype=torch.float32, device=device)
            zim = torch.tensor(np.stack([b["z_im"] for b in bat]), dtype=torch.float32, device=device)
            zex = torch.tensor(np.stack([b["z_ex"] for b in bat]), dtype=torch.float32, device=device)
            prop = torch.tensor(np.stack([b["proprio"] for b in bat]), dtype=torch.float32, device=device)
            a1 = torch.tensor(np.stack([b["a1"] for b in bat]), dtype=torch.float32, device=device)
            vis = torch.tensor(np.array([b["visible"] for b in bat]), dtype=torch.float32, device=device)
            mask = torch.ones(vc.shape[:2], dtype=torch.bool, device=device)
            loss = action_flow_loss(dit, a1, zex, zim, vc, prop, mask, vis)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"[train] epoch {ep+1}/{epochs}  action_flow={tot/max(nb,1):.4f}", flush=True)
    # save a stage2-format ckpt (only DiT changed)
    ck = torch.load(cfg["_ckpt"], map_location="cpu", weights_only=False)
    ck["dit"] = {k: v.cpu() for k, v in dit.state_dict().items()}
    torch.save(ck, save_ckpt)
    print(f"[train] saved fine-tuned ckpt → {save_ckpt} (eval closed-loop with --ckpt {save_ckpt})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--ckpt", default="runs/stage2_v5/stage2_best.pt")
    ap.add_argument("--stage", default="all", choices=["cache", "train", "all"])
    ap.add_argument("--cache", default="runs/bc_center_cache.pt")
    ap.add_argument("--save-ckpt", default="runs/stage2_bc_center.pt")
    ap.add_argument("--episodes", type=int, default=999)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--last-layers", type=int, default=4)
    ap.add_argument("--yaw-cap-deg", type=float, default=20.0)
    ap.add_argument("--center-blend", type=float, default=1.0,
                    help="dyaw target = b·centering + (1-b)·recorded; <1 = gentler (full 1.0 destabilized track)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); cfg["_ckpt"] = a.ckpt; cfg["_yaw_cap_deg"] = a.yaw_cap_deg
    cfg["_center_blend"] = a.center_blend
    if a.stage in ("cache", "all"):
        if not _select_episodes(cfg, "train"):
            print("[cache] NO DATA — regenerate mvp per acot_note/mvp-regen-spec.md"); return
        cache_stage(cfg, a.cache, a.device, a.episodes, a.stride, a.cap)
    if a.stage in ("train", "all"):
        train_stage(cfg, a.cache, a.device, a.epochs, a.lr, a.last_layers, a.save_ckpt, blend=a.center_blend)


if __name__ == "__main__":
    main()
