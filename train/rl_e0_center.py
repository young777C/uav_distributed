"""Stage-3 RL — E0 gate: optimize R_center via a DIFFERENTIABLE multi-step rollout (SVG).

Why not GRPO-in-CARLA for E0: the harness (on-policy CARLA-in-the-loop) is the hard blocker
(stage3-rl-design §9). E0 only needs to answer "can optimizing R_center raise closed-loop
centering WITHOUT hacking?" — and the drone kinematics (drone.step: pos+=a·scale, yaw+=dyaw·scale),
the camera mount, the projection, and R_center are ALL differentiable, and the true target's
future trajectory is in the GT. So we roll the policy forward H steps through a KNOWN differentiable
dynamics model (no CARLA), sum R_center, and backprop to the DiT (last-4 + out; VLM/EAR/IAR frozen).

This is model-based analytic policy gradient (SVG). Unlike BC (imitates a per-frame centering
action → aggressive open-loop yaw → over-rotates on noisy z_ex → loses target; ruled out), the
MULTI-STEP return penalizes over-rotation (overshoot → target leaves frame → future R_center=0),
so the policy learns STABLE damped centering — the closed-loop constraint BC structurally lacked.
z_ex noise is injected during rollout to match the deploy reid/CV noise the BC blend sweep exposed.
R_center is on the TRUE target (GT, reward-only, never in obs) → anti-hack (centering a distractor
earns ~0). VLM feats (vlm_ctx,z_im) are held fixed over the short H (DiT tolerates staleness);
z_ex is recomputed differentiably each micro-step from the evolving camera + GT target.

Stages: check_proj (validate torch projection == numpy project_point) | train | (eval = CARLA
closed-loop via eval_bc_center.sh with the saved ckpt). Reuses the BC cache (vlm_ctx,z_im,proprio)
+ reconstructs each frame's (episode, index) in cache order to read cam pose + GT target from h5.
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

from acot_probe.projection import project_point  # numpy reference
from train.dataset import _select_episodes


# --------------------------------------------------------------------------- torch projection
def proj_torch(tw, cam_xyz, cam_pitch_deg, cam_yaw_deg, W, H, fov_deg):
    """Differentiable world->pixel, matching acot_probe.projection (CARLA conv, roll=0).

    tw: (...,3) world point(s); cam_xyz: (...,3); cam_pitch/yaw: (...) degrees.
    Returns (u, v, depth), each (...). depth<=0.1 => behind camera.
    """
    import torch
    d2r = math.pi / 180.0
    p = cam_pitch_deg * d2r
    y = cam_yaw_deg * d2r
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    dx = tw[..., 0] - cam_xyz[..., 0]
    dy = tw[..., 1] - cam_xyz[..., 1]
    dz = tw[..., 2] - cam_xyz[..., 2]
    depth = cp * cy * dx + cp * sy * dy + sp * dz          # q[0] forward
    right = -sy * dx + cy * dy                             # q[1]
    up = -cy * sp * dx - sy * sp * dy + cp * dz            # q[2]
    f = W / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    u = f * (right / depth) + W / 2.0
    v = f * (-up / depth) + H / 2.0
    return u, v, depth


def _cam_from_drone(drone_xyz, drone_yaw_deg, cam_pitch):
    """drone (x,y,z,yaw) -> camera (x,y,z) [pitch fixed, yaw=drone_yaw]. Matches drone.get_camera_transform."""
    import torch
    yr = drone_yaw_deg * (math.pi / 180.0)
    fwd_x, fwd_y = torch.cos(yr), torch.sin(yr)
    cam = torch.stack([drone_xyz[..., 0] - 0.5 * fwd_x,
                       drone_xyz[..., 1] - 0.5 * fwd_y,
                       drone_xyz[..., 2] - 0.3], dim=-1)
    return cam  # cam_yaw = drone_yaw, cam_pitch = fixed


def _drone_from_cam(cam_pose):
    """Invert _cam_from_drone: recover drone (x,y,z,yaw) from a stored cam pose (x,y,z,pitch,yaw)."""
    cx, cy, cz, cpitch, cyaw = cam_pose
    yr = math.radians(cyaw)
    dx = cx + 0.5 * math.cos(yr)
    dy = cy + 0.5 * math.sin(yr)
    dz = cz + 0.3
    return np.array([dx, dy, dz], np.float64), float(cyaw)


# --------------------------------------------------------------------------- cache-order (ep,i) reconstruction
def reconstruct_frame_index(cfg, episodes, stride, cap, H):
    """Replay cache_stage's exact iteration order (no VLM) -> list of (ep_path, i) per cache rec."""
    out = []
    for split in ["train", "val"]:
        eps = _select_episodes(cfg, split)[:episodes]
        for ep in eps:
            with h5py.File(ep, "r") as f:
                n = f["rgb"].shape[0]
            for i in list(range(0, n - H, stride))[:cap]:
                out.append((ep, i))
    return out


def check_proj(cfg, cache, episodes, stride, cap, H):
    import torch
    W, Hi, fov = cfg["image"]["width"], cfg["image"]["height"], cfg["image"]["fov_deg"]
    idx = reconstruct_frame_index(cfg, episodes, stride, cap, H)
    data = torch.load(cache, weights_only=False)["recs"]
    print(f"[check_proj] cache recs={len(data)}  reconstructed (ep,i)={len(idx)}  "
          f"{'ALIGNED' if len(data) == len(idx) else 'MISALIGNED!!'}", flush=True)
    errs = []
    open_eps = {}
    for j in np.linspace(0, len(idx) - 1, 300).astype(int):
        ep, i = idx[j]
        if ep not in open_eps:
            open_eps[ep] = h5py.File(ep, "r")
        f = open_eps[ep]
        cam = [float(f[f"state/cam_{k}"][i]) for k in ["x", "y", "z", "pitch", "yaw"]]
        tw = [float(f[f"target/t{k}"][i]) for k in ["x", "y", "z"]]
        ref = project_point(tuple(tw), tuple(cam), W, Hi, fov)
        if ref is None:
            continue
        u, v, dep = proj_torch(torch.tensor(tw), torch.tensor(cam[:3]),
                               torch.tensor(cam[3]), torch.tensor(cam[4]), W, Hi, fov)
        errs.append((abs(float(u) - ref[0]), abs(float(v) - ref[1]), abs(float(dep) - ref[2])))
    for f in open_eps.values():
        f.close()
    errs = np.array(errs)
    print(f"[check_proj] n={len(errs)}  max|Δu|={errs[:,0].max():.4f}px  max|Δv|={errs[:,1].max():.4f}px  "
          f"max|Δdepth|={errs[:,2].max():.5f}m", flush=True)
    ok = errs.max() < 1e-2
    print(f"[check_proj] {'PASS — torch projection matches numpy' if ok else 'FAIL — projection mismatch'}", flush=True)
    return ok


# --------------------------------------------------------------------------- differentiable rollout train
def train_e0(cfg, cache, ckpt, save_ckpt, device, episodes, stride, cap, H_data,
             roll_h, epochs, lr, last_layers, sigma_frac, zex_noise, batch,
             sample_steps=0, max_frames=0):
    import torch, time
    from rollout.policy import load_policy_modules
    from train.stage2 import action_sample  # noqa (kept for parity)

    W, Hi, fov = cfg["image"]["width"], cfg["image"]["height"], cfg["image"]["fov_deg"]
    a_scale = float(cfg["action"]["scale"])
    wp = cfg["waypoint"]; scale = float(wp["scale"]); fps = int(cfg.get("fps", 10))
    offs_frames = [int(o * fps) for o in wp["offsets_s"]]
    steps = sample_steps if sample_steps else int(cfg["flow"]["sample_steps"])

    idx = reconstruct_frame_index(cfg, episodes, stride, cap, H_data)
    data = torch.load(cache, weights_only=False)["recs"]
    assert len(data) == len(idx), f"cache/index misalign {len(data)} vs {len(idx)}"
    if max_frames and max_frames < len(data):   # subsample for tractable SVG (full 6282×48-fwd/batch too slow)
        np.random.seed(0)
        sub = sorted(np.random.choice(len(data), max_frames, replace=False).tolist())
        idx = [idx[j] for j in sub]; data = [data[j] for j in sub]
        print(f"[e0] subsampled to {len(data)} frames (seed 0); rollout sample_steps={steps}", flush=True)

    # preload per-frame rollout GT (cam pose + target world traj out to max horizon + roll_h) from h5
    maxh = max(offs_frames) + roll_h + 2
    cams, tgts, valid = [], [], []
    open_eps = {}
    for (ep, i) in idx:
        if ep not in open_eps:
            open_eps[ep] = h5py.File(ep, "r")
        f = open_eps[ep]
        n = f["rgb"].shape[0]
        cam = np.array([f[f"state/cam_{k}"][i] for k in ["x", "y", "z", "pitch", "yaw"]], np.float64)
        tt = np.stack([np.array([f[f"target/t{k}"][min(i + s, n - 1)] for s in range(maxh + 1)])
                       for k in ["x", "y", "z"]], -1)  # (maxh+1,3)
        cams.append(cam); tgts.append(tt.astype(np.float32)); valid.append(True)
    for f in open_eps.values():
        f.close()
    cams = np.stack(cams); tgts = np.stack(tgts)
    cam_pitch = cams[:, 3].astype(np.float32)

    ear, iar, dit, tid, d, meta = load_policy_modules(ckpt, cfg, device)
    K = int(d["K"])
    import re
    for p in dit.parameters():
        p.requires_grad_(False)
    names = [n for n, _ in dit.named_parameters()]
    blk = sorted(set(int(m.group(1)) for n in names for m in [re.search(r'blocks\.(\d+)\.', n)] if m))
    keep = set(blk[-last_layers:])
    trainable = [p for n, p in dit.named_parameters()
                 if (re.search(r'blocks\.(\d+)\.', n) and int(re.search(r'blocks\.(\d+)\.', n).group(1)) in keep)
                 or n.startswith("out.")]
    for p in trainable:
        p.requires_grad_(True)
    print(f"[e0] DiT last {last_layers} blocks + out trainable: {sum(p.numel() for p in trainable)} params; "
          f"roll_h={roll_h} sigma={sigma_frac} zex_noise={zex_noise}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    vlm_all = [torch.tensor(b["vlm_ctx"], dtype=torch.float32) for b in data]
    zim_all = [torch.tensor(b["z_im"], dtype=torch.float32) for b in data]
    prop_all = [torch.tensor(b["proprio"], dtype=torch.float32) for b in data]  # held fixed over short rollout
    T = lambda a: torch.tensor(a, device=device)

    N = len(data)
    nbatch = (N + batch - 1) // batch
    for ep in range(epochs):
        order = np.random.permutation(N)
        tot_r = 0.0; nb = 0; t0 = time.time()
        for s0 in range(0, N, batch):
            bi = order[s0:s0 + batch]
            B = len(bi)
            vc = torch.stack([vlm_all[j] for j in bi]).to(device)
            zim = torch.stack([zim_all[j] for j in bi]).to(device)
            prop = torch.stack([prop_all[j] for j in bi]).to(device)   # (B,5) fixed over rollout
            mask = torch.ones(vc.shape[:2], dtype=torch.bool, device=device)
            cpitch = T(cam_pitch[bi])                                   # (B,)
            tw = T(tgts[bi])                                            # (B, maxh+1, 3)
            # init drone state from stored cam pose
            dpos = np.stack([_drone_from_cam(cams[j])[0] for j in bi]).astype(np.float32)
            dyaw = np.array([cams[j][4] for j in bi], np.float32)
            dpos = T(dpos).float(); dyaw = T(dyaw).float()             # (B,3),(B,)
            R = 0.0
            for h in range(roll_h):
                cam_xyz = _cam_from_drone(dpos, dyaw, cpitch)          # (B,3)
                # z_ex = CV-of-true-target future at offsets, projected to cam frame /scale (+ noise)
                zex_rows = []
                for s in offs_frames:
                    tw_s = tw[:, min(s, tw.shape[1] - 1), :]           # (B,3) true future target
                    q = _world_to_cam(tw_s, cam_xyz, cpitch, dyaw)     # (B,3) cam-frame
                    zex_rows.append(q / scale)
                zex = torch.stack(zex_rows, dim=1)                     # (B,K,3)
                if zex_noise > 0:
                    zex = zex + zex_noise * torch.randn_like(zex)
                act = _flow_sample_grad(dit, zex, zim, vc, prop, mask, H_data, int(d["A"]), steps)  # (B,H,A)
                a0 = act[:, 0, :4] * a_scale                           # (B,4) dx,dy,dz,dyaw
                dpos = dpos + a0[:, :3]
                dyaw = dyaw + a0[:, 3]
                # reward: TRUE target centered in the NEW camera (anti-hack, GT)
                cam_xyz2 = _cam_from_drone(dpos, dyaw, cpitch)
                tgt_next = tw[:, min(h + 1, tw.shape[1] - 1), :]
                u, v, dep = proj_torch(tgt_next, cam_xyz2, cpitch, dyaw, W, Hi, fov)
                du = u / W - 0.5; dv = v / Hi - 0.5
                infront = (dep > 0.1).float()
                r_center = torch.exp(-(du * du + dv * dv) / (sigma_frac * sigma_frac)) * infront
                R = R + r_center.mean()
            loss = -(R / roll_h)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tot_r += float(R / roll_h); nb += 1
            if ep == 0 and nb % 20 == 0:
                el = time.time() - t0
                print(f"[e0]  ep1 batch {nb}/{nbatch}  R={tot_r/nb:.4f}  {el/nb:.2f}s/batch  "
                      f"eta_epoch~{el/nb*nbatch/60:.1f}min", flush=True)
        print(f"[e0] epoch {ep+1}/{epochs}  mean R_center={tot_r/max(nb,1):.4f}  ({(time.time()-t0)/60:.1f}min)", flush=True)

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck["dit"] = {k: v.cpu() for k, v in dit.state_dict().items()}
    torch.save(ck, save_ckpt)
    print(f"[e0] saved → {save_ckpt} (eval: eval_bc_center.sh ARMS=\"base:... e0:{save_ckpt}\")", flush=True)


def _flow_sample_grad(dit, z_ex, z_im, vlm_ctx, proprio, vlm_mask, h, a_dim, steps):
    """Grad-enabled flow-matching sampler (stage2.action_sample is @torch.no_grad — cuts the
    SVG graph). Same Euler integration; noise is the reparameterized draw, grad flows through dit."""
    import torch
    b = z_ex.shape[0]
    x = torch.randn(b, h, a_dim, device=z_ex.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((b,), i * dt, device=z_ex.device)
        x = x + dt * dit(x, t, z_ex, z_im, vlm_ctx, proprio, vlm_mask)
    return x


def _world_to_cam(tw, cam_xyz, cam_pitch_deg, cam_yaw_deg):
    """World point -> camera-local (x=forward,y=right,z=up), differentiable (matches inverse_pose_matrix)."""
    import torch
    d2r = math.pi / 180.0
    p = cam_pitch_deg * d2r; y = cam_yaw_deg * d2r
    cp, sp = torch.cos(p), torch.sin(p); cy, sy = torch.cos(y), torch.sin(y)
    dx = tw[..., 0] - cam_xyz[..., 0]; dy = tw[..., 1] - cam_xyz[..., 1]; dz = tw[..., 2] - cam_xyz[..., 2]
    fwd = cp * cy * dx + cp * sy * dy + sp * dz
    right = -sy * dx + cy * dy
    up = -cy * sp * dx - sy * sp * dy + cp * dz
    return torch.stack([fwd, right, up], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--ckpt", default="runs/stage2_v5/stage2_best.pt")
    ap.add_argument("--cache", default="runs/bc_center_cache2.pt")
    ap.add_argument("--stage", default="all", choices=["check_proj", "train", "all"])
    ap.add_argument("--save-ckpt", default="runs/stage2_rl_e0.pt")
    ap.add_argument("--episodes", type=int, default=999)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--data-h", type=int, default=None, help="H used when caching (drop last H frames); default=max offset")
    ap.add_argument("--roll-h", type=int, default=6, help="differentiable rollout horizon (steps)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--last-layers", type=int, default=4)
    ap.add_argument("--sigma-frac", type=float, default=0.25)
    ap.add_argument("--zex-noise", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--sample-steps", type=int, default=0, help="flow steps in rollout sampler (0=cfg; 2 is enough for E0, 4x faster)")
    ap.add_argument("--max-frames", type=int, default=0, help="subsample N frames for tractable SVG (0=all)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    # H_data MUST equal the H the cache used for frames=range(0,n-H,stride); it's stored in the cache dims
    import torch as _t
    H_data = a.data_h if a.data_h is not None else int(_t.load(a.cache, weights_only=False)["dims"]["H"])
    print(f"[e0] using H_data={H_data} (from cache dims); roll_h={a.roll_h}", flush=True)
    if a.stage in ("check_proj", "all"):
        ok = check_proj(cfg, a.cache, a.episodes, a.stride, a.cap, H_data)
        if not ok and a.stage == "all":
            print("[e0] projection check FAILED — aborting before train"); return
    if a.stage in ("train", "all"):
        train_e0(cfg, a.cache, a.ckpt, a.save_ckpt, a.device, a.episodes, a.stride, a.cap, H_data,
                 a.roll_h, a.epochs, a.lr, a.last_layers, a.sigma_frac, a.zex_noise, a.batch,
                 sample_steps=a.sample_steps, max_frames=a.max_frames)


if __name__ == "__main__":
    main()
