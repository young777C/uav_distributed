"""Baseline ladder for out-of-frame prediction — the 'is the reactive model at its
ceiling?' test. Runs a PER-EPISODE CAUSAL pass (stateful memory of the last-visible
target) and scores three velocity-free / low-cost predictors against the same GT
waypoints EAR is trained on, so they are directly comparable:

  ② zerovel : last-seen target world pos, frozen (zero velocity)          -> memory only
  ③ cv      : last-seen pos + last-seen world velocity * elapsed (const-v) -> memory + GT speed
  persist   : predict future = the 1s GT waypoint (GT-informed REFERENCE,  not inference-realizable)
  ear       : the trained EAR's own sampled waypoints (optional, --ckpt)

Everything is dead-reckoned in the WORLD frame then reprojected through the CURRENT
camera pose (inverse_pose_matrix), so UAV self-motion is fully compensated and the
③ velocity is the target's ABSOLUTE world velocity (target/tv{x,y,z}), not relative.

Reported STRATIFIED by  visible/loss  ×  loss-duration (short/long)  ×  turn/straight
(target heading change over the loss+horizon window), and each error split into
DEPTH (forward/optical axis, dim0) vs LATERAL (image-plane, dims1:2) — this is where
monocular depth ambiguity shows up. Expected: cv wins short+straight but BLOWS UP on
long+turn -> that residual is exactly the road-graph WM's value region.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from acot_probe.projection import inverse_pose_matrix
from train.dataset import _select_episodes
from train import labels


def _homog(p):                                   # (N,3) -> (N,4)
    return np.concatenate([p, np.ones((p.shape[0], 1), p.dtype)], -1)


def _to_cam(world_pts, cam_pose, scale):         # (N,3) world -> (N,3) cam-frame / scale
    M = inverse_pose_matrix(*cam_pose)           # world->local 4x4 (x=fwd/depth, y=right, z=up)
    return (_homog(world_pts.astype(np.float64)) @ M.T)[:, :3] / scale


def _dyaw(a, b):                                  # min angular diff (deg), wrapped
    return abs(((b - a + 180.0) % 360.0) - 180.0)


class _Acc:
    """Accumulate per-horizon FDE / lateral / depth errors (meters)."""
    def __init__(self, K):
        self.K = K
        self.fde = [[] for _ in range(K)]
        self.lat = [[] for _ in range(K)]
        self.dep = [[] for _ in range(K)]
        self.n = 0

    def add(self, pred, gt, scale):              # pred,gt: (K,3) scaled cam-frame
        diff = (pred - gt) * scale               # -> meters
        f = np.linalg.norm(diff, axis=-1)
        d = np.abs(diff[:, 0])                    # depth = forward axis
        l = np.linalg.norm(diff[:, 1:], axis=-1)  # lateral = image plane
        for k in range(self.K):
            self.fde[k].append(f[k]); self.lat[k].append(l[k]); self.dep[k].append(d[k])
        self.n += 1

    def mean(self, which, k):
        arr = getattr(self, which)[k]
        return float(np.mean(arr)) if arr else float("nan")


def _load_ear(ckpt_path, cfg, dev):
    import torch
    from train.ear import EAR
    ck = torch.load(ckpt_path, map_location="cpu")
    K = ck["dims"]["K"]; cond_dim = ck["dims"]["cond_dim"]; pdim = ck["dims"]["pdim"]
    m = cfg["model"]
    ear = EAR(cond_dim, k=K, d_model=m["ear"]["d_model"], n_layers=m["ear"]["n_layers"],
              n_heads=m["ear"]["n_heads"], proprio_dim=pdim).to(dev)
    ear.load_state_dict(ck["ear"]); ear.eval()
    return ear


def _ctx_loader(ctx_dir, stem):
    """Return (frames_array, ctx_mmap[n,L,M,C]) or None. Mirrors dataset._ctx."""
    npz = ctx_dir / (stem + ".npz")
    if not npz.exists():
        return None
    d = np.load(npz)
    frames = np.asarray(d["frames"])
    ctx_npy = npz.with_name(npz.stem + ".ctx.npy")
    ctx = np.load(ctx_npy, mmap_mode="r") if ctx_npy.exists() else np.asarray(d["ctx"])
    return frames, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--ckpt", default=None, help="EAR ckpt (stage2_*.pt) to also score EAR; omit = baselines only")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--long-thresh", type=float, default=3.0, help="loss-duration (s) split short/long")
    ap.add_argument("--turn-deg", type=float, default=30.0, help="target heading change (deg) -> 'turn'")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    fps = cfg.get("fps", 10)
    off_s = np.array(cfg["waypoint"]["offsets_s"], dtype=np.float64)   # [1,2,4,6] s
    off_f = (off_s * fps).astype(int)                                 # frame offsets
    K = len(off_s); scale = float(cfg["waypoint"]["scale"])
    max_off = int(off_f.max())
    eps = _select_episodes(cfg, a.split)

    ear = None; ctx_dir = None; steps = cfg["flow"]["sample_steps"]
    if a.ckpt:
        import torch
        dev = a.device if torch.cuda.is_available() else "cpu"
        ear = _load_ear(a.ckpt, cfg, dev)
        ctx_dir = Path(cfg["data"]["stage2_context_cache"])

    ACC = defaultdict(lambda: _Acc(K))            # key: (group, strat, method)
    dloss_hist = []                               # for coverage transparency

    for ep in eps:
        import h5py
        with h5py.File(ep, "r") as f:
            miss = [k for k in ("target/tvx", "target/tyaw") if k not in f]
            if miss:
                print(f"[dr] skip {Path(ep).name}: missing {miss}"); continue
            g = lambda k: np.asarray(f[k][:], np.float64)
            tgt = np.stack([g("target/tx"), g("target/ty"), g("target/tz")], -1)      # (T,3)
            vt = np.stack([g("target/tvx"), g("target/tvy"), g("target/tvz")], -1)    # (T,3) world m/s
            cam = np.stack([g("state/cam_x"), g("state/cam_y"), g("state/cam_z"),
                            g("state/cam_pitch"), g("state/cam_yaw")], -1)            # (T,5)
            uavv = np.stack([g("state/uav_vx"), g("state/uav_vy"), g("state/uav_vz")], -1)
            tyaw = g("target/tyaw")                                                   # (T,) deg
            vis = ((labels.combined_occlusion(f) < 0.8) &
                   (f["annotation/off_screen"][:] < 0.5
                    if "annotation/off_screen" in f else True))
            # WM M0: per-frame road-traversal prediction (K,3 world) + junction flag, if補标注 done.
            road_pred = g("annotation/road_pred") if "annotation/road_pred" in f else None   # (T,K,3)
            is_junc = (f["annotation/target_is_junction"][:] > 0.5
                       if "annotation/target_is_junction" in f else None)
        T = len(tgt)
        cache = _ctx_loader(ctx_dir, Path(ep).stem) if ear else None
        frames_cached = set(int(x) for x in cache[0]) if cache else set()
        frame_row = {int(x): r for r, x in enumerate(cache[0])} if cache else {}

        p_last = t_last = v_last = None
        ear_batch = []                            # (row, group, strat, gt, proprio5)
        for t in range(T - max_off):
            cam_t = tuple(cam[t])
            fut = tgt[np.minimum(t + off_f, T - 1)]                    # (K,3) world future
            gt = _to_cam(fut, cam_t, scale)                           # (K,3) == dataset waypoint
            is_vis = bool(vis[t])
            if is_vis:
                p_last, t_last, v_last = tgt[t], t, vt[t]
            if p_last is None:
                continue                                              # not seen yet this episode
            dloss_s = (t - t_last) / fps
            group = "vis" if is_vis else "loss"

            # turn = target heading change over [last-seen -> 6s-ahead] window
            yaw_end = tyaw[min(t + int(off_f[-1]), T - 1)]
            turn = "turn" if _dyaw(tyaw[t_last], yaw_end) > a.turn_deg else "straight"
            if group == "loss":
                dur = "long" if dloss_s > a.long_thresh else "short"
                strat = f"{dur}_{turn}"
                dloss_hist.append(dloss_s)
            else:
                strat = turn                                          # visible: turn/straight only

            # --- predictions (world -> current cam frame) ---
            ACC[(group, strat, "persist")].add(np.repeat(gt[0:1], K, axis=0), gt, scale)
            zc = _to_cam(p_last[None], cam_t, scale)                  # (1,3)
            ACC[(group, strat, "zerovel")].add(np.repeat(zc, K, axis=0), gt, scale)
            horizon_s = dloss_s + off_s                               # (K,) time since last-seen
            cvw = p_last[None] + v_last[None] * horizon_s[:, None]    # (K,3) world const-vel
            ACC[(group, strat, "cv")].add(_to_cam(cvw, cam_t, scale), gt, scale)

            # --- WM road-traversal (per-frame) vs matched straight (cv_frame) : H7 mechanism test ---
            # road_pred[t] and cv_frame both predict from the CURRENT frame's target state, so the
            # ONLY difference is "follow the lane" vs "go straight" -> isolates the road-graph value.
            # Also stratify by is_junction (precise turn label) alongside the tyaw-based strat.
            if road_pred is not None:
                jstrat = f"{group}_" + ("junction" if (is_junc is not None and is_junc[t]) else "road")
                road_cam = _to_cam(road_pred[t], cam_t, scale)                    # (K,3) lane-following
                cvf = tgt[t][None] + vt[t][None] * off_s[:, None]                 # (K,3) straight-from-current
                cvf_cam = _to_cam(cvf, cam_t, scale)
                for st in (strat, jstrat):
                    ACC[(group, st, "road")].add(road_cam, gt, scale)
                    ACC[(group, st, "cv_frame")].add(cvf_cam, gt, scale)

            if ear and t in frames_cached:
                prop5 = np.array([cam[t][2] / 30.0, cam[t][3] / 90.0,
                                  uavv[t][0] / 15.0, uavv[t][1] / 15.0, uavv[t][2] / 15.0], np.float32)
                ear_batch.append((frame_row[t], group, strat, gt, prop5))

        if ear and ear_batch:
            import torch
            from train.flow_matching import sample as ear_sample
            _, ctx_mmap = cache
            dev = next(ear.parameters()).device
            for s0 in range(0, len(ear_batch), 128):
                chunk = ear_batch[s0:s0 + 128]
                ctx = torch.tensor(np.stack([np.asarray(ctx_mmap[r][-1]) for r, *_ in chunk]),
                                   dtype=torch.float32, device=dev)       # (b,M,C) last layer
                cm = torch.ones(ctx.shape[:2], dtype=torch.bool, device=dev)
                prop = torch.tensor(np.stack([c[4] for c in chunk]), device=dev)
                with torch.no_grad():
                    pr = ear_sample(ear, ctx, cm, K, steps, proprio=prop).cpu().numpy()
                for bi, (_, group, strat, gt, _) in enumerate(chunk):
                    ACC[(group, strat, "ear")].add(pr[bi].astype(np.float64), gt, scale)

    # ------------------------------- report -------------------------------
    methods = (["ear", "persist", "zerovel", "cv"] if ear else ["persist", "zerovel", "cv"]) + ["cv_frame", "road"]
    hz = "  ".join(f"{int(o)}s" for o in off_s)
    print(f"\n[dead-reckon baseline]  split={a.split}  eps={len(eps)}  "
          f"long>{a.long_thresh}s  turn>{a.turn_deg}deg  ckpt={a.ckpt}")
    if dloss_hist:
        h = np.array(dloss_hist)
        print(f"  丢失时长覆盖: n={len(h)}  median={np.median(h):.1f}s  "
              f"p90={np.percentile(h,90):.1f}s  max={h.max():.1f}s  "
              f"(>{a.long_thresh}s 占 {100*(h>a.long_thresh).mean():.0f}%)")

    def block(group, strats):
        print(f"\n=== {group.upper()} 帧 (FDE / 深度 / 横向, 米) — 视界 {hz} ===")
        for strat in strats:
            present = [m for m in methods if (group, strat, m) in ACC and ACC[(group, strat, m)].n]
            if not present:
                continue
            n = ACC[(group, strat, present[0])].n
            print(f"  [{strat}]  n={n}")
            for m in present:
                A = ACC[(group, strat, m)]
                fde = "  ".join(f"{A.mean('fde',k):5.1f}" for k in range(K))
                dep = "  ".join(f"{A.mean('dep',k):5.1f}" for k in range(K))
                lat = "  ".join(f"{A.mean('lat',k):5.1f}" for k in range(K))
                print(f"    {m:8s} FDE {fde}   | 深度 {dep} | 横向 {lat}")

    block("vis", ["straight", "turn"])
    block("loss", ["short_straight", "short_turn", "long_straight", "long_turn"])
    print("\n判读: ①可见帧看 EAR 的[深度]是否远大于[横向](→14m 主要是单目深度歧义); "
          "②丢失帧看 cv 是否在 short_straight 最小、在 long_turn 崩塌(→WM 价值区)。")

    # ---- WM M2 gate: road-traversal vs matched straight (cv_frame), by junction ----
    has_road = any(k[2] == "road" for k in ACC)
    if has_road:
        print(f"\n=== WM M2 门 · road(沿车道) vs cv_frame(从当前帧直线) · by is_junction — 视界 {hz} ===")
        for group in ("loss", "vis"):
            for st in (f"{group}_junction", f"{group}_road"):
                r = ACC.get((group, st, "road")); c = ACC.get((group, st, "cv_frame"))
                if not (r and r.n and c and c.n):
                    continue
                rf = "  ".join(f"{r.mean('fde', k):5.1f}" for k in range(K))
                cf = "  ".join(f"{c.mean('fde', k):5.1f}" for k in range(K))
                dlt = "  ".join(f"{r.mean('fde', k)-c.mean('fde', k):+5.1f}" for k in range(K))
                print(f"  [{st}]  n={r.n}\n    road     FDE {rf}\n    cv_frame FDE {cf}\n    Δ(road-cv) {dlt}")
        print("  ★ M2 判据: **junction 帧上 road-FDE < cv_frame-FDE**(Δ 为负,尤其 4/6s)→ H7 几何成立、WM 有据;"
              "road 帧(直路)应 ≈(路网≈直线)。")


if __name__ == "__main__":
    main()
