"""Render episode HDF5 -> MP4 for visual inspection.

Overlays (default on): green=target, yellow=similar look-alike, red=distinct
distractor (projected from recorded camera pose). Top-left shows the target desc.

Usage (inside the cyh-carla container):
    python scripts/render_video.py --data /data/cov --out /data/cov/videos --fps 15
    python scripts/render_video.py --data /data/cov/episode_000003.h5 --raw   # no overlay
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import carla
import cv2
import h5py
import numpy as np


def project(cam_pose, wp, w, h, fov=90.0):
    T = carla.Transform(
        carla.Location(float(cam_pose[0]), float(cam_pose[1]), float(cam_pose[2])),
        carla.Rotation(pitch=float(cam_pose[3]), yaw=float(cam_pose[4]), roll=0.0))
    M = np.array(T.get_inverse_matrix())
    p = M @ np.array([wp[0], wp[1], wp[2], 1.0])
    x_c, y_c, z_c = p[0], p[1], p[2]
    if x_c <= 0.1:
        return None
    f = w / (2.0 * np.tan(np.radians(fov) / 2.0))
    return (f * (y_c / x_c) + w / 2.0, f * (-z_c / x_c) + h / 2.0)


def render(path, out_path, fps, overlay, maxf):
    with h5py.File(path, "r") as f:
        n, H, W = f["rgb"].shape[:3]
        n = min(n, maxf) if maxf else n
        attrs = {k: f.attrs[k] for k in f.attrs}
        meta = json.loads(attrs.get("distractors", "[]"))
        sim = {i for i, d in enumerate(meta) if d.get("similar")}
        cam = np.stack([f["state/cam_x"][:n], f["state/cam_y"][:n], f["state/cam_z"][:n],
                        f["state/cam_pitch"][:n], f["state/cam_yaw"][:n]], axis=1)
        tgt = np.stack([f["target/tx"][:n], f["target/ty"][:n], f["target/tz"][:n]], axis=1)
        dpos = f["distractors/positions"][:n, :, :3] if "distractors/positions" in f \
            else np.zeros((n, 0, 3))
        # A1c② occlusion labels (if present) → colour the target marker by state.
        occl = f["annotation/occluded"][:n] if "annotation/occluded" in f else np.zeros(n)
        ostr = f["annotation/occ_structural"][:n] if "annotation/occ_structural" in f else np.zeros(n)
        offs = f["annotation/off_screen"][:n] if "annotation/off_screen" in f else np.zeros(n)
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for i in range(n):
            img = np.ascontiguousarray(f["rgb"][i][:, :, ::-1])  # RGB -> BGR
            if overlay:
                occluded = occl[i] > 0
                # BGR: red when occluded (marker sits on the occluding structure), else green
                tcol = (0, 0, 255) if occluded else (0, 255, 0)
                uv = project(cam[i], tgt[i], W, H)
                if uv and 0 <= uv[0] < W and 0 <= uv[1] < H:
                    cv2.circle(img, (int(uv[0]), int(uv[1])), 9, tcol, -1 if occluded else 2)
                for j in range(dpos.shape[1]):
                    uv = project(cam[i], dpos[i, j], W, H)
                    if uv and 0 <= uv[0] < W and 0 <= uv[1] < H:
                        col = (0, 255, 255) if j in sim else (0, 0, 255)
                        cv2.circle(img, (int(uv[0]), int(uv[1])), 6, col, 2)
                cv2.putText(img, str(attrs.get("target_desc", ""))[:40], (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                state = ("OFF-SCREEN" if offs[i] > 0 else
                         ("OCCLUDED %.2f" % float(ostr[i]) if ostr[i] > 0.3 else "visible"))
                cv2.putText(img, state, (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            tcol, 1, cv2.LINE_AA)
            vw.write(img)
        vw.release()
    return out_path, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="HDF5 file or dir of episodes")
    ap.add_argument("--out", default=None, help="output dir (default: <data>/videos)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--raw", action="store_true", help="no overlay (what the model sees)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    files = [args.data] if args.data.endswith(".h5") else sorted(glob.glob(os.path.join(args.data, "*.h5")))
    out_dir = args.out or os.path.join(os.path.dirname(files[0]) or ".", "videos")
    os.makedirs(out_dir, exist_ok=True)
    for p in files:
        name = os.path.basename(p).replace(".h5", ("_raw" if args.raw else "") + ".mp4")
        out = os.path.join(out_dir, name)
        _, n = render(p, out, args.fps, not args.raw, args.max_frames)
        print(f"  {name}: {n} frames @ {args.fps}fps -> {out}")
    print(f"videos in {out_dir}")


if __name__ == "__main__":
    main()
