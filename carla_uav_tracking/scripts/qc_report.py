"""QC report for generated tracking episodes.

Verifies the properties the training data must have (design §5.1/§6.3/§7.4):
  - target stays in the camera frame,
  - >=2 *similar* distractors are actually co-visible with the target,
  - distractor tracks + episode attrs (language, seed, ...) are recorded,
  - no corrupt/black frames or NaNs.

Uses the recorded per-frame camera pose (state/cam_*) + CARLA's own transform math
to project world points into the image, so co-visibility is measured, not assumed.

Usage (inside the cyh-carla container):
    python scripts/qc_report.py --data /data/carla_data/pilot [--previews]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import carla  # only used for its transform/inverse-matrix math
import h5py
import numpy as np


def project(cam_pose, world_pt, w, h, fov=90.0):
    """Project a world point into image (u, v); return None if behind camera."""
    T = carla.Transform(
        carla.Location(float(cam_pose[0]), float(cam_pose[1]), float(cam_pose[2])),
        carla.Rotation(pitch=float(cam_pose[3]), yaw=float(cam_pose[4]), roll=0.0),
    )
    M = np.array(T.get_inverse_matrix())
    p = M @ np.array([world_pt[0], world_pt[1], world_pt[2], 1.0])
    x_c, y_c, z_c = p[0], p[1], p[2]        # CARLA cam frame: x fwd, y right, z up
    if x_c <= 0.1:
        return None
    f = w / (2.0 * np.tan(np.radians(fov) / 2.0))
    u = f * (y_c / x_c) + w / 2.0
    v = f * (-z_c / x_c) + h / 2.0
    return u, v


def in_frame(uv, w, h):
    return uv is not None and 0 <= uv[0] < w and 0 <= uv[1] < h


def qc_episode(path, previews_dir=None):
    with h5py.File(path, "r") as f:
        n, H, W = f["rgb"].shape[:3]
        attrs = {k: f.attrs[k] for k in f.attrs}
        distractors_meta = json.loads(attrs.get("distractors", "[]"))
        similar_idx = [i for i, d in enumerate(distractors_meta) if d.get("similar")]

        cam = np.stack([f["state/cam_x"][:], f["state/cam_y"][:], f["state/cam_z"][:],
                        f["state/cam_pitch"][:], f["state/cam_yaw"][:]], axis=1)
        tgt = np.stack([f["target/tx"][:], f["target/ty"][:], f["target/tz"][:]], axis=1)
        dpos = f["distractors/positions"][:, :, :3] if "distractors/positions" in f else \
            np.zeros((n, 0, 3))

        tgt_in = 0
        dist_in_counts, sim_in_counts = [], []
        for i in range(n):
            if in_frame(project(cam[i], tgt[i], W, H), W, H):
                tgt_in += 1
            din = sum(in_frame(project(cam[i], dpos[i, j], W, H), W, H)
                      for j in range(dpos.shape[1]))
            sin = sum(in_frame(project(cam[i], dpos[i, j], W, H), W, H)
                      for j in similar_idx)
            dist_in_counts.append(din)
            sim_in_counts.append(sin)

        # frame health
        sample = f["rgb"][:: max(1, n // 20)]
        black = float(np.mean(sample.reshape(sample.shape[0], -1).mean(1) < 5))
        nan_action = bool(np.isnan(f["action/dx"][:]).any())

        res = {
            "file": os.path.basename(path), "n": int(n), "res": f"{W}x{H}",
            "class": attrs.get("target_class"), "strategy": attrs.get("strategy"),
            "num_distractors": len(distractors_meta), "num_similar_meta": len(similar_idx),
            "target_in_frame_rate": tgt_in / max(n, 1),
            "mean_distractors_in_frame": float(np.mean(dist_in_counts)) if n else 0,
            "mean_similar_in_frame": float(np.mean(sim_in_counts)) if n else 0,
            "frames_with_2plus_similar": float(np.mean(np.array(sim_in_counts) >= 2)) if n else 0,
            "black_frame_rate": black, "nan_in_action": nan_action,
            "language": attrs.get("language", "")[:90],
        }

        if previews_dir is not None:
            _save_preview(f, path, cam, tgt, dpos, similar_idx, W, H, previews_dir)
    return res


def _save_preview(f, path, cam, tgt, dpos, similar_idx, W, H, out_dir):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    n = f["rgb"].shape[0]
    idxs = np.linspace(0, n - 1, min(9, n)).astype(int)
    tiles = []
    for i in idxs:
        img = f["rgb"][i][:, :, ::-1].copy()  # RGB->BGR for cv2
        uv = project(cam[i], tgt[i], W, H)
        if in_frame(uv, W, H):
            cv2.circle(img, (int(uv[0]), int(uv[1])), 8, (0, 255, 0), 2)
        for j in range(dpos.shape[1]):
            uv = project(cam[i], dpos[i, j], W, H)
            if in_frame(uv, W, H):
                col = (0, 255, 255) if j in similar_idx else (0, 0, 255)
                cv2.circle(img, (int(uv[0]), int(uv[1])), 5, col, 2)
        tiles.append(img)
    rows = [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles) - len(tiles) % 3, 3)]
    if rows:
        grid = np.vstack(rows)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(path).replace(".h5", ".png")), grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--previews", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.h5")))
    if not files:
        print(f"No episodes in {args.data}")
        return
    prev = os.path.join(args.data, "qc_previews") if args.previews else None
    rows = [qc_episode(p, prev) for p in files]

    print(f"\n=== QC: {len(rows)} episodes in {args.data} ===")
    for r in rows:
        print(f"\n{r['file']}  [{r['class']}/{r['strategy']}]  n={r['n']} {r['res']}")
        print(f"  target_in_frame       : {r['target_in_frame_rate']:.2%}")
        print(f"  distractors (meta/sim): {r['num_distractors']}/{r['num_similar_meta']}")
        print(f"  mean distractors in-frame: {r['mean_distractors_in_frame']:.2f}"
              f"  (similar: {r['mean_similar_in_frame']:.2f})")
        print(f"  frames w/ >=2 similar visible: {r['frames_with_2plus_similar']:.2%}")
        print(f"  black_frames={r['black_frame_rate']:.2%}  nan_action={r['nan_in_action']}")
        print(f"  lang: {r['language']}")

    agg = lambda k: float(np.mean([r[k] for r in rows]))
    print("\n=== AGGREGATE ===")
    print(f"  target_in_frame_rate         : {agg('target_in_frame_rate'):.2%}")
    print(f"  mean_similar_in_frame        : {agg('mean_similar_in_frame'):.2f}")
    print(f"  frames_with_2plus_similar    : {agg('frames_with_2plus_similar'):.2%}")
    print(f"  black_frame_rate             : {agg('black_frame_rate'):.2%}")
    if prev:
        print(f"  previews written to {prev}")


if __name__ == "__main__":
    main()
