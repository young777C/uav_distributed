"""Visualize a generated tracking episode — RGB frames + state overlay.

Usage:
    python scripts/visualize_episode.py --episode ./data/raw/episode_000000.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize tracking episode")
    parser.add_argument("--episode", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", type=str, default=None,
                        help="Save as video instead of display")
    args = parser.parse_args()

    ep_path = Path(args.episode)
    if not ep_path.exists():
        print(f"Episode not found: {ep_path}")
        return

    with h5py.File(ep_path, "r") as f:
        n_steps = f["rgb"].shape[0]
        print(f"Episode: {ep_path.name}, {n_steps} steps")

        frames = []
        for i in range(n_steps):
            rgb = f["rgb"][i]  # (H, W, 3) uint8
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Overlay state info
            uav_x = f["state/uav_x"][i]
            uav_y = f["state/uav_y"][i]
            uav_z = f["state/uav_z"][i]
            tx = f["target/tx"][i]
            ty = f["target/ty"][i]
            action = [f["action/dx"][i], f["action/dy"][i],
                       f["action/dz"][i], f["action/dyaw"][i]]

            # Get annotations if available
            occlusion = 0.0
            if "annotation/occlusion" in f:
                occlusion = f["annotation/occlusion"][i]
            search_mode = 0.0
            if "annotation/search_mode" in f:
                search_mode = f["annotation/search_mode"][i]
            bbox = None
            if "annotation/bbox_u" in f:
                bbox = [
                    f["annotation/bbox_u"][i],
                    f["annotation/bbox_v"][i],
                    f["annotation/bbox_w"][i],
                    f["annotation/bbox_h"][i],
                ]

            # Draw bbox
            if bbox is not None and bbox[0] > 0:
                u, v, w, h = [int(x) for x in bbox]
                cv2.rectangle(rgb_bgr, (u - w//2, v - h//2),
                              (u + w//2, v + h//2), (0, 255, 0), 2)

            # Draw state text
            cv2.putText(rgb_bgr, f"UAV: ({uav_x:.0f},{uav_y:.0f},{uav_z:.0f})",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(rgb_bgr, f"TGT: ({tx:.0f},{ty:.0f})",
                        (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(rgb_bgr, f"Act: dx={action[0]:.1f} dy={action[1]:.1f} dz={action[2]:.1f}",
                        (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(rgb_bgr, f"Occ: {occlusion:.2f} Search: {search_mode:.2f}",
                        (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255) if occlusion > 0.5 else (0, 255, 0), 1)

            frames.append(rgb_bgr)

            if args.output is None:
                cv2.imshow("Tracking Episode", rgb_bgr)
                if cv2.waitKey(1000 // args.fps) == 27:  # ESC
                    break

    if args.output:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()
        print(f"Saved: {args.output}")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
