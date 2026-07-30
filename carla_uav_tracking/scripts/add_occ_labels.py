"""A1c ② (offline, additive): add separated occlusion labels to existing episodes.

Adds three NON-overlapping annotation datasets, computed from the already-stored
correct-projection target bbox (annotation/bbox_*) and structural raycast
(state/occ_raycast) — so the IAR occlusion head can learn structural occlusion
instead of "out of frame":

  annotation/occ_structural [0,1]  in-frame, hidden by geometry (raycast, gated to in-frame)
  annotation/off_screen     {0,1}  target behind camera or projected outside the image
  annotation/occluded       {0,1}  "not usefully visible" mask = (occ_structural>=0.5) | off_screen

Purely additive: RGB, actions, waypoints, existing occlusion/bbox are NOT touched.
Idempotent (deletes+rewrites only the three fields). Safe to re-run.

Usage (inside the container):
    python scripts/add_occ_labels.py /data/mvp
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recording.postprocess import compute_occlusion_labels  # noqa: E402


def process(fp: Path) -> dict:
    with h5py.File(fp, "r+") as f:
        n, H, W = (int(f["rgb"].shape[0]), int(f["rgb"].shape[1]), int(f["rgb"].shape[2]))
        bu = f["annotation/bbox_u"][:n]
        bv = f["annotation/bbox_v"][:n]
        bw = f["annotation/bbox_w"][:n]
        ray = f["state/occ_raycast"][:n].astype(np.float32) if "state/occ_raycast" in f else None
        occ_struct, off_screen, occluded = compute_occlusion_labels(bu, bv, bw, ray, W, H)

        # waypoints must stay valid through occlusion (EAR through-occlusion supervision)
        wp = f["annotation/waypoints"][:n] if "annotation/waypoints" in f else np.zeros((n, 9))
        wp_bad_in_occ = int(np.isnan(wp).any(axis=1)[occluded > 0].sum())

        for name, data in (("annotation/occ_structural", occ_struct),
                           ("annotation/off_screen", off_screen),
                           ("annotation/occluded", occluded)):
            if name in f:
                del f[name]
            f.create_dataset(name, data=data.astype(np.float32),
                             compression="gzip", compression_opts=4)

    return {
        "n": n,
        "struct_frac": float((occ_struct >= 0.3).mean()),
        "off_frac": float(off_screen.mean()),
        "occluded_frac": float(occluded.mean()),
        "wp_bad_in_occ": wp_bad_in_occ,
    }


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/mvp")
    files = sorted(root.glob("*.h5"))
    if not files:
        print(f"no .h5 in {root}")
        return
    agg = {"struct": [], "off": [], "occ": [], "wp_bad": 0}
    for fp in files:
        r = process(fp)
        agg["struct"].append(r["struct_frac"])
        agg["off"].append(r["off_frac"])
        agg["occ"].append(r["occluded_frac"])
        agg["wp_bad"] += r["wp_bad_in_occ"]
    print(f"episodes: {len(files)}")
    print(f"occ_structural>0.3  frames: {100*np.mean(agg['struct']):.2f}%")
    print(f"off_screen          frames: {100*np.mean(agg['off']):.2f}%")
    print(f"occluded (loss mask) frames: {100*np.mean(agg['occ']):.2f}%")
    print(f"waypoints NaN within occluded frames (should be 0): {agg['wp_bad']}")


if __name__ == "__main__":
    main()
