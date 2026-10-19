"""Convert recorded episode_*.h5 (dataset["rgb"], (N,H,W,3) uint8) to .mp4 for human viewing,
optionally burning in target/similar/distractor bounding boxes for annotation QA.

Uses imageio[pyav] (self-contained H.264 — the host/training-image ffmpeg binaries here are
stripped builds with no libx264/rawvideo support, confirmed 2026-10-04). Install once:
    pip install imageio av pillow

    python -m carla_uav_tracking.scripts.h5_to_mp4 \
        --src /nvidia/hque/data/carla_data/mvp_full_v5 \
        --out runs/mp4_preview/mvp_full_v5 --fps 10 --workers 8 --boxes

Box source/convention (matches rollout/candidates.py): annotation/bbox_{u,v,w,h} = target box
CENTER (u,v) + square SIZE (w==h); distractors/bbox (N,D,4) same per-distractor convention;
distractors/similar (D,) is fixed per episode (1=look-alike, 0=plain distractor). Off-screen
boxes are skipped by requiring the box to overlap the frame rectangle (not via off_screen flag
alone, since distractor boxes have no such flag)."""
from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

_COLORS = {"target": (40, 220, 40), "similar": (255, 60, 60), "distractor": (255, 210, 0)}
_STATUS_COLORS = {"visible": (40, 220, 40), "occluded": (255, 165, 0), "off-screen": (255, 60, 60)}


def _draw_box(draw, u, v, w, h, frame_w, frame_h, color, label):
    import math
    if not all(math.isfinite(x) for x in (u, v, w, h)):
        return   # NaN/inf (e.g. target fully behind camera, undefined projection) — nothing to draw
    l, t, r, b = u - w / 2, v - h / 2, u + w / 2, v + h / 2
    if r <= 0 or b <= 0 or l >= frame_w or t >= frame_h or w <= 0 or h <= 0:
        return   # fully off-frame — don't draw (also filters the "huge/negative" off-screen rows)
    draw.rectangle([l, t, r, b], outline=color, width=2)
    draw.text((max(l, 0) + 2, max(t, 0) - 11), label, fill=color)


def _frame_status(off_screen_i: float, occluded_i: float) -> str:
    # off-screen (fully out of frame) takes priority over occluded (in-frame but blocked) —
    # both annotation/off_screen and annotation/occluded are already-computed per-frame flags
    # (train/labels.py::combined_occlusion folds the same raycast signal into `occluded`).
    if off_screen_i:
        return "off-screen"
    if occluded_i:
        return "occluded"
    return "visible"


def _annotate_frames(f, n, frame_w, frame_h):
    """Yields annotated (H,W,3) uint8 frames, one at a time (keeps peak memory to ~1 frame)."""
    import numpy as np
    from PIL import Image, ImageDraw

    tu, tv = f["annotation/bbox_u"][:], f["annotation/bbox_v"][:]
    tw, th = f["annotation/bbox_w"][:], f["annotation/bbox_h"][:]
    off_screen = f["annotation/off_screen"][:] if "annotation/off_screen" in f else np.zeros(n)
    occluded = f["annotation/occluded"][:] if "annotation/occluded" in f else np.zeros(n)
    has_d = "distractors/bbox" in f
    dbox = f["distractors/bbox"][:] if has_d else None             # (N,D,4)
    similar = f["distractors/similar"][:] if "distractors/similar" in f else None  # (D,)
    rgb = f["rgb"]
    for i in range(n):
        img = Image.fromarray(np.asarray(rgb[i]))
        draw = ImageDraw.Draw(img)
        if not off_screen[i]:
            _draw_box(draw, tu[i], tv[i], tw[i], th[i], frame_w, frame_h, _COLORS["target"], "target")
        if has_d:
            for d in range(dbox.shape[1]):
                kind = "similar" if (similar is not None and similar[d]) else "distractor"
                _draw_box(draw, *dbox[i, d], frame_w, frame_h, _COLORS[kind], kind)
        status = _frame_status(off_screen[i], occluded[i])
        draw.rectangle([0, 0, 92, 13], fill=(0, 0, 0))
        draw.text((3, 2), status, fill=_STATUS_COLORS[status])
        yield np.asarray(img)


def convert_one(h5_path: str, out_path: str, fps: int, qscale: int, boxes: bool) -> tuple[str, str]:
    import h5py
    import imageio.v3 as iio

    if os.path.exists(out_path):
        return h5_path, "skip (exists)"
    try:
        with h5py.File(h5_path, "r") as f:
            if "rgb" not in f:
                return h5_path, "skip (no rgb dataset)"
            rgb = f["rgb"]
            n, frame_h, frame_w = rgb.shape[0], rgb.shape[1], rgb.shape[2]
            tmp = out_path + ".tmp.mp4"
            # imageio's pyav plugin (no usable system ffmpeg here, confirmed 2026-10-04) doesn't
            # expose a crf/quality knob in this version — `qscale` unused; libx264 defaults are
            # fine for review-quality playback.
            del qscale
            if boxes and "annotation/bbox_u" in f:
                iio.imwrite(tmp, list(_annotate_frames(f, n, frame_w, frame_h)), fps=fps,
                           codec="libx264", plugin="pyav")
            else:
                # v3.imwrite wants the WHOLE (N,H,W,3) batch array for the unannotated path (the
                # v2 append_data legacy wrapper mis-batches single frames in this imageio version).
                iio.imwrite(tmp, rgb[:], fps=fps, codec="libx264", plugin="pyav")
            os.replace(tmp, out_path)
            return h5_path, f"ok ({n} frames, {n / fps:.0f}s)"
    except Exception as e:  # noqa: BLE001 — report per-episode, don't kill the batch
        return h5_path, f"FAIL {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of episode_*.h5 (or a glob pattern)")
    ap.add_argument("--out", required=True, help="output dir for .mp4 (mirrors episode basenames)")
    ap.add_argument("--fps", type=int, default=10, help="recording fps (10 for mvp_full_v5)")
    ap.add_argument("--qscale", type=int, default=23, help="libx264 crf, lower=better/bigger (18-28 typical)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="only convert the first N episodes (0=all)")
    ap.add_argument("--boxes", action="store_true",
                    help="burn in target(green)/similar(red)/distractor(yellow) boxes + labels "
                         "from annotation/* + distractors/* — for annotation QA")
    a = ap.parse_args()

    pattern = a.src if "*" in a.src else os.path.join(a.src, "episode_*.h5")
    files = sorted(glob.glob(pattern))
    if a.limit:
        files = files[:a.limit]
    if not files:
        print(f"[h5_to_mp4] no files matched {pattern}", file=sys.stderr)
        return 1
    os.makedirs(a.out, exist_ok=True)
    print(f"[h5_to_mp4] {len(files)} episodes -> {a.out} "
          f"(fps={a.fps} qscale={a.qscale} workers={a.workers} boxes={a.boxes})")

    done = 0
    for attempt in range(4):   # a worker occasionally dies (native crash in the pyav/libx264 path
                                # under parallel load) and takes the whole pool down with it; since
                                # convert_one skips files whose .mp4 already exists, just resubmit
                                # whatever's left with a fresh pool instead of losing the batch.
        pending = [p for p in files
                  if not os.path.exists(os.path.join(a.out, os.path.splitext(os.path.basename(p))[0] + ".mp4"))]
        if not pending:
            break
        if attempt:
            print(f"[h5_to_mp4] pool broke — retrying {len(pending)} remaining episode(s) "
                  f"(attempt {attempt + 1})", flush=True)
        try:
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                jobs = {}
                for p in pending:
                    out_path = os.path.join(a.out, os.path.splitext(os.path.basename(p))[0] + ".mp4")
                    jobs[ex.submit(convert_one, p, out_path, a.fps, a.qscale, a.boxes)] = p
                for fut in as_completed(jobs):
                    p = jobs[fut]
                    _, status = fut.result()
                    done += 1
                    print(f"[{done}/{len(files)}] {os.path.basename(p)}: {status}", flush=True)
        except BrokenProcessPool:
            continue

    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out) if f.endswith(".mp4"))
    print(f"[h5_to_mp4] done. output size = {total / 1e9:.2f} GB in {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
