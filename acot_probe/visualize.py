"""Sanity-check the OFFLINE projection: overlay candidate boxes on real frames.

The probe's validity depends on the projected boxes actually landing on the right
vehicles (target vs distractors). RUN THIS before trusting any probe number.

    python -m acot_probe.visualize --config acot_probe/config.yaml \
        --episode data/probe/episode_000000.h5 --frames 8 --out runs/proj_check.png

Green box = target, red = distractor. If boxes are off the cars, the stored
camera convention differs from this module's assumption — record true 2D bboxes
at generation time instead (see README, "Projection caveat").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from .projection import extract_candidates


def draw(config, episode, n_frames, out_path):
    import h5py
    from PIL import Image, ImageDraw

    cfg = yaml.safe_load(Path(config).read_text())
    W, H = cfg["image"]["width"], cfg["image"]["height"]
    with h5py.File(episode, "r") as ep:
        total = ep["rgb"].shape[0]
        picks = np.linspace(0, total - 1, n_frames).astype(int)
        tiles = []
        for i in picks:
            img = Image.fromarray(np.asarray(ep["rgb"][i])).convert("RGB")
            dr = ImageDraw.Draw(img)
            for c in extract_candidates(ep, int(i), cfg):
                x0, y0 = c.u - c.w / 2, c.v - c.h / 2
                x1, y1 = c.u + c.w / 2, c.v + c.h / 2
                color = (0, 220, 0) if c.is_target else (230, 40, 40)
                dr.rectangle([x0, y0, x1, y1], outline=color, width=3)
            tiles.append(np.asarray(img))
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * H, cols * W, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = t
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    print(f"[viz] wrote {out_path} — verify green=target lands on the referred car.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", default="runs/proj_check.png")
    a = ap.parse_args()
    draw(a.config, a.episode, a.frames, a.out)


if __name__ == "__main__":
    main()
