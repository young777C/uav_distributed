"""Dump a montage around the first occlusion event of an episode (before→during→after)
so we can see WHAT the occlusion is (target behind a building vs framing failure)."""
import sys
import h5py
import numpy as np
from PIL import Image, ImageDraw

fp = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "/data/occ_montage.png"
with h5py.File(fp, "r") as f:
    rgb = f["rgb"]
    n = rgb.shape[0]
    occl = f["annotation/occluded"][:] if "annotation/occluded" in f else np.zeros(n)
    offs = f["annotation/off_screen"][:] if "annotation/off_screen" in f else np.zeros(n)
    ostr = f["annotation/occ_structural"][:] if "annotation/occ_structural" in f else np.zeros(n)
    bu = f["annotation/bbox_u"][:]; bv = f["annotation/bbox_v"][:]
    bw = f["annotation/bbox_w"][:]; bh = f["annotation/bbox_h"][:]
    # first frame the target becomes occluded (after being visible)
    ev = None
    for i in range(5, n):
        if occl[i] > 0 and occl[i - 1] == 0:
            ev = i; break
    if ev is None:
        ev = int(np.argmax(occl)) if occl.max() > 0 else n // 2
    idxs = [max(0, ev - 15), max(0, ev - 6), ev, min(n - 1, ev + 6),
            min(n - 1, ev + 15), min(n - 1, ev + 30)]
    tiles = []
    for i in idxs:
        im = Image.fromarray(rgb[i][:]).convert("RGB")
        d = ImageDraw.Draw(im)
        u, v, w, h = float(bu[i]), float(bv[i]), float(bw[i]), float(bh[i])
        kind = ("OFF" if offs[i] > 0 else ("STRUCT" if ostr[i] > 0.3 else "vis"))
        col = (255, 40, 40) if occl[i] > 0 else (40, 220, 40)
        if w > 0 and 0 <= u < 336 and 0 <= v < 336:
            d.rectangle([u - w / 2, v - h / 2, u + w / 2, v + h / 2], outline=col, width=2)
        d.text((4, 4), "f%d %s occ=%.2f" % (i, kind, float(occl[i])), fill=col)
        tiles.append(im)
    W = 336 * len(tiles)
    canvas = Image.new("RGB", (W, 336))
    for j, t in enumerate(tiles):
        canvas.paste(t, (j * 336, 0))
    canvas.save(out)
    print("event frame", ev, "| saved", out, "| idxs", idxs)
    print("off_screen%%=%.1f struct>0.3%%=%.1f meanOcc=%.3f"
          % (100 * offs.mean(), 100 * (ostr > 0.3).mean(), occl.mean()))
