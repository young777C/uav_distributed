"""B1 validation: does re-cropping a candidate region (from the 336px frame),
upscaling, and re-encoding through the VLM recover per-candidate identity that the
full-frame grid pooling loses (color 23% / make 12% at ~13px boxes)?

Encodes each candidate crop separately, then re-runs the color/make linear probe.
If re-crop >> pooled-grid baseline -> B1 works (build the full pipeline).
"""
from __future__ import annotations
import argparse, json
import numpy as np, h5py, torch, torch.nn as nn, yaml
from PIL import Image

from train.dataset_stage2 import RealStage2Dataset


def crop_res(box_px, margin=3.0, lo=48, hi=336):
    return int(np.clip(box_px * margin, lo, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v4.yaml")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--ntr", type=int, default=4000)
    ap.add_argument("--nte", type=int, default=2500)
    ap.add_argument("--res", type=int, default=224)   # VLM input size for the crop
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    W, H = cfg["image"]["width"], cfg["image"]["height"]
    from acot_probe.backbones import build_backbone
    b = cfg["backbone"]
    bk = build_backbone(b["kind"], b["model_id"], a.device,
                        min_pixels=b.get("min_pixels"), max_pixels=b.get("max_pixels")).load()
    LAYER = 24
    ac = {}
    def ga(ep):
        if ep not in ac:
            with h5py.File(ep, "r") as f:
                ac[ep] = (f.attrs["target_color"], f.attrs["target_bp"], json.loads(f.attrs["distractors"]))
        return ac[ep]
    rgb_cache = {}
    def rgb(ep, i):
        key = (ep, i)
        if key not in rgb_cache:
            if len(rgb_cache) > 8:
                rgb_cache.clear()
            with h5py.File(ep, "r") as f:
                rgb_cache[key] = np.asarray(f["rgb"][i])
        return rgb_cache[key]

    def collect(split, ntarget, step):
        ds = RealStage2Dataset(cfg, split)
        X, col, bp, box = [], [], [], []
        for j in range(0, len(ds.index), step):
            if len(X) >= ntarget:
                break
            ep, cp, i, _ = ds.index[j]
            h = ds._h5(ep); cands = ds._candidates(h, i)
            if len(cands) < 2 or not any(c.is_target for c in cands):
                continue
            tcol, tbp, distr = ga(ep)
            im = rgb(ep, i)
            for c in cands:
                s = crop_res(max(c.w, c.h))
                x0 = int(np.clip(c.u - s / 2, 0, W)); x1 = int(np.clip(c.u + s / 2, 0, W))
                y0 = int(np.clip(c.v - s / 2, 0, H)); y1 = int(np.clip(c.v + s / 2, 0, H))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                crop = Image.fromarray(im[y0:y1, x0:x1]).resize((a.res, a.res))
                g = bk.encode(crop, "the vehicle", [LAYER])[LAYER][0]      # (Gh,Gw,C)
                X.append(g.reshape(-1, g.shape[-1]).mean(0).astype(np.float32))
                box.append(float(c.w))
                if c.is_target:
                    col.append(tcol); bp.append(tbp)
                else:
                    dd = distr[c.idx] if c.idx < len(distr) else {"color": "?", "bp": "?"}
                    col.append(dd["color"]); bp.append(dd["bp"])
        return np.stack(X), np.array(col), np.array(bp), np.array(box)

    print(f"[recrop] encoding train candidates (res={a.res}) ...", flush=True)
    Xtr, ctr, btr, _ = collect("train", a.ntr, 6)
    print(f"[recrop] encoding val candidates ...", flush=True)
    Xte, cte, bte, boxte = collect("val", a.nte, 3)
    bk.free()
    print(f"[recrop] train={len(Xtr)} test={len(Xte)}", flush=True)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

    def probe(ytr_raw, yte_raw, name):
        voc = {v: i for i, v in enumerate(sorted(set(ytr_raw)))}
        ytr = np.array([voc[v] for v in ytr_raw])
        net = nn.Linear(Xtr.shape[1], len(voc)); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
        Xg, yg = torch.tensor(Xtr), torch.tensor(ytr)
        for _ in range(120):
            opt.zero_grad(); nn.functional.cross_entropy(net(Xg), yg).backward(); opt.step()
        keep = np.array([k for k, v in enumerate(yte_raw) if v in voc])
        with torch.no_grad():
            pred = net(torch.tensor(Xte[keep])).argmax(1).numpy()
        yt = np.array([voc[yte_raw[k]] for k in keep]); ok = (pred == yt)
        bx = boxte[keep]
        print(f"  {name}: classes={len(voc)} chance={100/len(voc):.1f}%  RE-CROP acc={100*ok.mean():.1f}%  (n={len(keep)})")
        for lo, hi in [(0, 20), (20, 200)]:
            m = (bx >= lo) & (bx < hi)
            if m.sum():
                print(f"      box {lo}-{hi}px: {100*ok[m].mean():.1f}%  (n={m.sum()})")

    print("=== RE-CROP probe (compare to baseline pooled-grid: color 23% / make 12%) ===")
    probe(ctr, cte, "color")
    probe(btr, bte, "make ")


if __name__ == "__main__":
    main()
