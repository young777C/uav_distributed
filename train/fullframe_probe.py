"""GATE probe for v5 (512px): does the full-frame VLM grid, pooled at candidate
boxes (exactly the stage2 cand_feats), now encode per-candidate identity that the
336px pipeline lost (make 12% / color 23% at ~13px boxes)?

Encodes each 512px frame ONCE, pools at every candidate box (mimics precompute +
dataset_stage2 pool_box), then runs the color/make linear probe. No 400GB cache.
"""
from __future__ import annotations
import argparse, json
import numpy as np, h5py, torch, torch.nn as nn, yaml
from PIL import Image

from train.dataset_stage2 import RealStage2Dataset
from acot_probe.backbones import pool_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--ntr", type=int, default=5000)
    ap.add_argument("--nte", type=int, default=3000)
    ap.add_argument("--recrop", action="store_true", help="crop+encode each candidate (B1) instead of full-frame pool_box")
    ap.add_argument("--random", action="store_true", help="in-distribution random 80/20 split (vs by-episode)")
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

    from train.dataset import _select_episodes
    ds = RealStage2Dataset(cfg, "val")   # only for _h5/_candidates helpers (no cache needed)
    fps = cfg.get("fps", 10)
    horizon = int(cfg["waypoint"].get("max_horizon_s", 6) * fps)
    cap = int(cfg["data"].get("max_frames_per_episode", 500))
    base_stride = int(cfg["data"].get("frame_stride", 3))

    def collect(split, ntarget, step):
        eps = (_select_episodes(cfg, "train") + _select_episodes(cfg, "val")) if split == "all" \
            else _select_episodes(cfg, split)
        stride = base_stride * step
        X, col, bp, box = [], [], [], []
        grid_shape = [None]
        for ep in eps:
            if len(X) >= ntarget:
                break
            try:
                h = ds._h5(ep)
            except Exception:
                continue                                 # bad/partial episode (e.g. ep066)
            n = len(h["tgt"])
            tcol, tbp, distr = ga(ep)
            with h5py.File(ep, "r") as f:
                for i in list(range(0, max(0, n - horizon), stride))[:cap]:
                    if len(X) >= ntarget:
                        break
                    cands = ds._candidates(h, i)
                    if len(cands) < 2 or not any(c.is_target for c in cands):
                        continue
                    im = np.asarray(f["rgb"][i])
                    if not a.recrop:
                        grid = bk.encode(Image.fromarray(im), "the scene", [LAYER])[LAYER][0]
                        grid_shape[0] = grid.shape
                    for c in cands:
                        if a.recrop:                                    # B1: crop candidate, encode alone
                            s = int(np.clip(max(c.w, c.h) * 3.0, 64, W))
                            x0 = int(np.clip(c.u - s / 2, 0, W)); x1 = int(np.clip(c.u + s / 2, 0, W))
                            y0 = int(np.clip(c.v - s / 2, 0, H)); y1 = int(np.clip(c.v + s / 2, 0, H))
                            if x1 - x0 < 4 or y1 - y0 < 4:
                                continue
                            crop = Image.fromarray(im[y0:y1, x0:x1]).resize((224, 224))
                            g = bk.encode(crop, "the vehicle", [LAYER])[LAYER][0]
                            grid_shape[0] = g.shape
                            X.append(g.reshape(-1, g.shape[-1]).mean(0).astype(np.float32))
                        else:
                            X.append(pool_box(grid, c.frac_box(W, H)).astype(np.float32))
                        box.append(float(c.w))
                        if c.is_target:
                            col.append(tcol); bp.append(tbp)
                        else:
                            dd = distr[c.idx] if c.idx < len(distr) else {"color": "?", "bp": "?"}
                            col.append(dd["color"]); bp.append(dd["bp"])
        print(f"[gate] {split}: native grid={grid_shape[0]}  candidates={len(X)}", flush=True)
        return np.stack(X), np.array(col), np.array(bp), np.array(box)

    if a.random:                                   # in-distribution: pool all, random 80/20
        print("[gate] IN-DISTRIBUTION (random split): encoding all episodes ...", flush=True)
        X, cc, bb, bx = collect("all", a.ntr + a.nte, 4)
        g = np.random.default_rng(0).permutation(len(X)); ntr = int(len(X) * 0.8)
        tri, tei = g[:ntr], g[ntr:]
        Xtr, ctr, btr = X[tri], cc[tri], bb[tri]
        Xte, cte, bte, boxte = X[tei], cc[tei], bb[tei], bx[tei]
    else:
        print("[gate] BY-EPISODE: encoding train frames (512px) ...", flush=True)
        Xtr, ctr, btr, _ = collect("train", a.ntr, 5)
        print("[gate] encoding val frames ...", flush=True)
        Xte, cte, bte, boxte = collect("val", a.nte, 3)
    bk.free()
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
        yt = np.array([voc[yte_raw[k]] for k in keep]); ok = (pred == yt); bx = boxte[keep]
        print(f"  {name}: classes={len(voc)} chance={100/len(voc):.1f}%  v5(512) acc={100*ok.mean():.1f}%  (n={len(keep)})", flush=True)
        for lo, hi in [(0, 25), (25, 40), (40, 300)]:
            m = (bx >= lo) & (bx < hi)
            if m.sum():
                print(f"      box {lo}-{hi}px: {100*ok[m].mean():.1f}%  (n={m.sum()})", flush=True)

    coarse = lambda arr: np.array([str(c).split()[-1] for c in arr])   # "dark red"->red, "light blue"->blue
    print("=== v5 512px GATE (baseline 336px: color 23% / make 12%; PASS if >=40%) ===", flush=True)
    probe(ctr, cte, "color(fine) ")
    probe(coarse(ctr), coarse(cte), "color(coarse)")
    probe(btr, bte, "make(bp)    ")


if __name__ == "__main__":
    main()
