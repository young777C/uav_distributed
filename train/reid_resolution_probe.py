"""Offline reid-vs-resolution probe — test (a): is the 0.57 closed-loop reid mis_follow
ceiling caused by APPEARANCE-FEATURE resolution (the reframe root cause), or an intrinsic
look-alike wall?

Encode raw episode RGB at a chosen --max-pixels (Qwen dynamic vision grid → finer/coarser
grid_last), pool each candidate box → per-candidate appearance feature, run the SAME temporal
appearance-memory re-ID as policy.py `tid_head=reid` (EMA template; SELECT with the PAST
template THEN update from the GT target → no re-appearance leak), and report reid mis_follow.

Sweep --max-pixels low→high:
  reid mis_follow DROPS with a finer grid  → resolution/discriminability IS the root
                                             (raise VLM max_pixels / render higher)
  reid mis_follow FLAT                      → intrinsic look-alike wall OR need >512 renders
                                             (512 data has ~16x16 native grid → limited headroom)
See memory acot-uav-reid-reframe.
"""
from __future__ import annotations
import argparse
import math

import numpy as np
import yaml
import h5py
from PIL import Image

from acot_probe.projection import project_point, Candidate
from acot_probe.backbones import build_backbone, pool_box
from train.backbone_kv import neutralize_language
from train.dataset import _select_episodes


def build_cands(tgt_xyz, distr_i, cam, img):
    W, H, fov = img["width"], img["height"], img["fov_deg"]
    f = W / (2.0 * math.tan(math.radians(fov) / 2.0))

    def mk(xyz, is_t, idx):
        pr = project_point(tuple(float(x) for x in xyz), tuple(float(x) for x in cam), W, H, fov)
        if pr is None:
            return None
        u, v, depth = pr
        if not (0 <= u < W and 0 <= v < H):
            return None
        s = float(np.clip(f * img.get("vehicle_size_m", 4.0) / max(depth, 0.1),
                          img.get("box_min_px", 10), img.get("box_max_px", 140)))
        return Candidate(u, v, s, s, depth, is_t, idx)

    t = mk(tgt_xyz, True, -1)
    out = [t] if t is not None else []
    if distr_i is not None:
        for d in range(distr_i.shape[0]):
            c = mk(distr_i[d, :3], False, d)
            if c is not None:
                out.append(c)
    return out


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _load_dino(model_name, device):
    import timm
    m = timm.create_model(model_name, pretrained=True, num_classes=0,
                          dynamic_img_size=True).eval().to(device)   # allow 224 input (default 518)
    return m


def _dino_feats(dino, rgb, cands, W, H, size, pad, device):
    """Crop each candidate's PIXEL box from the raw RGB → resize → DINOv2 → (N, D) instance feats.
    Uses the SAME pixels as vlmpool but gives the car ~ (size/14)^2 tokens instead of ~1 grid cell."""
    import torch
    crops = []
    for c in cands:
        w, h = c.w * pad, c.h * pad
        x0 = int(max(0, c.u - w / 2)); y0 = int(max(0, c.v - h / 2))
        x1 = int(min(W, c.u + w / 2)); y1 = int(min(H, c.v + h / 2))
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            crop = rgb
        img = np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR), np.float32) / 255.0
        crops.append(((img - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1))   # (3,H,W)
    x = torch.from_numpy(np.stack(crops)).to(device)
    with torch.no_grad():
        return dino(x).float().cpu().numpy()                                        # (N, D)


def bank_update(bank, f, K, tau):
    """Diversity gallery: keep up to K MUTUALLY-DIVERSE target views. Skip near-duplicate
    views (max sim to bank >= tau); when full, evict the most-redundant member."""
    if not bank:
        bank.append(f); return
    sims = [float(f @ g) for g in bank]
    if max(sims) >= tau:                                   # near-duplicate view → don't store
        return
    if len(bank) < K:
        bank.append(f)
    elif K == 1:                                           # K=1 gallery → keep the latest view
        bank[0] = f
    else:                                                  # full → drop the most-redundant member
        red = [max(float(bank[i] @ bank[j]) for j in range(len(bank)) if j != i) for i in range(len(bank))]
        bank[int(np.argmax(red))] = f


def bank_score(bank, fn, topm):
    """(N,) score per candidate = top-m mean cosine to the K-view gallery (robust to a single
    distractor fluke-matching a single stored view)."""
    S = fn @ np.stack(bank).T                              # (N, |bank|)
    m = min(topm, S.shape[1])
    return np.sort(S, axis=1)[:, -m:].mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-pixels", type=int, default=0, help="0 = Qwen default; else H*W cap (finer/coarser grid)")
    ap.add_argument("--feature", default="vlmpool", choices=["vlmpool", "dinov2"],
                    help="vlmpool=pool the coarse Qwen grid over the box (current) | "
                         "dinov2=crop raw RGB → resize → DINOv2 instance features (test: better feature "
                         "from the SAME pixels, no resolution change)")
    ap.add_argument("--dino-model", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--crop-size", type=int, default=224)
    ap.add_argument("--crop-pad", type=float, default=1.3, help="box padding for the crop (context)")
    ap.add_argument("--bank", type=int, default=1, help="K = multi-view gallery size (1 = single template)")
    ap.add_argument("--topm", type=int, default=1, help="match = mean of top-m cosines to the gallery")
    ap.add_argument("--div-tau", type=float, default=0.9, help="skip a view if its max sim to bank >= tau")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    b = cfg["backbone"]; img = cfg["image"]; layer = int(b["layer"])
    mp = a.max_pixels or b.get("max_pixels")
    bk = dino = None
    if a.feature == "vlmpool":
        bk = build_backbone(b["kind"], b["model_id"], a.device,
                            min_pixels=b.get("min_pixels"), max_pixels=mp).load()
    else:                                                    # dinov2: crop-based instance features (no Qwen)
        dino = _load_dino(a.dino_model, a.device)
    eps = _select_episodes(cfg, a.split)[:a.episodes]

    tot = [0, 0]; hard = [0, 0]; gsz = []; pxs = []; sims = []; self_sims = []
    for ep in eps:
        with h5py.File(ep, "r") as f:
            n = f["rgb"].shape[0]
            lang = f.attrs.get("language", "")
            lang = lang.decode() if isinstance(lang, bytes) else str(lang)
            if b.get("neutralize_language", False):
                lang = neutralize_language(lang)
            tgt = np.stack([f["target/tx"][:], f["target/ty"][:], f["target/tz"][:]], -1)
            cam = np.stack([f[f"state/cam_{k}"][:] for k in ["x", "y", "z", "pitch", "yaw"]], -1)
            distr = f["distractors/positions"][:] if "distractors/positions" in f else None
            similar = f["distractors/similar"][:] if "distractors/similar" in f else None
            bank = []                                                       # per-episode multi-view gallery
            for i in range(0, n - 1, a.stride):
                cands = build_cands(tgt[i], distr[i] if distr is not None else None, cam[i], img)
                if not cands:
                    continue
                rgb_i = np.asarray(f["rgb"][i])
                W, H = img["width"], img["height"]
                if a.feature == "vlmpool":
                    grid = bk.encode(Image.fromarray(rgb_i), lang, [layer])[layer][0]         # (Gh,Gw,C)
                    gsz.append(grid.shape[:2])
                    feats = np.stack([pool_box(grid, c.frac_box(W, H)) for c in cands]).astype(np.float64)
                else:
                    feats = _dino_feats(dino, rgb_i, cands, W, H, a.crop_size, a.crop_pad, a.device).astype(np.float64)
                fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
                tslot = next((k for k, c in enumerate(cands) if c.is_target), -1)
                if tslot >= 0 and len(cands) >= 2 and bank:                      # reid select vs PAST gallery
                    score = bank_score(bank, fn, a.topm)                        # (N,) top-m mean cosine
                    sel = int(score.argmax())
                    wrong = int(sel != tslot)
                    tot[0] += wrong; tot[1] += 1
                    self_sims.append(float(score[tslot]))                       # target-vs-gallery
                    sim_present = similar is not None and any(
                        (not c.is_target) and c.idx < len(similar) and similar[c.idx] for c in cands)
                    if sim_present:
                        hard[0] += wrong; hard[1] += 1
                    for k, c in enumerate(cands):                                # look-alike-vs-gallery
                        if (not c.is_target) and similar is not None and c.idx < len(similar) and similar[c.idx]:
                            sims.append(float(score[k]))
                if tslot >= 0:                                                   # THEN store this view (future only)
                    pxs.append(cands[tslot].w)
                    bank_update(bank, fn[tslot], a.bank, a.div_tau)

    r = lambda c: c[0] / c[1] if c[1] else float("nan")
    if a.feature == "vlmpool":
        gh = int(np.median([g[0] for g in gsz])) if gsz else 0
        gw = int(np.median([g[1] for g in gsz])) if gsz else 0
        desc = f"feature=vlmpool  max_pixels={mp}  grid_last≈{gh}x{gw}"
    else:
        desc = f"feature=dinov2({a.dino_model})  crop={a.crop_size}px pad={a.crop_pad}  tokens/car≈{(a.crop_size//14)**2}"
    print(f"\n[reid-res probe] {desc}  bank=K{a.bank}/top{a.topm}  eps={len(eps)} stride={a.stride}")
    print(f"  target box px: median={np.median(pxs):.0f}" if pxs else "  (no target frames)")
    print(f"  reid mis_follow OVERALL              = {r(tot):.3f}  (n={tot[1]})")
    print(f"  reid mis_follow w/ SIMILAR distractor = {r(hard):.3f}  (n={hard[1]})")
    if sims:
        ss = np.mean(self_sims) if self_sims else float("nan")
        sd = np.mean(sims)
        print(f"  target-vs-GALLERY score (higher=stable)   = {ss:.3f}  (n={len(self_sims)})")
        print(f"  similar-vs-GALLERY score (lower=separable) = {sd:.3f}  (n={len(sims)})")
        print(f"  MARGIN (self - similar, higher=better re-ID) = {ss - sd:+.3f}")


if __name__ == "__main__":
    main()
