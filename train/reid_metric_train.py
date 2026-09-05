"""Offline re-ID metric-learning fine-tune — make DINOv2 crop features robust to OFF-CENTER
framing (the deployable-SR lever per memory acot-uav-reid-reframe: control frames the target
off-center 92% of the time → off-center reid mis 0.39 → cascade; frame-gain can't fix it
because it amplifies identity errors; the only deployable break-point is reid that is RIGHT
even off-center).

Train a small projection head on top of FROZEN DINOv2 crop features with supervised contrastive
loss on INSTANCE identity (free cross-frame GT: the target is one instance across a whole episode,
each distractor another). Same-instance crops across DIFFERENT views/offsets are pulled together →
the off-center crop's feature moves toward the (more-central) gallery → better match → lower
off-center mis. Eval = the same K-view gallery reid protocol as reid_resolution_probe, with the
projected features, reporting mis_follow CENTRAL vs OFF vs raw DINOv2.

Stages (run --stage all): (1) cache crop→DINOv2 features to npz [expensive, once]; (2) train the
projection [fast, iterate]; (3) eval gallery reid central/off, projected vs raw.

Needs the mvp_full_v5 dataset (raw RGB + GT positions). If it was deleted, regenerate per
acot_note/mvp-regen-spec.md first.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import yaml
import h5py
from PIL import Image

from train.dataset import _select_episodes
from train.reid_resolution_probe import build_cands, _load_dino, _dino_feats, bank_update, bank_score


def _is_central(c, img, frac=0.3):
    W, H = img["width"], img["height"]
    return (abs(c.u / W - 0.5) <= frac / 2) and (abs(c.v / H - 0.5) <= frac / 2)


# --------------------------- stage 1: cache ---------------------------
def cache_features(cfg, out_path, device, episodes, stride):
    b = cfg["backbone"]; img = cfg["image"]
    dino = _load_dino("vit_small_patch14_dinov2.lvd142m", device)
    feats = []; gid = []; ep_a = []; fr_a = []; tgt_a = []; cen_a = []; spl_a = []
    for split_id, split in enumerate(["train", "val"]):
        eps = _select_episodes(cfg, split)[:episodes]
        for ei, ep in enumerate(eps):
            with h5py.File(ep, "r") as f:
                n = f["rgb"].shape[0]
                tgt = np.stack([f["target/tx"][:], f["target/ty"][:], f["target/tz"][:]], -1)
                cam = np.stack([f[f"state/cam_{k}"][:] for k in ["x", "y", "z", "pitch", "yaw"]], -1)
                distr = f["distractors/positions"][:] if "distractors/positions" in f else None
                for i in range(0, n - 1, stride):
                    cands = build_cands(tgt[i], distr[i] if distr is not None else None, cam[i], img)
                    if not cands:
                        continue
                    ff = _dino_feats(dino, np.asarray(f["rgb"][i]), cands, img["width"], img["height"], 224, 1.3, device)
                    for k, c in enumerate(cands):
                        feats.append(ff[k].astype(np.float32))
                        gid.append(split_id * 10_000_000 + ei * 100 + (c.idx + 1))   # unique per (split,ep,vehicle)
                        ep_a.append(ei); fr_a.append(i); tgt_a.append(int(c.is_target))
                        cen_a.append(int(_is_central(c, img))); spl_a.append(split_id)
            print(f"[cache] {split} ep{ei} {Path(ep).name}: total crops={len(feats)}", flush=True)
    np.savez_compressed(out_path, feats=np.stack(feats), gid=np.array(gid, np.int64),
                        ep=np.array(ep_a, np.int32), frame=np.array(fr_a, np.int32),
                        is_target=np.array(tgt_a, np.int8), central=np.array(cen_a, np.int8),
                        split=np.array(spl_a, np.int8))
    print(f"[cache] wrote {out_path}: {len(feats)} crops, dim={feats[0].shape[0]}", flush=True)


# --------------------------- stage 2: train ---------------------------
def supcon_loss(z, labels, tau):
    import torch
    sim = (z @ z.T) / tau
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    exp = torch.exp(sim).masked_fill(self_mask, 0)
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    pc = pos.sum(1)
    loss = -(log_prob * pos).sum(1) / pc.clamp(min=1)
    return loss[pc > 0].mean()


def train_proj(cache, device, epochs, dproj, tau, lr, p_inst, k_view, steps_per_epoch):
    import torch
    from torch import nn
    d = np.load(cache)
    tr = d["split"] == 0
    X = torch.tensor(d["feats"][tr], dtype=torch.float32)
    G = d["gid"][tr]
    # group row-indices by instance (only instances with >=2 crops are usable as positives)
    by = {}
    for idx, g in enumerate(G):
        by.setdefault(int(g), []).append(idx)
    usable = [g for g, ix in by.items() if len(ix) >= 2]
    print(f"[train] train crops={len(G)} instances={len(by)} usable(>=2 views)={len(usable)}", flush=True)

    din = X.shape[1]

    class Refine(nn.Module):                                   # residual full-dim: only REFINE raw DINOv2
        def __init__(self, d):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            for p in self.mlp[-1].parameters():
                nn.init.zeros_(p)                              # start = identity (can't hurt at init)
        def forward(self, x):
            return x + self.mlp(x)

    proj = Refine(din).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=1e-3)
    rng = np.random.default_rng(0)
    Xd = X.to(device)
    for ep in range(epochs):
        tot = 0.0
        for _ in range(steps_per_epoch):
            gs = rng.choice(usable, size=min(p_inst, len(usable)), replace=False)
            rows, labs = [], []
            for g in gs:
                pick = rng.choice(by[int(g)], size=min(k_view, len(by[int(g)])), replace=len(by[int(g)]) < k_view)
                rows += list(pick); labs += [int(g)] * len(pick)
            xb = Xd[torch.tensor(rows, device=device)]
            z = proj(xb); z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
            loss = supcon_loss(z, torch.tensor(labs, device=device), tau)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        print(f"[train] epoch {ep+1}/{epochs}  supcon={tot/steps_per_epoch:.4f}", flush=True)
    return proj


# --------------------------- stage 3: eval ---------------------------
def eval_gallery(cache, proj, device, bank_k, topm, div_tau=0.9):
    """Gallery reid on VAL, projected vs raw. Returns dict of mis (overall/central/off)."""
    import torch
    d = np.load(cache)
    va = d["split"] == 1
    F = d["feats"][va]; EP = d["ep"][va]; FR = d["frame"][va]
    TG = d["is_target"][va]; CEN = d["central"][va]

    def transform(feats):
        if proj is None:
            return feats.astype(np.float64)
        with torch.no_grad():
            z = proj(torch.tensor(feats, dtype=torch.float32, device=device))
            return z.cpu().numpy().astype(np.float64)

    # group by episode, then frame (ordered) → candidate lists
    res = {"all": [0, 0], "central": [0, 0], "off": [0, 0]}
    for ep in np.unique(EP):
        em = EP == ep
        frames = np.unique(FR[em])
        bank = []
        for fr in frames:
            m = em & (FR == fr)
            feats = transform(F[m])
            fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            tg = TG[m]; cen = CEN[m]
            tslot = int(np.argmax(tg)) if tg.any() else -1
            if tslot >= 0 and len(fn) >= 2 and bank:
                sel = int(bank_score(bank, fn, topm).argmax())
                wrong = int(sel != tslot)
                res["all"][0] += wrong; res["all"][1] += 1
                bucket = "central" if cen[tslot] else "off"
                res[bucket][0] += wrong; res[bucket][1] += 1
            if tslot >= 0:
                bank_update(bank, fn[tslot], bank_k, div_tau)
    return {k: (v[0] / v[1] if v[1] else float("nan"), v[1]) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--stage", default="all", choices=["cache", "train", "eval", "all"])
    ap.add_argument("--cache", default="runs/reid_metric_cache.npz")
    ap.add_argument("--episodes", type=int, default=999, help="cap episodes/split (cache stage)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dproj", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p-inst", type=int, default=32, help="instances per batch")
    ap.add_argument("--k-view", type=int, default=4, help="crops per instance per batch")
    ap.add_argument("--steps", type=int, default=100, help="steps per epoch")
    ap.add_argument("--bank", type=int, default=2)
    ap.add_argument("--topm", type=int, default=2)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    if a.stage in ("cache", "all"):
        if _select_episodes(cfg, "val"):
            cache_features(cfg, a.cache, a.device, a.episodes, a.stride)
        else:
            print("[cache] NO EPISODES FOUND — dataset missing? regenerate per acot_note/mvp-regen-spec.md")
            return
    if not Path(a.cache).exists():
        print(f"[eval] cache {a.cache} missing — run --stage cache first"); return

    proj = None
    if a.stage in ("train", "all"):
        proj = train_proj(a.cache, a.device, a.epochs, a.dproj, a.tau, a.lr, a.p_inst, a.k_view, a.steps)
        import torch
        torch.save(proj.state_dict(), a.cache.replace(".npz", "_proj.pt"))

    # eval: RAW dinov2 vs PROJECTED
    raw = eval_gallery(a.cache, None, a.device, a.bank, a.topm)
    print("\n=== reid gallery mis_follow (K%d/top%d) — RAW DINOv2 ===" % (a.bank, a.topm))
    for k, (r, n) in raw.items():
        print(f"  {k:8}: {r:.3f}  (n={n})")
    if proj is not None:
        prj = eval_gallery(a.cache, proj, a.device, a.bank, a.topm)
        print("=== PROJECTED (metric-learned) ===")
        for k, (r, n) in prj.items():
            print(f"  {k:8}: {r:.3f}  (n={n})   Δ={prj[k][0]-raw[k][0]:+.3f}")
        print("★ 判据: PROJECTED 的 off mis < RAW 的 off mis → 度量微调提升了 off-center 认车鲁棒")


if __name__ == "__main__":
    main()
