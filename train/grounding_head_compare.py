"""WHERE does attrbind's offline win live, and why does it vanish closed-loop?

Eval BOTH tid heads on the SAME offline val, per-frame, stratified by target box size
(px) and tier. Hypothesis: attrbind's advantage is concentrated on LARGE/READABLE
targets (identity legible); on SMALL targets neither head can read identity so they
converge. Closed-loop tracking/reacquire frames are dominated by small/far targets →
attrbind's (offline) edge disappears. If mis_follow(attrbind) << mis_follow(xattn)
only in the big-box buckets and ties in the small-box buckets, the hypothesis holds.

Host, no VLM (features from cache). xattn = stage2_v5 tid; attrbind = tid_attrbind.pt.
"""
from __future__ import annotations
import argparse
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from train.dataset_stage2 import RealStage2Dataset, collate_stage2
from train.dit import TargetIDHead
from train.grounding_train import AttrBindHead, build_attr_vocab, _attr_ids


def _sz(px):
    if px < 20: return "<20px"
    if px < 30: return "20-30px"
    if px < 45: return "30-45px"
    return ">=45px"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--xattn", default="runs/stage2_v5/stage2_best.pt")
    ap.add_argument("--attrbind", default="runs/tid_attrbind.pt")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = a.device
    cfg = yaml.safe_load(open(a.config)); cfg.setdefault("data", {})["use_last_layer_only"] = True
    ds = RealStage2Dataset(cfg, "val")
    dl = DataLoader(ds, batch_size=128, collate_fn=collate_stage2, num_workers=4)
    cand_dim = ds[0]["cand_feats"].shape[-1]; cond_dim = ds.cond_dim

    ckx = torch.load(a.xattn, map_location="cpu")
    xh = TargetIDHead(ckx["dims"]["cand_dim"], cond_dim, d=cfg["model"].get("tid_d", 256)).to(dev)
    xh.load_state_dict(ckx["tid"]); xh.eval()
    cvoc, mvoc = build_attr_vocab(cfg)
    ah = AttrBindHead(cand_dim, len(cvoc), len(mvoc), d=cfg["model"].get("tid_d", 256)).to(dev)
    ah.load_state_dict(torch.load(a.attrbind, map_location="cpu")["tid"]); ah.eval()

    # accumulators: [wrong,total] per (head, size), per (head, tier), overall
    acc = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for b in dl:
            cf = b["cand_feats"].to(dev); cm = b["cand_mask"].to(dev)
            lx = xh(cf, b["vlm_ctx"].to(dev), b["ctx_mask"].to(dev), cm)
            ci, mi = _attr_ids(b, cvoc, mvoc, dev)
            la = ah(cf, ci, mi, cm)
            px = b["target_px"].numpy()
            cen = b["target_central"].numpy() if "target_central" in b else None
            tgt = b["target_idx"].numpy(); tv = (b["tid_valid"] > 0.5).numpy(); tiers = b["tiers"]
            for name, logits in [("xattn", lx), ("attrbind", la)]:
                pred = logits.argmax(1).cpu().numpy()
                for i in range(len(pred)):
                    if not tv[i]:
                        continue
                    w = int(pred[i] != tgt[i])
                    acc[(name, "ALL")][0] += w; acc[(name, "ALL")][1] += 1
                    sb = _sz(float(px[i])) if px[i] > 0 else "n/a"
                    acc[(name, "sz:" + sb)][0] += w; acc[(name, "sz:" + sb)][1] += 1
                    acc[(name, "tier:" + tiers[i])][0] += w; acc[(name, "tier:" + tiers[i])][1] += 1
                    if cen is not None:
                        ck = "central" if cen[i] > 0.5 else "off"
                        acc[(name, "cen:" + ck)][0] += w; acc[(name, "cen:" + ck)][1] += 1

    def r(name, k):
        c = acc[(name, k)]; return (c[0] / c[1], c[1]) if c[1] else (float("nan"), 0)
    print(f"\n[head-compare] xattn={a.xattn}  attrbind={a.attrbind}")
    print(f"{'stratum':<14}{'n':>7}{'xattn':>9}{'attrbind':>10}{'Δ(attr-x)':>11}")
    order = ["ALL", "cen:central", "cen:off", "sz:<20px", "sz:20-30px", "sz:30-45px", "sz:>=45px",
             "tier:a", "tier:c"]
    for k in order:
        mx, n = r("xattn", k); ma, _ = r("attrbind", k)
        if n:
            print(f"{k:<14}{n:>7}{mx:>9.3f}{ma:>10.3f}{ma-mx:>+11.3f}")
    print("\n判读: attrbind 的负 Δ(更好)若集中在大框(>=45/30-45)、小框(<20)趋近 0 → "
          "优势只在'可读身份'的目标上;闭环跟踪/重捕获帧多为小/远目标 → 优势消失(= offline↔online gap 的来源)。")


if __name__ == "__main__":
    main()
