"""Grounding diagnostic — WHERE does the tid head fail? (the WHICH axis).

Closed-loop showed the language/identity axis is now the SR wall (lock_wrong 0.79,
mis_follow 0.72). Before picking a fix, localize the failure: run the tid head on
val and STRATIFY mis_follow by
  - similarity tier  a=distinct / b=same_class / c=same_shape|same_color (language REQUIRED)
  - #candidates N    (more distractors = harder; chance = 1 - 1/N)
so the lever is chosen by evidence (like the resolution wall was):
  * error concentrated in tier-c  -> harder-negative / stronger language discrimination
  * high even on tier-a (distinct) -> language not reaching tid / head capacity
  * tracks chance as N grows       -> head not discriminating at all

Loads ONLY the tid head (light). tier and N come from the collated batch (robust to
dataset_stage2's target-absent recursion). Uses the same shuffled candidate order as
training (seeded by j), so mis_follow here matches the honest metric.
"""
from __future__ import annotations
import argparse
from collections import defaultdict

import torch
import yaml

from torch.utils.data import DataLoader
from train.dataset_stage2 import RealStage2Dataset, collate_stage2
from train.dit import TargetIDHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--ckpt", default="runs/stage2_v5/stage2_best.pt")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(a.config))
    ck = torch.load(a.ckpt, map_location="cpu")
    d = ck["dims"]
    tid = TargetIDHead(d["cand_dim"], d["cond_dim"], d=cfg["model"].get("tid_d", 256)).to(dev)
    tid.load_state_dict(ck["tid"]); tid.eval()

    ds = RealStage2Dataset(cfg, a.split)
    dl = DataLoader(ds, batch_size=64, collate_fn=collate_stage2, num_workers=4)

    # accumulators: overall + by tier + by N + by target-box size (all from collated batch)
    tot = [0, 0]                                   # [wrong, total]
    by_tier = defaultdict(lambda: [0, 0])
    by_n = defaultdict(lambda: [0, 0])
    by_sz = defaultdict(lambda: [0, 0])
    by_cen = {"central": [0, 0], "off": [0, 0]}    # extreme-OOD check vs closed-loop

    def size_bucket(px):
        if px < 20: return "<20px"
        if px < 35: return "20-35px"
        if px < 60: return "35-60px"
        return ">=60px"

    with torch.no_grad():
        for b in dl:
            logits = tid(b["cand_feats"].to(dev), b["vlm_ctx"].to(dev),
                         b["ctx_mask"].to(dev), b["cand_mask"].to(dev))
            pred = logits.argmax(1).cpu().numpy()
            tgt = b["target_idx"].numpy()
            tv = (b["tid_valid"] > 0.5).numpy()
            ncand = b["cand_mask"].sum(1).numpy()
            tpx = b["target_px"].numpy() if "target_px" in b else None
            tcen = b["target_central"].numpy() if "target_central" in b else None
            tiers = b["tiers"]
            for i in range(len(pred)):
                if not tv[i]:
                    continue
                wrong = int(pred[i] != tgt[i])
                tot[0] += wrong; tot[1] += 1
                by_tier[tiers[i]][0] += wrong; by_tier[tiers[i]][1] += 1
                nb = int(ncand[i]); by_n[nb][0] += wrong; by_n[nb][1] += 1
                if tpx is not None and tpx[i] > 0:
                    sb = size_bucket(float(tpx[i])); by_sz[sb][0] += wrong; by_sz[sb][1] += 1
                if tcen is not None:
                    ck = "central" if tcen[i] > 0.5 else "off"
                    by_cen[ck][0] += wrong; by_cen[ck][1] += 1

    def rate(c): return c[0] / c[1] if c[1] else float("nan")
    print(f"\n[grounding-probe] ckpt={a.ckpt}  split={a.split}")
    print(f"  OVERALL mis_follow = {rate(tot):.3f}  (n={tot[1]})")
    print(f"  === by 相似度 tier (a=distinct / b=same_class / c=same_shape|color 语言必需) ===")
    for t in ["a", "b", "c"]:
        c = by_tier[t]
        if c[1]: print(f"    tier {t}: mis_follow={rate(c):.3f}  (n={c[1]})")
    print(f"  === by #candidates N (chance=1-1/N) ===")
    for nb in sorted(by_n):
        c = by_n[nb]
        if c[1]: print(f"    N={nb}: mis_follow={rate(c):.3f}  chance={1-1/nb:.3f}  (n={c[1]})")
    if by_sz:
        print(f"  === by 目标框大小 px (小=远/特征粗) ===")
        for sb in ["<20px", "20-35px", "35-60px", ">=60px"]:
            c = by_sz[sb]
            if c[1]: print(f"    {sb}: mis_follow={rate(c):.3f}  (n={c[1]})")
    if by_cen["central"][1] or by_cen["off"][1]:
        ncen = by_cen["central"][1]; noff = by_cen["off"][1]; ntot = max(ncen + noff, 1)
        print(f"  === by 取景 central/off-center (中心30%框, 同闭环口径) ===")
        print(f"    central: mis_follow={rate(by_cen['central']):.3f}  (n={ncen}, {100*ncen/ntot:.0f}%)")
        print(f"    off    : mis_follow={rate(by_cen['off']):.3f}  (n={noff}, {100*noff/ntot:.0f}%)")
        print(f"    ↔ 闭环参照: central 0.50 / off 0.74 ; 闭环 off-center 占 92%")
        print(f"    判读: 离线 off-center 占比 << 92% 且 mis << 0.74 → 闭环 off-center 是更极端 OOD(取景分布问题);"
              f" 若 ≈ → off-center 本身就难(与取景无关)")
    print("  判读: 错误集中在 tier-c → 强化语言判别/难负样本; "
          "tier-a(distinct)也高 → 语言没进 tid / 头容量; 随 N 逼近 chance → 头基本不判别。")


if __name__ == "__main__":
    main()
