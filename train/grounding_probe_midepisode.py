"""S0 probe (1003_dynamic-retarget-design.md) — does the DEPLOYED TargetIDHead's grounding
accuracy degrade away from episode start (cold-start), using the SAME unchanged per-episode
language the model was trained on?

This does not test "retarget to a new distractor" (no distractor-level language exists in the
data, see design doc §3.1) — it answers the narrower, cheaper question: is grounding on
mid/late-episode frames intrinsically harder than on cold-start frames for the one target it was
actually told about? If accuracy is flat or improves away from frame 0 (holding tier/N/size
fixed), the deployed head is not "secretly frame-0-only"; if it craters, that's a distribution
gap (training/eval sampling is dominated by early frames) that would also sink mid-episode
re-grounding before language-content even enters the picture.

Reuses RealStage2Dataset verbatim (train/dataset_stage2.py) but tags each item with its raw
frame index within the episode (not available in the stock collate output), bucketed into
elapsed-time-since-episode-start bins. Loads ONLY the tid head (same as grounding_probe.py) —
no VLM forward, cheap.

  python -m train.grounding_probe_midepisode --split val
  python -m train.grounding_probe_midepisode --split test
"""
from __future__ import annotations
import argparse
from collections import defaultdict

import torch
import yaml
from torch.utils.data import DataLoader

from train.dataset_stage2 import RealStage2Dataset, collate_stage2
from train.dit import TargetIDHead


class FrameTaggedDataset(RealStage2Dataset):
    """Same as RealStage2Dataset, but every returned item also carries the *requested* index's
    frame number. __getitem__ can internally redirect to j+1 (target-absent skip, see
    dataset_stage2.py ~L286); we re-derive frame_i from self.index at the SAME j the base class
    actually resolved to by re-walking the redirect chain (cheap: redirects are rare and local)."""

    def __getitem__(self, j):
        item = RealStage2Dataset.__getitem__(self, j)
        # Re-resolve which index the base class actually consumed by replaying its skip
        # condition cheaply: has_target / cands come from _candidates(h, i). We recompute
        # the same has_target test it used, walking the identical (j+1) % len chain.
        jj = j
        for _ in range(len(self.index)):
            ep, ctx_path, i, tier = self.index[jj]
            h = self._h5(ep)
            cands = self._candidates(h, i)
            has_target = any(c.is_target for c in cands)
            intercept = False
            if self.supervise_intercept and float(h["vis"][i]) < 0.5:
                fut_vis = h["vis"][i:i + self.intercept_hz]
                if fut_vis.size and float(fut_vis.max()) >= 0.5:
                    intercept = True
            if cands and (has_target or intercept):
                break
            jj = (jj + 1) % len(self.index)
        item["_frame_i"] = int(i)
        item["_ep"] = str(ep)
        return item


def collate_tagged(batch):
    frame_i = [b.pop("_frame_i") for b in batch]
    ep = [b.pop("_ep") for b in batch]
    out = collate_stage2(batch)
    out["_frame_i"] = frame_i
    out["_ep"] = ep
    return out


EDGES = [0, 30, 150, 400, 10**9]
LABELS = ["0-30(≤3s cold-start)", "30-150(3-15s)", "150-400(15-40s)", "400+(40s+)"]


def bucket(i):
    for lo, hi, lab in zip(EDGES, EDGES[1:], LABELS):
        if lo <= i < hi:
            return lab
    return LABELS[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--ckpt", default="runs/stage2_v5/stage2_best.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(a.config))
    ck = torch.load(a.ckpt, map_location="cpu")
    d = ck["dims"]
    tid = TargetIDHead(d["cand_dim"], d["cond_dim"], d=cfg["model"].get("tid_d", 256)).to(dev)
    tid.load_state_dict(ck["tid"]); tid.eval()

    ds = FrameTaggedDataset(cfg, a.split)
    dl = DataLoader(ds, batch_size=a.batch_size, collate_fn=collate_tagged, num_workers=a.workers)

    buckets = {lab: {"all": [0, 0], "lookC": [0, 0], "N": defaultdict(lambda: [0, 0])} for lab in LABELS}
    tot = [0, 0]
    with torch.no_grad():
        for b in dl:
            logits = tid(b["cand_feats"].to(dev), b["vlm_ctx"].to(dev),
                         b["ctx_mask"].to(dev), b["cand_mask"].to(dev))
            pred = logits.argmax(1).cpu().numpy()
            tgt = b["target_idx"].numpy()
            tv = (b["tid_valid"] > 0.5).numpy()
            ncand = b["cand_mask"].sum(1).numpy()
            tiers = b["tiers"]
            for i in range(len(pred)):
                if not tv[i]:
                    continue
                wrong = int(pred[i] != tgt[i])
                lab = bucket(b["_frame_i"][i])
                buckets[lab]["all"][0] += wrong; buckets[lab]["all"][1] += 1
                if tiers[i] == "c":
                    buckets[lab]["lookC"][0] += wrong; buckets[lab]["lookC"][1] += 1
                nb = int(ncand[i])
                buckets[lab]["N"][nb][0] += wrong; buckets[lab]["N"][nb][1] += 1
                tot[0] += wrong; tot[1] += 1

    def rate(c):
        return c[0] / c[1] if c[1] else float("nan")

    print(f"\n[grounding-probe-midepisode] ckpt={a.ckpt} split={a.split}")
    print(f"  OVERALL mis_follow = {rate(tot):.3f} (n={tot[1]})")
    print(f"  === mis_follow by elapsed-time-since-episode-start bucket ===")
    print(f"  {'bucket':24s} {'all':>18s} {'tier-c (lookalike, language-required)':>42s}")
    for lab in LABELS:
        c = buckets[lab]["all"]; cc = buckets[lab]["lookC"]
        print(f"  {lab:24s} {rate(c):.3f} (n={c[1]:4d})      {rate(cc):.3f} (n={cc[1]:4d})")
    print(f"  === tier-c mis_follow by bucket × N (chance=1-1/N), diagnoses confound ===")
    for lab in LABELS:
        for nb in sorted(buckets[lab]["N"]):
            c = buckets[lab]["N"][nb]
            if c[1] >= 5:
                print(f"    {lab:24s} N={nb}: mis_follow={rate(c):.3f} chance={1-1/nb:.3f} (n={c[1]})")
    print("  判读: 若各 bucket 的 tier-c mis_follow 持平/单调不恶化 -> 非首帧 grounding 本身不是瓶颈,"
          "中途重指定的核心代价在于语言需要描述一个新目标(训练从未见过的监督信号),而不是现有位置分布外;"
          "若随 bucket 明显恶化(即使控制 N)-> 训练数据的帧分布偏向 episode 早段,本身就是一堵墙。")


if __name__ == "__main__":
    main()
