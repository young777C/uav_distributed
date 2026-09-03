"""DAgger fine-tune of the attrbind tid head on CLOSED-LOOP frames.

The offline-trained attrbind (0.44 offline) collapses to ~0.74 online — its
identity→feature matching was fit on offline (expert, centered, uncrowded) frames.
DAgger fix: `policy.py` (DAGGER_DIR) logged the frames the policy ACTUALLY visits
(off-center/far/N6-crowded) with the GT identity (true_idx, free oracle). Here we
fine-tune attrbind on those frames (+ optional offline replay to avoid forgetting).

Eval: DAgger-val mis_follow (closed-loop proxy) + offline-val mis_follow (forgetting).
If DAgger-val drops a lot while offline-val holds → the online-grounding gap is a
DISTRIBUTION-coverage problem fixable by DAgger → then closed-loop re-eval for SR.
"""
from __future__ import annotations
import argparse, glob, os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader

from train.grounding_train import AttrBindHead, build_attr_vocab, _attr_ids


class DaggerDS(Dataset):
    def __init__(self, files):
        self.s = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            feats = z["feats"].astype(np.float32); n = z["n"]; ti = z["true_idx"]
            col = z["color"]; mk = z["make"]; off = 0
            for i in range(len(n)):
                N = int(n[i])
                self.s.append((feats[off:off + N], int(ti[i]), str(col[i]), str(mk[i]))); off += N

    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i]


class ListDS(Dataset):
    def __init__(self, samples): self.s = samples
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i]


def load_offline(cfg, n):
    """Extract up to n offline (target-present) samples into memory for DAgger replay:
    (cand_feats(N,C), true_idx, color, make) — same tuple format as DAgger frames."""
    from train.dataset_stage2 import RealStage2Dataset
    cfg2 = dict(cfg); cfg2.setdefault("data", {})["use_last_layer_only"] = True
    ds = RealStage2Dataset(cfg2, "train")
    out = []
    for i in range(len(ds)):
        if len(out) >= n:
            break
        s = ds[i]
        if float(s["tid_valid"]) < 0.5:
            continue
        out.append((s["cand_feats"].numpy().astype(np.float32), int(s["target_idx"]),
                    str(s.get("target_color", "?")), str(s.get("target_make", "?"))))
    return out


def collate(batch):
    Nmax = max(s[0].shape[0] for s in batch); C = batch[0][0].shape[1]; B = len(batch)
    cf = np.zeros((B, Nmax, C), np.float32); cm = np.zeros((B, Nmax), bool); ti = np.zeros(B, np.int64)
    cols, mks = [], []
    for i, s in enumerate(batch):
        N = s[0].shape[0]; cf[i, :N] = s[0]; cm[i, :N] = True; ti[i] = s[1]
        cols.append(s[2]); mks.append(s[3])
    return {"cand_feats": torch.from_numpy(cf), "cand_mask": torch.from_numpy(cm),
            "target_idx": torch.from_numpy(ti), "target_colors": cols, "target_makes": mks}


def _eval(head, dl, cvoc, mvoc, dev):
    head.eval(); w = t = 0
    with torch.no_grad():
        for b in dl:
            ci, mi = _attr_ids(b, cvoc, mvoc, dev)
            lg = head(b["cand_feats"].to(dev), ci, mi, b["cand_mask"].to(dev))
            pred = lg.argmax(1).cpu().numpy(); tgt = b["target_idx"].numpy()
            w += int((pred != tgt).sum()); t += len(tgt)
    return w / max(t, 1)


def _eval_offline(head, cfg, cvoc, mvoc, dev):
    from train.dataset_stage2 import RealStage2Dataset, collate_stage2
    cfg2 = dict(cfg); cfg2.setdefault("data", {})["use_last_layer_only"] = True
    ds = RealStage2Dataset(cfg2, "val")
    dl = DataLoader(ds, batch_size=128, collate_fn=collate_stage2, num_workers=4)
    head.eval(); w = t = 0
    with torch.no_grad():
        for b in dl:
            ci, mi = _attr_ids(b, cvoc, mvoc, dev)
            lg = head(b["cand_feats"].to(dev), ci, mi, b["cand_mask"].to(dev))
            pred = lg.argmax(1).cpu().numpy(); tgt = b["target_idx"].numpy()
            tv = (b["tid_valid"] > 0.5).numpy()
            w += int(((pred != tgt) & tv).sum()); t += int(tv.sum())
    return w / max(t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/config_v5.yaml")
    ap.add_argument("--dagger-dir", default="runs/dagger_v5")
    ap.add_argument("--warm", default="runs/tid_attrbind.pt", help="offline-trained attrbind to warmstart")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--offline-n", type=int, default=6000, help="offline replay frames mixed into train (0=off, DAgger-only)")
    ap.add_argument("--out", default="runs/tid_attrbind_dagger.pt")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(a.config))

    files = sorted(glob.glob(os.path.join(a.dagger_dir, "dagger_ep*.npz")))
    assert files, f"no DAgger npz in {a.dagger_dir}"
    ds = DaggerDS(files)
    g = np.random.default_rng(0).permutation(len(ds)); ntr = int(len(ds) * 0.8)
    dag_tr = [ds.s[i] for i in g[:ntr]]; dag_va = [ds.s[i] for i in g[ntr:]]
    # DAgger aggregation: TRAIN on DAgger-train ∪ offline replay (prevents overfitting the
    # collected scenes + forgetting general grounding); VAL on held-out DAgger (closed-loop proxy).
    offline = load_offline(cfg, a.offline_n) if a.offline_n > 0 else []
    train_samples = dag_tr + offline
    dtr = DataLoader(ListDS(train_samples), batch_size=128, shuffle=True, collate_fn=collate, num_workers=4, drop_last=True)
    dva = DataLoader(ListDS(dag_va), batch_size=128, collate_fn=collate, num_workers=2)
    print(f"[dagger] replay: DAgger-train={len(dag_tr)} + offline={len(offline)} = {len(train_samples)} train samples")
    cand_dim = ds[0][0].shape[1]
    cvoc, mvoc = build_attr_vocab(cfg)
    head = AttrBindHead(cand_dim, len(cvoc), len(mvoc), d=cfg["model"].get("tid_d", 256)).to(dev)
    if a.warm and os.path.exists(a.warm):
        head.load_state_dict(torch.load(a.warm, map_location="cpu")["tid"])
        print(f"[dagger] warmstart attrbind from {a.warm}")
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    print(f"[dagger] {len(files)} eps, {len(ds)} frames (train {ntr}/val {len(ds)-ntr}); "
          f"n_color={len(cvoc)} n_make={len(mvoc)}")
    mf0 = _eval(head, dva, cvoc, mvoc, dev); off0 = _eval_offline(head, cfg, cvoc, mvoc, dev)
    print(f"[dagger] init  DAgger-val mis_follow={mf0:.3f}  offline-val mis_follow={off0:.3f}")
    best = 1.0
    for ep in range(a.epochs):
        head.train()
        for b in dtr:
            ci, mi = _attr_ids(b, cvoc, mvoc, dev)
            lg = head(b["cand_feats"].to(dev), ci, mi, b["cand_mask"].to(dev))
            loss = F.cross_entropy(lg, b["target_idx"].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        mf = _eval(head, dva, cvoc, mvoc, dev)
        tag = ""
        if mf < best:
            best = mf; tag = " *best"
            torch.save({"tid": head.state_dict(), "head": "attrbind", "dagger_val": mf}, a.out)
        print(f"[dagger] ep{ep:02d} DAgger-val={mf:.3f}{tag}")
    offN = _eval_offline(head, cfg, cvoc, mvoc, dev)
    print(f"[dagger] BEST DAgger-val={best:.3f} (was {mf0:.3f}) | offline-val now {offN:.3f} (was {off0:.3f})")
    print("  判读: DAgger-val 大降 + offline-val 基本保持 → DAgger 补上了闭环分布 → 去闭环复测 SR")


if __name__ == "__main__":
    main()
