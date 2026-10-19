"""Merge per-episode TAH DAgger files (policy TAH_DAGGER_DIR → tahdag_*_epNNNN.pt, schema of train/tah_cache.py) into
one cache {"episodes": [...], "d_feat": 384}; empty episodes are dropped.
  python train/tah_merge.py --glob 'runs/tah_v90/dag_v2/*.pt' --out runs/tah_v90/dag_v2.pt"""
import argparse
import glob

import torch

ap = argparse.ArgumentParser(); ap.add_argument("--glob", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
eps, keys = [], []
for p in sorted(glob.glob(a.glob)):
    d = torch.load(p, weights_only=False)
    for k, e in enumerate(d["episodes"]):
        if len(e) > 0:
            eps.append(e); keys.append(d["order"][k] if "order" in d else len(keys))
if all("order" in torch.load(p, weights_only=False) for p in glob.glob(a.glob)):   # sharded tah_cache → original order
    eps = [e for _, e in sorted(zip(keys, eps), key=lambda t: t[0])]
nf = sum(len(e) for e in eps); nt = sum(1 for e in eps for fr in e if int(fr["tidx"]) >= 0)
torch.save({"episodes": eps, "d_feat": 384}, a.out)
print(f"[tah_merge] {len(eps)} episodes, {nf} frames ({nt} with the target among the candidates) → {a.out}")
