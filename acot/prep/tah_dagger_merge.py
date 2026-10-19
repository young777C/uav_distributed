"""1006 DINOv3 stage 2: build the full-recipe TAH cache = the offline cache's episodes followed by the DAgger episodes of
a tah_dagger collection (one per complete seed, seed order; empty episodes dropped) — the same composition as
runs/tah_cache_dag.pt (64 offline + DAgger, ONE file, so train/tah_train.py takes the validation split from the first
15 % of the merged list exactly as for the DINOv2 recipe). train/tah_merge.py is not used: it hard-codes d_feat 384.
  python -m acot.prep.tah_dagger_merge --offline runs/tah_1006/tah_cache_dinov3.pt \\
      --experiment acot/configs/experiments/t0_dinov3_dagger.yaml --arm dag_v3off --out runs/tah_1006/tah_cache_dinov3_dag.pt"""
import argparse
import json
import os

import torch

from acot import ROOT
from acot.config.resolve import resolve_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    off = torch.load(os.path.join(ROOT, a.offline), map_location="cpu", weights_only=False)
    R = resolve_experiment(os.path.join(ROOT, a.experiment))
    base = os.path.join(ROOT, R["infra"]["results_root"], R["name"], a.arm, "seeds")
    dag, used, skipped = [], [], {}
    for s in sorted(int(x["seed"]) for x in R["seeds"]):
        d = os.path.join(base, str(s))
        st = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
        if st.get("status") != "complete":
            skipped[s] = st.get("status") or "missing"
            continue
        ep = torch.load(os.path.join(d, "tah_dagger.pt"), map_location="cpu", weights_only=False)
        assert ep["seed"] == s, (ep["seed"], s)
        if ep["d_feat"] is not None and ep["d_feat"] != off["d_feat"]:
            raise SystemExit(f"seed {s}: d_feat {ep['d_feat']} ≠ offline {off['d_feat']}")
        if ep["episodes"][0]:
            dag.append(ep["episodes"][0])
            used.append(s)
        else:
            skipped[s] = "empty"
    eps = list(off["episodes"]) + dag
    torch.save({"episodes": eps, "d_feat": off["d_feat"],
                "sources": {"offline": a.offline, "n_offline": len(off["episodes"]), "dagger": a.experiment,
                            "arm": a.arm, "dagger_seeds": used, "skipped": skipped}}, os.path.join(ROOT, a.out))
    nf = sum(len(e) for e in dag)
    nt = sum(1 for e in dag for fr in e if int(fr["tidx"]) >= 0)
    print(f"[tah_dagger_merge] {len(off['episodes'])} offline + {len(dag)} DAgger episodes ({nf} frames, target among the "
          f"candidates in {nt}) → {a.out}; d_feat {off['d_feat']}; skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
