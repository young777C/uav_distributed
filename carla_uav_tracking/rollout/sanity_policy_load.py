"""Milestone-2 loader check: reconstruct the 4-module policy and reproduce the
checkpoint's recorded offline metrics via the SAME _evaluate the trainer used.

If the loader (dims block + cfg["model"] hidden dims + state_dict keys) is wired
correctly, running train.stage2._evaluate on the val set reproduces the ckpt's
`mis_follow` almost exactly (tid is deterministic in eval; act_mse has small
flow-sampling noise). A mis-constructed module collapses mis_follow to chance
(~0.7), so this is a strict test — and it needs NO VLM (reads the ctx cache).

Run in the training container:
    bash train/docker/run.sh python carla_uav_tracking/rollout/sanity_policy_load.py
"""

from __future__ import annotations

import os
import sys

import yaml

_CARLA_PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO = os.path.dirname(_CARLA_PKG)
for _p in (_REPO, _CARLA_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from train.dataset_stage2 import RealStage2Dataset, collate_stage2  # noqa: E402
from train.stage2 import _evaluate  # noqa: E402
from train.stage1_ear import safe_device  # noqa: E402
from rollout.policy import load_policy_modules  # noqa: E402

CONFIG = os.path.join(_REPO, "train", "config_v5.yaml")
CKPT = os.path.join(_REPO, "runs", "stage2_v5", "stage2_best.pt")
TOL_MISFOLLOW = 0.02      # deterministic → should match tightly


def main() -> int:
    cfg = yaml.safe_load(open(CONFIG))
    device = safe_device(cfg["train"].get("device", "auto"))
    torch.manual_seed(int(cfg.get("seed", 0)))

    ear, iar, dit, tid, dims, meta = load_policy_modules(CKPT, cfg, device)
    ref = meta.get("metrics", {})
    print(f"[m2] loaded stage2_v5 epoch={meta.get('epoch')} dims={dims}")
    print(f"[m2] ckpt-recorded metrics: {ref}")

    val = RealStage2Dataset(cfg, "val")
    print(f"[m2] val set: {len(val)} samples")
    loader = DataLoader(val, batch_size=cfg["train"]["batch_size"],
                        num_workers=2, collate_fn=collate_stage2)
    steps = cfg["flow"]["sample_steps"]
    got = _evaluate(ear, iar, dit, tid, loader, device, steps,
                    dims["K"], dims["H"], dims["A"])
    print(f"[m2] re-evaluated via loader:   {got}")

    dmf = abs(got["mis_follow"] - ref.get("mis_follow", 1.0))
    ok = dmf <= TOL_MISFOLLOW
    print(f"\n[m2] Δmis_follow={dmf:.4f} (tol {TOL_MISFOLLOW}) → "
          f"{'PASS ✅ loader reproduces the trained policy' if ok else 'FAIL ✘'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
