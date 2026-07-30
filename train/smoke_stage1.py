"""Stage-1 plumbing test — NO VLM, NO CARLA, NO GPU.

Runs the REAL Stage-1 loop (run_training) on a synthetic learnable cond->waypoint
mapping and asserts the flow-matching EAR actually learns it (val wp_mse drops far
below the untrained baseline). Proves ear + flow_matching + dataset + train loop
are wired correctly.

    python -m train.smoke_stage1
"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import SyntheticEARDataset
from .stage1_ear import run_training


def main():
    torch.manual_seed(0)
    k, c = 3, 64
    train_ds = SyntheticEARDataset(n=1024, m=16, c=c, k=k, seed=0)
    val_ds = SyntheticEARDataset(n=256, m=16, c=c, k=k, seed=1)

    # untrained baseline: variance of GT waypoints (mse of predicting ~0/noise)
    wp = np.stack([val_ds[i]["wp"].numpy() for i in range(len(val_ds))])
    baseline = float(np.mean(wp ** 2))

    cfg = {
        "ear": {"d_model": 128, "n_layers": 3, "n_heads": 4},
        "flow": {"sample_steps": 20},
        "train": {"epochs": 80, "lr": 1e-3, "batch_size": 64, "device": "cpu",
                  "ckpt_dir": "runs/stage1_smoke"},
        "eval": {"every": 10, "gate_mse_6s_m": 0.5},   # scale=1 -> err in synthetic units
    }
    out = run_training(train_ds, val_ds, cond_dim=c, k=k, cfg=cfg, scale=1.0, tag="smoke")

    print(f"[smoke] baseline wp_mse={baseline:.4f}  learned wp_mse={out['best_wp_mse']:.4f}")
    assert out["best_wp_mse"] < 0.4 * baseline, "EAR failed to learn synthetic mapping"
    print("[smoke] OK — ear + flow_matching + dataset + train loop are sound.")


if __name__ == "__main__":
    main()
