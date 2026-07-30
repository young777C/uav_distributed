"""Stage-2 plumbing test — NO VLM, NO CARLA, NO GPU.

Runs the REAL Stage-2 loop (run_training_stage2) on a synthetic dataset with
learnable mappings and asserts that the two things that must be learnable get
learned: the action chunk (act_mse drops) and target selection (mis_follow drops).
Exercises EAR+IAR+DiT+target_id + multi-task loss + curriculum sampler + staleness.

    python -m train.smoke_stage2
"""

from __future__ import annotations

import torch

from .dataset_stage2 import SyntheticStage2Dataset
from .stage2 import run_training_stage2


def main():
    torch.manual_seed(0)
    train_ds = SyntheticStage2Dataset(n=768, L=4, M=12, C=48, H=16, K=3, N=5, seed=0)
    val_ds = SyntheticStage2Dataset(n=192, L=4, M=12, C=48, H=16, K=3, N=5, seed=1)

    cfg = {
        "model": {
            "ear": {"d_model": 128, "n_layers": 3, "n_heads": 4},
            "iar": {"d": 96, "n_im": 6, "n_heads": 4},
            "dit": {"d_model": 128, "n_layers": 8, "n_heads": 4},
            "tid_d": 96,
        },
        "loss": {"ear": 0.3, "vis": 0.1, "maneuver": 0.1, "target_id": 0.2},
        "flow": {"sample_steps": 16},
        "train": {"epochs": 100, "lr": 1e-3, "batch_size": 64, "device": "cpu",
                  "staleness_p": 0.3, "staleness_noise": 0.2},
        "eval": {"every": 10},
    }
    out = run_training_stage2(train_ds, val_ds, cfg=cfg, tag="smoke2")

    m = out["final"]
    chance = 1.0 - 1.0 / 5                     # 5 candidates -> chance mis_follow 0.8
    print(f"[smoke2] final mis_follow={m['mis_follow']:.3f} (chance {chance:.2f}) "
          f"act_mse={m['act_mse']:.3f}")
    assert m["mis_follow"] < 0.4, "target_id head failed to learn (language grounding)"
    assert m["act_mse"] < 0.6, "DiT action head failed to learn the chunk"
    print("[smoke2] OK — EAR+IAR+DiT+target_id + multi-task loss + curriculum are sound.")


if __name__ == "__main__":
    main()
