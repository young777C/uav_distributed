#!/usr/bin/env bash
# E1 re-run with STABILIZED training (item 1): tid-head dropout 0.1 + tid weight_decay 1e-2
# + cosine LR decay + best.pt selected on mis_follow (not act_mse). Reuses v3 caches + EAR.
# Goal: hold the early mis_follow gap to convergence instead of drifting back to chance.
set -u
CW=train/config_v3.yaml
CN=train/config_v3_nolang.yaml
echo "[v3-stab] === Stage-2 BOTH arms (stabilized) in parallel  $(date +%H:%M:%S) ==="
OMP_NUM_THREADS=8 python -u -m train.stage2 --config "$CW" --device cuda:2 \
  --ckpt-dir runs/stage2_v3_stab       > runs/train_v3_stab.log        2>&1 &
OMP_NUM_THREADS=8 python -u -m train.stage2 --config "$CN" --device cuda:3 \
  --ckpt-dir runs/stage2_v3_stab_nolang > runs/train_v3_stab_nolang.log 2>&1 &
wait
echo "[v3-stab] ALL DONE  $(date +%H:%M:%S)"
