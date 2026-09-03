#!/usr/bin/env bash
# Relaunch ONLY the two Stage-2 arms for E1 (precompute + EAR already done).
# Bad episode ep066 (missing search_mode/off_screen) now auto-skipped in _build_index.
set -u
CW=train/config_v3.yaml
CN=train/config_v3_nolang.yaml
echo "[v3-s2] === Stage-2 BOTH arms in parallel  $(date +%H:%M:%S) ==="
OMP_NUM_THREADS=8 python -u -m train.stage2 --config "$CW" --device cuda:2 > runs/train_v3.log        2>&1 &
OMP_NUM_THREADS=8 python -u -m train.stage2 --config "$CN" --device cuda:3 > runs/train_v3_nolang.log 2>&1 &
wait
echo "[v3-s2] ALL DONE  $(date +%H:%M:%S)"
