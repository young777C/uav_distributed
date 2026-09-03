#!/usr/bin/env bash
# E1 on mvp_full_v5 (512px, vehicles ~27px, in-distribution make=55% -> identity IS encoded).
# LIMITED DISK: cache ONLY layer 24 (--layers 24) + fp16 -> ~43GB/cache (vs 426GB for 5-layer fp32).
# mis_follow depends only on the tid head (last layer), so 1-layer cache gives the SAME E1 result.
# Skip single-layer + Stage-1 EAR: mis_follow is EAR-independent (stage2 trains EAR from scratch).
# Stabilized training (tid dropout + indep wd + cosine LR + best-on-mis_follow) already in stage2.py.
set -u
CW=train/config_v5.yaml
CN=train/config_v5_nolang.yaml
GPUS=(0 2 3 4 5 6 7); NG=${#GPUS[@]}
pc(){ local cfg="$1" dir="$2"; rm -rf "$dir"; mkdir -p "$dir"
  local s=0; for g in "${GPUS[@]}"; do
    python -u -m train.backbone_kv --config "$cfg" --layers 24 --shard "$s" --num-shards "$NG" --device "cuda:$g" & s=$((s+1))
  done; wait; }

echo "[v5-e1] === precompute layer24 fp16 · with-language  $(date +%H:%M:%S) ==="
pc "$CW" runs/ctx_cache_ml_v5
echo "[v5-e1] === precompute layer24 fp16 · NO-language  $(date +%H:%M:%S) ==="
pc "$CN" runs/ctx_cache_ml_v5_nolang
echo "[v5-e1] === disk after caches ==="; du -sh runs/ctx_cache_ml_v5 runs/ctx_cache_ml_v5_nolang; df -h /nvidia | tail -1

echo "[v5-e1] === Stage-2 BOTH arms parallel (no EAR warmstart)  $(date +%H:%M:%S) ==="
OMP_NUM_THREADS=8 python -u -m train.stage2 --config "$CW" --device cuda:2 > runs/train_v5.log        2>&1 &
OMP_NUM_THREADS=8 python -u -m train.stage2 --config "$CN" --device cuda:3 > runs/train_v5_nolang.log 2>&1 &
wait
echo "[v5-e1] ALL DONE  $(date +%H:%M:%S)"
