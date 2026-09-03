#!/usr/bin/env bash
# Full-data training on mvp_full (61 good episodes, ep011 excluded). Uses GPUs
# 0,2,3,4,5,6,7 — SKIPS GPU1 which runs the CARLA data-gen (do-not-disturb).
# cap=500 (full 1500-frame coverage), neutral language (A), fixed curriculum (C),
# supervise_intercept=on (predict-intercept training on the long losses).
set -u
CFG=train/config_full.yaml
GPUS=(0 2 3 4 5 6 7); NG=${#GPUS[@]}

echo "[full] === precompute single-layer (Stage-1) $(date +%H:%M:%S) ==="
rm -rf runs/ctx_cache_full; mkdir -p runs/ctx_cache_full
s=0; for g in "${GPUS[@]}"; do python -u -m train.backbone_kv --config $CFG --shard "$s" --num-shards "$NG" --device "cuda:$g" & s=$((s+1)); done; wait

echo "[full] === precompute multi-layer (Stage-2) $(date +%H:%M:%S) ==="
rm -rf runs/ctx_cache_ml_full; mkdir -p runs/ctx_cache_ml_full
s=0; for g in "${GPUS[@]}"; do python -u -m train.backbone_kv --config $CFG --layers 4 8 12 16 24 --shard "$s" --num-shards "$NG" --device "cuda:$g" & s=$((s+1)); done; wait

echo "[full] === Stage-1 EAR warmup $(date +%H:%M:%S) ==="
python -u -m train.stage1_ear --config $CFG

echo "[full] === Stage-2 (supervise_intercept on) $(date +%H:%M:%S) ==="
python -u -m train.stage2 --config $CFG --device cuda:0

echo "[full] ALL DONE $(date +%H:%M:%S)"
