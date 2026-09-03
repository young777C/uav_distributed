#!/usr/bin/env bash
# FAIR no-WM baseline retrain on mvp_full_v5 (disk now ample):
#   - Stage-1a EAR warmup (the E1 run SKIPPED this -> EAR was under-trained, FDE 15m@1s)
#   - full 5-layer IAR cache (E1 used 1-layer for disk)
#   - supervise_intercept ON (intercept-BC)
# Then auto-run intercept_eval (EAR FDE + intercept cos) = the fair out-of-frame baseline.
# ckpt -> stage2_v5_fair (preserves E1's stage2_v5). SKIPS GPU1 (data-gen leftover).
set -u
CFG=train/config_v5.yaml
GPUS=(0 2 3 4 5 6 7); NG=${#GPUS[@]}

echo "[fair] === 0a precompute single-layer (EAR, fp32) -> ctx_cache_v5  $(date +%H:%M:%S) ==="
rm -rf runs/ctx_cache_v5; mkdir -p runs/ctx_cache_v5
s=0; for g in "${GPUS[@]}"; do python -u -m train.backbone_kv --config $CFG --shard "$s" --num-shards "$NG" --device "cuda:$g" & s=$((s+1)); done; wait

echo "[fair] === 0b precompute 5-layer (IAR, fp16) -> ctx_cache_ml_v5  $(date +%H:%M:%S) ==="
rm -rf runs/ctx_cache_ml_v5; mkdir -p runs/ctx_cache_ml_v5
s=0; for g in "${GPUS[@]}"; do python -u -m train.backbone_kv --config $CFG --layers 4 8 12 16 24 --shard "$s" --num-shards "$NG" --device "cuda:$g" & s=$((s+1)); done; wait

echo "[fair] === 缓存磁盘 ==="; du -sh runs/ctx_cache_v5 runs/ctx_cache_ml_v5; df -h /nvidia | tail -1

echo "[fair] === 1 Stage-1a EAR warmup -> stage1_ear_v5  $(date +%H:%M:%S) ==="
python -u -m train.stage1_ear --config $CFG

echo "[fair] === 2 Stage-2 joint (warmstart EAR + 5L IAR + supervise_intercept) -> stage2_v5_fair  $(date +%H:%M:%S) ==="
OMP_NUM_THREADS=8 python -u -m train.stage2 --config $CFG --device cuda:2 --ckpt-dir runs/stage2_v5_fair

echo "[fair] === 3 intercept_eval (FAIR no-WM baseline)  $(date +%H:%M:%S) ==="
python -u -m train.intercept_eval --config $CFG --ckpt runs/stage2_v5_fair/stage2_best.pt --device cuda:2

echo "[fair] ALL DONE  $(date +%H:%M:%S)"
