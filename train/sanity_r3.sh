#!/usr/bin/env bash
# End-to-end pipeline sanity on mvp_review_r3 (10 eps): verify the new data (long
# losses, new bbox, schema) flows through precompute -> Stage-1 EAR -> Stage-2
# without error. Not chasing metrics; 3 epochs each. supervise_intercept=on to
# exercise the §3.2 mask reversal on real long-loss frames.
set -u
CFG=train/config_r3_sanity.yaml

echo "[sanity] === precompute single-layer (Stage-1 cache) $(date +%H:%M:%S) ==="
rm -rf runs/ctx_cache_r3; mkdir -p runs/ctx_cache_r3
for i in 0 1 2 3; do python -u -m train.backbone_kv --config $CFG --shard "$i" --num-shards 4 --device "cuda:$i" & done; wait

echo "[sanity] === precompute multi-layer (Stage-2 cache) $(date +%H:%M:%S) ==="
rm -rf runs/ctx_cache_ml_r3; mkdir -p runs/ctx_cache_ml_r3
for i in 0 1 2 3; do python -u -m train.backbone_kv --config $CFG --layers 4 8 12 16 24 --shard "$i" --num-shards 4 --device "cuda:$i" & done; wait

echo "[sanity] === Stage-1 EAR warmup (3 ep) $(date +%H:%M:%S) ==="
python -u -m train.stage1_ear --config $CFG

echo "[sanity] === Stage-2 (3 ep, supervise_intercept=on) $(date +%H:%M:%S) ==="
python -u -m train.stage2 --config $CFG --device cuda:0

echo "[sanity] ALL DONE $(date +%H:%M:%S)"
