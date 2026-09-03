#!/usr/bin/env bash
# w/o-language ablation on mvp_full: generic prompt ('Track the target vehicle'),
# same config/EAR-warmstart otherwise. Diagnoses whether the model uses language:
# if no-lang mis_follow ≈ with-lang (~42%), the model ignores language.
set -u
CFG=train/config_full_nolang.yaml
GPUS=(0 2 3 4 5 6 7); NG=${#GPUS[@]}
echo "[wolang] === precompute no-language multi-layer $(date +%H:%M:%S) ==="
rm -rf runs/ctx_cache_ml_full_nolang; mkdir -p runs/ctx_cache_ml_full_nolang
s=0; for g in "${GPUS[@]}"; do python -u -m train.backbone_kv --config $CFG --layers 4 8 12 16 24 --shard "$s" --num-shards "$NG" --device "cuda:$g" & s=$((s+1)); done; wait
echo "[wolang] === Stage-2 no-language $(date +%H:%M:%S) ==="
python -u -m train.stage2 --config $CFG --device cuda:0
echo "[wolang] ALL DONE $(date +%H:%M:%S)"
