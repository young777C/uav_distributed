#!/usr/bin/env bash
# E1 language-necessity ablation on mvp_full_v3 (depth shortcut closed: nearest 31%->80%,
# MLP floor 40%->54%). Honest metric (candidate order now shuffled in dataset_stage2).
# with-language = neutralize_language (identity: color+make, no positional leak)
# no-language   = "Track the target vehicle." (generic)
# SKIPS GPU1 (CARLA data-gen). Reuses one EAR warmstart for both arms (isolates stage2).
set -u
CW=train/config_v3.yaml           # with-language (identity)
CN=train/config_v3_nolang.yaml    # no-language
GPUS=(0 2 3 4 5 6 7); NG=${#GPUS[@]}
pc(){ # $1=config $2=cache_dir $3...=extra (e.g. --layers ...)
  local cfg="$1" dir="$2"; shift 2
  rm -rf "$dir"; mkdir -p "$dir"
  local s=0; for g in "${GPUS[@]}"; do
    python -u -m train.backbone_kv --config "$cfg" "$@" --shard "$s" --num-shards "$NG" --device "cuda:$g" & s=$((s+1))
  done; wait
}

echo "[v3-e1] === precompute single-layer with-lang (EAR)  $(date +%H:%M:%S) ==="
pc "$CW" runs/ctx_cache_v3

echo "[v3-e1] === precompute multi-layer with-lang (stage2)  $(date +%H:%M:%S) ==="
pc "$CW" runs/ctx_cache_ml_v3 --layers 4 8 12 16 24

echo "[v3-e1] === precompute multi-layer NO-lang (stage2)  $(date +%H:%M:%S) ==="
pc "$CN" runs/ctx_cache_ml_v3_nolang --layers 4 8 12 16 24

echo "[v3-e1] === Stage-1 EAR (with-lang single cache)  $(date +%H:%M:%S) ==="
python -u -m train.stage1_ear --config "$CW"

echo "[v3-e1] === Stage-2 BOTH arms in parallel  $(date +%H:%M:%S) ==="
OMP_NUM_THREADS=8 python -u -m train.stage2 --config "$CW" --device cuda:2 > runs/train_v3.log        2>&1 &
OMP_NUM_THREADS=8 python -u -m train.stage2 --config "$CN" --device cuda:3 > runs/train_v3_nolang.log 2>&1 &
wait
echo "[v3-e1] ALL DONE  $(date +%H:%M:%S)"
