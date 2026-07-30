#!/usr/bin/env bash
# Formal w/o-language ablation: precompute a NO-LANGUAGE cache (generic prompt,
# zero target identity) on all 8 GPUs, then train 3 seeds. Compare mis_follow to
# the with-language arm (A+C, neutral language, same fixed curriculum).
set -u
mkdir -p runs/seed_exp

echo "[nolang] precompute (generic prompt) on 8 GPUs  ($(date +%H:%M:%S))"
rm -rf runs/ctx_cache_ml_nolang; mkdir -p runs/ctx_cache_ml_nolang
for i in 0 1 2 3 4 5 6 7; do
  python -u -m train.backbone_kv --config train/config_nolang.yaml \
    --layers 4 8 12 16 24 --shard "$i" --num-shards 8 --device "cuda:$i" &
done
wait
echo "[nolang] precompute DONE -> 3-seed stage2  ($(date +%H:%M:%S))"

run(){  # $1=seed $2=gpu
  echo "[nolang] START seed=$1 gpu=$2  ($(date +%H:%M:%S))"
  OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 NUMEXPR_NUM_THREADS=20 \
  python -u -m train.stage2 --config train/config_nolang.yaml --seed "$1" --device "cuda:$2" \
    --ckpt-dir "runs/seed_exp/nolang_s$1" > "runs/seed_exp/nolang_s$1.log" 2>&1
  echo "[nolang] DONE seed=$1  ($(date +%H:%M:%S))"
}
run 0 0 &
run 1 1 &
run 2 2 &
wait
echo "[nolang] ALL DONE  ($(date +%H:%M:%S))"
