#!/usr/bin/env bash
# Recovery after the no-language experiment was killed by a system event at ep10.
# 1) finish mmap conversion of the caches (safe now: nothing is running on them)
# 2) relaunch the 3 no-language seeds on the mmap cache with workers=4.
# OMP capped at 4 so 3 runs x (main+4 workers) x 4 threads = 60 < 80 cores.
set -u
mkdir -p runs/seed_exp

echo "[recover] convert caches to mmap  ($(date +%H:%M:%S))"
python -u -m train.mmap_cache runs/ctx_cache_ml runs/ctx_cache_ml_nolang

echo "[recover] relaunch 3 no-language seeds (mmap + workers=4)  ($(date +%H:%M:%S))"
run(){  # $1=seed $2=gpu
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  python -u -m train.stage2 --config train/config_nolang.yaml --seed "$1" --device "cuda:$2" \
    --ckpt-dir "runs/seed_exp/nolang_s$1" > "runs/seed_exp/nolang_s$1.log" 2>&1
  echo "[recover] DONE seed=$1  ($(date +%H:%M:%S))"
}
run 0 0 &
run 1 1 &
run 2 2 &
wait
echo "[recover] ALL DONE  ($(date +%H:%M:%S))"
