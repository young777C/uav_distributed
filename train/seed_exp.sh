#!/usr/bin/env bash
# Seed experiment: baseline (orig language + orig curriculum) vs A+C (neutral
# language + fixed curriculum: a_floor 0.25, ramp 0.9, grad_clip 1.0), 3 seeds each.
# Each run is single-GPU; 3 concurrent per wave (RAM-bound: ~50GB/run).
set -u
mkdir -p runs/seed_exp

run(){  # $1=config  $2=tag  $3=seed  $4=gpu
  echo "[seed_exp] START $2 seed=$3 gpu=$4  ($(date +%H:%M:%S))"
  # cap CPU threads per process: 3 concurrent x 20 = 60 < 80 cores (avoid BLAS thrash)
  OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 NUMEXPR_NUM_THREADS=20 \
  python -u -m train.stage2 --config "$1" --seed "$3" --device "cuda:$4" \
    --ckpt-dir "runs/seed_exp/$2_s$3" > "runs/seed_exp/$2_s$3.log" 2>&1
  echo "[seed_exp] DONE  $2 seed=$3  ($(date +%H:%M:%S))"
}

echo "[seed_exp] wave 1 = baseline x3"
run train/config.yaml    base 0 0 &
run train/config.yaml    base 1 1 &
run train/config.yaml    base 2 2 &
wait

echo "[seed_exp] wave 2 = A+C x3"
run train/config_ac.yaml ac   0 3 &
run train/config_ac.yaml ac   1 4 &
run train/config_ac.yaml ac   2 5 &
wait

echo "[seed_exp] ALL DONE"
