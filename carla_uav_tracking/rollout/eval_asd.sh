#!/usr/bin/env bash
# ASD (Anchored Sequential Drift-correction) PoC closed-loop eval — parametric, parallel-safe.
# Each run = its own policy-server container (SRV) + its own CARLA server (CARLA_PORT) → many arms in parallel.
#   SRV=asd-a0 CARLA_PORT=2000 GPU=1 OUT_NAME=paper_asd_x HEAD_ARGS="--tid-head tah ..." bash eval_asd.sh
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}; NET=cyh-carla-net; CONFIG=train/config_v5.yaml
EPISODES=${EPISODES:-20}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-1}; SRV=${SRV:-asd-srv}; CARLA_PORT=${CARLA_PORT:-2000}
OUT_NAME=${OUT_NAME:?}; HEAD_ARGS=${HEAD_ARGS:?}; ASD_LOG=${ASD_LOG:-}
OUT=/workspace/rollout/eval_out/${OUT_NAME}.json
docker rm -f $SRV >/dev/null 2>&1 || true
docker run -d --gpus device=$GPU --network "$NET" --name $SRV --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e ASD_LOG="$ASD_LOG" -e ASD_CFG="${ASD_CFG:-}" -e COLD_WINDOW="${COLD_WINDOW:-0}" -e COLD_LAMBDA="${COLD_LAMBDA:-0.9}" -e COLD_SINGLE="${COLD_SINGLE:-0}" -e V1_SELFRESET="${V1_SELFRESET:-0}" -e DAM4SAM_DIAG="${DAM4SAM_DIAG:-0}" -e DAM4SAM_EVERY_FRAME="${DAM4SAM_EVERY_FRAME:-0}" -e TRK_CFG="${TRK_CFG:-}" -e OPENING_DUMP="${OPENING_DUMP:-}" -e OPENING="${OPENING:-head}" -e OPENING_CFG="${OPENING_CFG:-}" -e ORACLE_ABSENT="${ORACLE_ABSENT:-0}" -e TAH_DAGGER_DIR="${TAH_DAGGER_DIR:-}" -e OMP_NUM_THREADS=${NTHR:-6} -e MKL_NUM_THREADS=${NTHR:-6} \
  -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
  acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
    python carla_uav_tracking/rollout/policy_server.py --config '$CONFIG' \
    --ckpt runs/stage2_v5/stage2_best.pt --language-mode neutral --s2-period '$S2' \
    --ex-source track $HEAD_ARGS --port 5555" >/dev/null
for _ in $(seq 1 150); do
  docker logs $SRV 2>&1 | grep -q "listening" && break
  docker ps --format '{{.Names}}' | grep -q "^$SRV\$" || { echo "server died:"; docker logs $SRV|tail -25; exit 1; }
  sleep 3
done
echo "=== [$OUT_NAME] $EPISODES ep seed=$SEED_BASE town=$TOWN carla=$CARLA_PORT gpu=$GPU :: $HEAD_ARGS ==="
docker exec -e DUMP_LABELS="${DUMP_LABELS:-}" -e CARLA_PORT=$CARLA_PORT -e CARLA_TM_PORT=${TM_PORT:-$((CARLA_PORT+6000))} cyh-carla python /workspace/rollout/run_rollout.py --policy-host $SRV \
  --policy-port 5555 --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$OUT" ${VIS_RAYCAST:+--vis-raycast} ${HIRES:+--hires $HIRES} || echo FAILED
docker logs $SRV 2>&1 | grep -E "\[asd\]|\[reanchor\]|Traceback|Error" | tail -40
docker rm -f $SRV >/dev/null 2>&1 || true
echo "=== DONE $OUT_NAME ==="
