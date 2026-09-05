#!/usr/bin/env bash
# Temporal re-ID closed-loop PAIRED eval: does per-actor temporal evidence accumulation (EMA of
# the reid match score) beat single-frame argmax? Same ckpt, same FULL deployable reid config
# (track + reid-dinov2 + K2 gallery + topm2 + conf-gate), SAME seeds — only --reid-tavg differs:
#   base = tavg 0    (single-frame argmax reid; the SR 0.105 deployable config)
#   temporal = tavg T (per-actor EMA → robust to off-center/far single frames → break cascade from
#                      the identity side; the reframe's temporal-association lever)
# GATE: mis_follow ↓ AND/OR lock_wrong ↓ AND SR ≥ base (temporal identity helps, doesn't hurt control).
#
#   EPISODES=6 SEED_BASE=91000 STEPS=900 TAVG=0.7 GPU=2 bash carla_uav_tracking/rollout/eval_reid_temporal.sh
set -euo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}
NET=cyh-carla-net
CKPT=${CKPT:-runs/stage2_v5/stage2_best.pt}
CONFIG=${CONFIG:-train/config_v5.yaml}
EPISODES=${EPISODES:-6}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-2}; TAVG=${TAVG:-0.7}
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT_CTR=/workspace/rollout/eval_out
mkdir -p "$REPO/carla_uav_tracking/rollout/eval_out"

run_arm () {  # name tavg out
  local name=$1 tavg=$2 out=$3
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
  docker run -d $GPUFLAG --network "$NET" --name acot-policy-server --shm-size=16g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
      python carla_uav_tracking/rollout/policy_server.py \
      --config '$CONFIG' --ckpt '$CKPT' --language-mode neutral --s2-period '$S2' \
      --ex-source track --tid-head reid --reid-feature dinov2 --reid-bank 2 --reid-topm 2 \
      --conf-tau 0.05 --reid-tavg '$tavg' --port 5555" >/dev/null
  for _ in $(seq 1 100); do
    docker logs acot-policy-server 2>&1 | grep -q "listening" && break
    docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "[$name] server died:"; docker logs acot-policy-server; exit 1; }
    sleep 3
  done
  echo "=== [$name] $EPISODES ep × $STEPS on $TOWN (reid-tavg=$tavg) ==="
  docker exec cyh-carla python /workspace/rollout/run_rollout.py \
    --policy-host acot-policy-server --policy-port 5555 \
    --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$out"
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
}

# SKIP_BASE=1 to reuse an existing reid_tavg0.json; TAVGS = space-separated list (default = $TAVG)
[ "${SKIP_BASE:-0}" = "1" ] || run_arm base 0 "$OUT_CTR/reid_tavg0.json"
for t in ${TAVGS:-$TAVG}; do
  run_arm "t$t" "$t" "$OUT_CTR/reid_tavg${t}.json"
done

echo ""
echo "=== PAIRED comparison (same seeds): single-frame reid (base) vs each tavg ==="
for t in ${TAVGS:-$TAVG}; do
  echo "--- reid0 vs tavg$t ---"
  docker exec cyh-carla python /workspace/rollout/compare_runs.py \
    "$OUT_CTR/reid_tavg0.json" "$OUT_CTR/reid_tavg${t}.json" --label-a reid0 --label-b "t${t}"
done
