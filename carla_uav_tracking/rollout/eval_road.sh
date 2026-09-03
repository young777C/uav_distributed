#!/usr/bin/env bash
# P2a-1 · M3 closed-loop gate: road-graph WM prior vs the no-WM baselines.
#   ear       = reactive no-WM control
#   cv_gated  = memory+CV straight dead-reckon during a loss (the no-WM baseline)
#   road      = memory+ROAD-GRAPH lane-traversal during a loss (this contribution)
# Same ckpt, same held-out seeds → paired. The road arm ALSO passes run_rollout
# --predict-road so the env computes the lane-traversal (carla.Map) → gt_candidates[3];
# the other arms don't (no CARLA-map cost). Offline M2 said road << cv_frame on
# turns/junctions → expect closed-loop turn-REACQUIRE up for road vs cv_gated.
#
#   EPISODES=20 SEED_BASE=91000 STEPS=900 TOWN=Town05 S2_PERIOD=6 GPU=2 \
#     bash carla_uav_tracking/rollout/eval_road.sh
set -euo pipefail

REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}
NET=cyh-carla-net
CKPT=${CKPT:-runs/stage2_v5/stage2_best.pt}
CONFIG=${CONFIG:-train/config_v5.yaml}
EPISODES=${EPISODES:-20}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-2}
ARMS=${ARMS:-"ear cv_gated road"}
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT_HOST=$REPO/carla_uav_tracking/rollout/eval_out; OUT_CTR=/workspace/rollout/eval_out
mkdir -p "$OUT_HOST"

run_arm () {  # ex_source out
  local ex=$1 out=$2 extra="" srcex=$1
  [ "$ex" = "road" ] && extra="--predict-road"          # env computes lane-traversal only for road
  # road_oracle (P2a-2 positive control): same policy (ex-source road), env anchors at TRUE current
  [ "$ex" = "road_oracle" ] && { extra="--road-oracle"; srcex="road"; }
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
  docker run -d $GPUFLAG --network "$NET" --name acot-policy-server --shm-size=16g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 python carla_uav_tracking/rollout/policy_server.py \
      --config "$CONFIG" --ckpt "$CKPT" --language-mode neutral --s2-period "$S2" \
      --ex-source "$srcex" --port 5555 >/dev/null
  for _ in $(seq 1 80); do
    docker logs acot-policy-server 2>&1 | grep -q "listening" && break
    docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "[$ex] server died:"; docker logs acot-policy-server; exit 1; }
    sleep 3
  done
  echo "=== [$ex] $EPISODES ep × $STEPS on $TOWN (ckpt=$CKPT) ${extra} ==="
  docker exec cyh-carla python /workspace/rollout/run_rollout.py \
    --policy-host acot-policy-server --policy-port 5555 \
    --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" $extra --out "$out"
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
}

for ex in $ARMS; do
  run_arm "$ex" "$OUT_CTR/road_${ex}.json"
done

echo ""
echo "=== paired comparison vs cv_gated (no-WM baseline) ==="
for ex in $ARMS; do
  [ "$ex" = "cv_gated" ] && continue
  echo "--- cv_gated vs ${ex} ---"
  docker exec cyh-carla python /workspace/rollout/compare_runs.py \
    "$OUT_CTR/road_cv_gated.json" "$OUT_CTR/road_${ex}.json" --label-a cv_gated --label-b "$ex" || true
done
