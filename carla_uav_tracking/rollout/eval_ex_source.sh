#!/usr/bin/env bash
# Closed-loop no-WM+memory baseline: SAME ckpt, vary what fills DiT z_ex.
#   ear         = reactive no-WM control (reproduces reacq_nowm.json ~27.5%)
#   cv_gated    = + memory+CV target-state prior during a loss  (the new baseline)
#   zerovel_gated (optional) = frozen-last-seen prior
# Offline (baseline_deadreckon) said CV crushes EAR on straight losses, collapses on
# turns -> expect closed-loop reacquire UP on straight, turn residual left for WM.
# All arms share held-out seeds (91000+), so it's a paired comparison (compare_runs.py).
#
#   EPISODES=20 SEED_BASE=91000 STEPS=900 TOWN=Town05 S2_PERIOD=6 GPU=2 \
#     bash carla_uav_tracking/rollout/eval_ex_source.sh
set -euo pipefail

REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}
NET=cyh-carla-net
CKPT=${CKPT:-runs/stage2_v5/stage2_best.pt}          # SAME ckpt as the no-WM baseline
CONFIG=${CONFIG:-train/config_v5.yaml}
EPISODES=${EPISODES:-20}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-2}
ARMS=${ARMS:-"ear cv_gated zerovel_gated"}           # space-separated ex-source list
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT_HOST=$REPO/carla_uav_tracking/rollout/eval_out; OUT_CTR=/workspace/rollout/eval_out
mkdir -p "$OUT_HOST"

run_arm () {  # ex_source out
  local ex=$1 out=$2
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
  docker run -d $GPUFLAG --network "$NET" --name acot-policy-server --shm-size=16g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 python carla_uav_tracking/rollout/policy_server.py \
      --config "$CONFIG" --ckpt "$CKPT" --language-mode neutral --s2-period "$S2" \
      --ex-source "$ex" --port 5555 >/dev/null
  for _ in $(seq 1 80); do
    docker logs acot-policy-server 2>&1 | grep -q "listening" && break
    docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "[$ex] server died:"; docker logs acot-policy-server; exit 1; }
    sleep 3
  done
  echo "=== [$ex] $EPISODES ep × $STEPS on $TOWN (ckpt=$CKPT) ==="
  docker exec cyh-carla python /workspace/rollout/run_rollout.py \
    --policy-host acot-policy-server --policy-port 5555 \
    --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$out"
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
}

for ex in $ARMS; do
  run_arm "$ex" "$OUT_CTR/ex_${ex}.json"
done

echo ""
echo "=== paired comparison vs ear (no-WM control) ==="
for ex in $ARMS; do
  [ "$ex" = "ear" ] && continue
  echo "--- ear vs ${ex} ---"
  docker exec cyh-carla python /workspace/rollout/compare_runs.py \
    "$OUT_CTR/ex_ear.json" "$OUT_CTR/ex_${ex}.json" --label-a ear --label-b "$ex"
done
