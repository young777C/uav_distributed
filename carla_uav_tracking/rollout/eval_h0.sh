#!/usr/bin/env bash
# H0 closed-loop language-ablation evaluation (design §2/§7): run the WITH-language
# model (M0) and the W/O-language model (M1) on the SAME held-out seed set, then
# compare mis_follow_sustained (paired bootstrap CI). Language is load-bearing iff
# the w/o-language mis_follow increase is significant.
#
# Usage:  EPISODES=20 STEPS=1500 TOWN=Town05 S2_PERIOD=1 bash carla_uav_tracking/rollout/eval_h0.sh
# Long batches: run detached (nohup ... &) — at s2_period=1 each episode is minutes.
set -euo pipefail

REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}
NET=cyh-carla-net
EPISODES=${EPISODES:-10}
SEED_BASE=${SEED_BASE:-90000}
STEPS=${STEPS:-1500}
TOWN=${TOWN:-Town05}
S2=${S2_PERIOD:-1}
GPU=${GPU:-all}                              # pin the VLM server to a GPU, e.g. GPU=2
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT_HOST=$REPO/carla_uav_tracking/rollout/eval_out
OUT_CTR=/workspace/rollout/eval_out          # same dir seen inside cyh-carla (/workspace=carla_uav_tracking)
mkdir -p "$OUT_HOST"

run_arm () {
  local name=$1 config=$2 ckpt=$3 mode=$4 out=$5
  echo "=== [$name] starting policy server (config=$config mode=$mode) ==="
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
  docker run -d $GPUFLAG --network "$NET" --name acot-policy-server \
    --shm-size=16g -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 python carla_uav_tracking/rollout/policy_server.py \
      --config "$config" --ckpt "$ckpt" --language-mode "$mode" --s2-period "$S2" --port 5555 >/dev/null
  for _ in $(seq 1 80); do
    docker logs acot-policy-server 2>&1 | grep -q "listening" && break
    docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "[$name] server died:"; docker logs acot-policy-server; exit 1; }
    sleep 3
  done
  echo "=== [$name] server up; $EPISODES episodes × $STEPS steps on $TOWN (s2_period=$S2) ==="
  docker exec cyh-carla python /workspace/rollout/run_rollout.py \
    --policy-host acot-policy-server --policy-port 5555 \
    --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" \
    --out "$out"
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
}

run_arm M0_withlang train/config_v5.yaml        runs/stage2_v5/stage2_best.pt        neutral "$OUT_CTR/m0.json"
run_arm M1_nolang   train/config_v5_nolang.yaml runs/stage2_v5_nolang/stage2_best.pt none    "$OUT_CTR/m1.json"

echo "=== H0 paired comparison ==="
docker exec cyh-carla python /workspace/rollout/compare_runs.py \
  "$OUT_CTR/m0.json" "$OUT_CTR/m1.json" --label-a M0_withlang --label-b M1_nolang
echo "=== outputs in $OUT_HOST/{m0,m1}.json ==="
