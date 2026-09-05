#!/usr/bin/env bash
# BC-centering closed-loop PAIRED eval: does fine-tuning the DiT toward centering actions
# raise closed-loop centering (and SR) WITHOUT breaking identity? Two arms, SAME seeds,
# SAME track+reid config — only the ckpt differs:
#   base = runs/stage2_v5/stage2_best.pt      (off-center-by-design DiT)
#   bc   = runs/stage2_bc_center.pt           (DiT last-4 fine-tuned to center; offline corr 0.36->0.97)
# Both run --ex-source track --tid-head reid --reid-feature dinov2 --reid-bank 2 (the 0.105 config).
# GATE: centering_rate ↑, mis_follow NOT ↑, SR NOT ↓ (BC beats base on the SAME seeds).
#
#   EPISODES=4 SEED_BASE=91000 STEPS=900 GPU=2 bash carla_uav_tracking/rollout/eval_bc_center.sh
set -euo pipefail

REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}
NET=cyh-carla-net
CONFIG=${CONFIG:-train/config_v5.yaml}
EPISODES=${EPISODES:-4}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-2}
REIDBANK=${REIDBANK:-2}; CONF_TAU=${CONF_TAU:-0.0}
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT_HOST=$REPO/carla_uav_tracking/rollout/eval_out; OUT_CTR=/workspace/rollout/eval_out
mkdir -p "$OUT_HOST"

run_arm () {  # name ckpt out
  local name=$1 ckpt=$2 out=$3
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
  docker run -d $GPUFLAG --network "$NET" --name acot-policy-server --shm-size=16g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
      python carla_uav_tracking/rollout/policy_server.py \
      --config '$CONFIG' --ckpt '$ckpt' --language-mode neutral --s2-period '$S2' \
      --ex-source track --tid-head reid --reid-feature dinov2 --reid-bank '$REIDBANK' \
      --conf-tau '$CONF_TAU' --port 5555" >/dev/null
  for _ in $(seq 1 100); do
    docker logs acot-policy-server 2>&1 | grep -q "listening" && break
    docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "[$name] server died:"; docker logs acot-policy-server; exit 1; }
    sleep 3
  done
  echo "=== [$name] $EPISODES ep × $STEPS on $TOWN (ckpt=$ckpt, reid=dinov2 K$REIDBANK tau=$CONF_TAU) ==="
  docker exec cyh-carla python /workspace/rollout/run_rollout.py \
    --policy-host acot-policy-server --policy-port 5555 \
    --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$out"
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
}

# ARMS = space-separated "name:ckpt" pairs (default: base vs full-centering)
ARMS=${ARMS:-"base:runs/stage2_v5/stage2_best.pt bc:runs/stage2_bc_center.pt"}
for spec in $ARMS; do
  name=${spec%%:*}; ckpt=${spec#*:}
  run_arm "$name" "$ckpt" "$OUT_CTR/bc_${name}.json"
done

echo ""
echo "=== PAIRED comparison (same seeds), all arms vs base ==="
base_name=$(echo $ARMS | awk '{print $1}'); base_name=${base_name%%:*}
for spec in $ARMS; do
  name=${spec%%:*}; [ "$name" = "$base_name" ] && continue
  echo "--- $base_name vs $name ---"
  docker exec cyh-carla python /workspace/rollout/compare_runs.py \
    "$OUT_CTR/bc_${base_name}.json" "$OUT_CTR/bc_${name}.json" --label-a "$base_name" --label-b "$name"
done
