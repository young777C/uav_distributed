#!/usr/bin/env bash
# Does the better grounding head recover the closed-loop reacquire gap? Both arms use
# ex_source=track (grounding DRIVES control); the ONLY diff is the tid head:
#   track_xattn = current cross-attn tid (mis_follow 0.51)
#   track_attr  = standalone CLIP-style attrbind head (mis_follow 0.44), swapped in
#                 (no Stage-2 retrain — tid is decoupled). GT target color/make = the
#                 offline attr upper bound, sent over the wire.
# Same seeds → paired. cyh-carla must have CARLA up on 2012.
#   EPISODES=20 SEED_BASE=91000 STEPS=900 GPU=2 bash carla_uav_tracking/rollout/eval_attrbind.sh
set -euo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}
NET=cyh-carla-net
CKPT=${CKPT:-runs/stage2_v5/stage2_best.pt}
CONFIG=${CONFIG:-train/config_v5.yaml}
ATTR_CKPT=${ATTR_CKPT:-runs/tid_attrbind.pt}
EPISODES=${EPISODES:-20}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-2}
OUT_HOST=$REPO/carla_uav_tracking/rollout/eval_out; OUT_CTR=/workspace/rollout/eval_out
mkdir -p "$OUT_HOST"

run_arm () {  # name tid_head tid_flag out
  local name=$1 thead=$2 tflag=$3 out=$4
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
  docker run -d --gpus all --network "$NET" --name acot-policy-server --shm-size=16g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 python carla_uav_tracking/rollout/policy_server.py \
      --config "$CONFIG" --ckpt "$CKPT" --language-mode neutral --s2-period "$S2" \
      --ex-source track --tid-head "$thead" $tflag --device "cuda:$GPU" --port 5555 >/dev/null
  for _ in $(seq 1 80); do
    docker logs acot-policy-server 2>&1 | grep -q "listening" && break
    docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "[$name] server died:"; docker logs acot-policy-server; exit 1; }
    sleep 3
  done
  echo "=== [$name] $EPISODES ep × $STEPS on $TOWN ==="
  docker exec cyh-carla python /workspace/rollout/run_rollout.py \
    --policy-host acot-policy-server --policy-port 5555 \
    --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$out"
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
}

run_arm track_xattn xattn    ""                        "$OUT_CTR/track_xattn.json"
run_arm track_attr  attrbind "--tid-ckpt $ATTR_CKPT"   "$OUT_CTR/track_attr.json"

echo ""
echo "=== paired: track_xattn vs track_attr (does better grounding recover reacquire?) ==="
docker exec cyh-carla python /workspace/rollout/compare_runs.py \
  "$OUT_CTR/track_xattn.json" "$OUT_CTR/track_attr.json" --label-a track_xattn --label-b track_attr
