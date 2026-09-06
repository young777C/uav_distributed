#!/usr/bin/env bash
# External baseline eval: tid_head=assoc (DeepSORT/OC-SORT tracking-by-detection), same deployable
# config as the ladder's l4 (track + conf-gate) — only the WHICH mechanism differs. 20ep seeds 91000-19.
#   EPISODES=20 SEED_BASE=91000 STEPS=900 GPU=2 bash carla_uav_tracking/rollout/eval_ext_assoc.sh
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}
NET=cyh-carla-net; CONFIG=${CONFIG:-train/config_v5.yaml}
EPISODES=${EPISODES:-20}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-2}
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT_CTR=/workspace/rollout/eval_out
ARMS=(
  "ext_assoc_motion|--ex-source track --tid-head assoc --assoc-mode motion --conf-tau 0.05"
  "ext_assoc_deepsort|--ex-source track --tid-head assoc --assoc-mode deepsort --conf-tau 0.05"
)
run_arm () {  # name flags out
  local name=$1 flags=$2 out=$3
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
  docker run -d $GPUFLAG --network "$NET" --name acot-policy-server --shm-size=16g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
      python carla_uav_tracking/rollout/policy_server.py --config '$CONFIG' \
      --ckpt runs/stage2_v5/stage2_best.pt --language-mode neutral --s2-period '$S2' $flags --port 5555" >/dev/null
  for _ in $(seq 1 120); do
    docker logs acot-policy-server 2>&1 | grep -q "listening" && break
    docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "[$name] server died:"; docker logs acot-policy-server | tail -20; return 1; }
    sleep 3
  done
  echo "=== [$name] $EPISODES ep × $STEPS ($flags) ==="
  docker exec cyh-carla python /workspace/rollout/run_rollout.py --policy-host acot-policy-server \
    --policy-port 5555 --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$out" \
    || echo "[$name] FAILED"
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
}
for spec in "${ARMS[@]}"; do
  IFS='|' read -r name flags <<< "$spec"
  run_arm "$name" "$flags" "$OUT_CTR/paper_${name}.json"
done
echo "=== DONE assoc. compare: continuous_metrics.py paper_l4_dino_conf.json paper_l5_dino_conf_tavg.json paper_ext_assoc_motion.json paper_ext_assoc_deepsort.json ==="
