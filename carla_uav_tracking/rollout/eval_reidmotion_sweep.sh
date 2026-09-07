#!/usr/bin/env bash
# borrow#1 robustness sweep: RMOT 0.3 and 0.7 (0.5 already done -> paper_ext_reidmotion.json).
# Same l5 config + --reid-motion, 20ep same seeds. Single CARLA — caller confirms FREE/NO_BG.
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}; NET=cyh-carla-net; CONFIG=train/config_v5.yaml
EPISODES=${EPISODES:-20}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-2}
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
run () {  # rmot tag
  local rmot=$1; local tag=$2; local out=/workspace/rollout/eval_out/paper_ext_reidmotion${tag}.json
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
  docker run -d $GPUFLAG --network "$NET" --name acot-policy-server --shm-size=16g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
      python carla_uav_tracking/rollout/policy_server.py --config '$CONFIG' \
      --ckpt runs/stage2_v5/stage2_best.pt --language-mode neutral --s2-period '$S2' \
      --ex-source track --tid-head reid --reid-feature dinov2 --reid-bank 2 --reid-topm 2 \
      --conf-tau 0.05 --reid-tavg 0.4 --reid-motion '$rmot' --port 5555" >/dev/null
  for _ in $(seq 1 120); do
    docker logs acot-policy-server 2>&1 | grep -q "listening" && break
    docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "[$tag] died:"; docker logs acot-policy-server|tail -12; return 1; }
    sleep 3
  done
  echo "=== [RMOT=$rmot] $EPISODES ep ==="
  docker exec cyh-carla python /workspace/rollout/run_rollout.py --policy-host acot-policy-server \
    --policy-port 5555 --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$out" || echo "[$tag] FAILED"
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
}
run 0.3 03
run 0.7 07
echo "=== DONE sweep. curve l5/0.3/0.5/0.7: ==="
docker exec cyh-carla python /workspace/rollout/continuous_metrics.py \
  /workspace/rollout/eval_out/paper_l5_dino_conf_tavg.json \
  /workspace/rollout/eval_out/paper_ext_reidmotion03.json \
  /workspace/rollout/eval_out/paper_ext_reidmotion.json \
  /workspace/rollout/eval_out/paper_ext_reidmotion07.json || true
