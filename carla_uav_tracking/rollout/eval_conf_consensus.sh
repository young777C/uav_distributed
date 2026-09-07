#!/usr/bin/env bash
# borrow#2: consensus-gated control ON TOP of borrow#1 (RMOT=0.3). Same l5 deployable config +
# --reid-motion 0.3 + --conf-consensus. Pairs vs borrow#1 (paper_ext_reidmotion03.json, no consensus).
#   CONS=1.0 RMOT=0.3 EPISODES=20 SEED_BASE=91000 STEPS=900 GPU=2 bash carla_uav_tracking/rollout/eval_conf_consensus.sh
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}; NET=cyh-carla-net; CONFIG=train/config_v5.yaml
EPISODES=${EPISODES:-20}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-2}; RMOT=${RMOT:-0.3}; CONS=${CONS:-1.0}
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT=/workspace/rollout/eval_out/paper_ext_reidmotion03_cons${CONS}.json
docker rm -f acot-policy-server >/dev/null 2>&1 || true
docker run -d $GPUFLAG --network "$NET" --name acot-policy-server --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
  acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
    python carla_uav_tracking/rollout/policy_server.py --config '$CONFIG' \
    --ckpt runs/stage2_v5/stage2_best.pt --language-mode neutral --s2-period '$S2' \
    --ex-source track --tid-head reid --reid-feature dinov2 --reid-bank 2 --reid-topm 2 \
    --conf-tau 0.05 --reid-tavg 0.4 --reid-motion '$RMOT' --conf-consensus '$CONS' --port 5555" >/dev/null
for _ in $(seq 1 120); do
  docker logs acot-policy-server 2>&1 | grep -q "listening" && break
  docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "server died:"; docker logs acot-policy-server|tail -15; exit 1; }
  sleep 3
done
echo "=== [borrow#1+#2 RMOT=$RMOT CONS=$CONS] $EPISODES ep ==="
docker exec cyh-carla python /workspace/rollout/run_rollout.py --policy-host acot-policy-server \
  --policy-port 5555 --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$OUT" || echo FAILED
docker rm -f acot-policy-server >/dev/null 2>&1 || true
echo "=== DONE consensus. compare l5 / borrow#1(m0.3) / +#2consensus: ==="
docker exec cyh-carla python /workspace/rollout/continuous_metrics.py \
  /workspace/rollout/eval_out/paper_l5_dino_conf_tavg.json \
  /workspace/rollout/eval_out/paper_ext_reidmotion03.json "$OUT" || true
