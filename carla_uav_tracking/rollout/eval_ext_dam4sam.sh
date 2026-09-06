#!/usr/bin/env bash
# External baseline eval: tid_head=dam4sam (DAM4SAM SOTA distractor-aware SAM2 memory via socket
# service dam4sam-svc:5601). Same deployable config as ladder l4 (track + conf-gate) — only WHICH
# differs. Requires dam4sam-svc container running the service. 20ep seeds 91000-19.
#   GPU=4 EPISODES=20 SEED_BASE=91000 STEPS=900 bash carla_uav_tracking/rollout/eval_ext_dam4sam.sh
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}; NET=cyh-carla-net; CONFIG=${CONFIG:-train/config_v5.yaml}
EPISODES=${EPISODES:-20}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-4}
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT=/workspace/rollout/eval_out/paper_ext_dam4sam.json
docker rm -f acot-policy-server >/dev/null 2>&1 || true
docker run -d $GPUFLAG --network "$NET" --name acot-policy-server --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
  acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
    python carla_uav_tracking/rollout/policy_server.py --config '$CONFIG' \
    --ckpt runs/stage2_v5/stage2_best.pt --language-mode neutral --s2-period '$S2' \
    --ex-source track --tid-head dam4sam --dam4sam-host dam4sam-svc --dam4sam-port 5601 \
    --conf-tau 0.05 --port 5555" >/dev/null
for _ in $(seq 1 120); do
  docker logs acot-policy-server 2>&1 | grep -q "listening" && break
  docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "server died:"; docker logs acot-policy-server | tail -20; exit 1; }
  sleep 3
done
echo "=== [dam4sam] $EPISODES ep × $STEPS (SOTA distractor-aware SAM2 memory) ==="
docker exec cyh-carla python /workspace/rollout/run_rollout.py --policy-host acot-policy-server \
  --policy-port 5555 --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$OUT" \
  || echo "[dam4sam] FAILED"
docker rm -f acot-policy-server >/dev/null 2>&1 || true
echo "=== DONE dam4sam ==="
