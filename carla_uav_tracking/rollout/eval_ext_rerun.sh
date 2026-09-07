#!/usr/bin/env bash
# Re-run the external-baseline arms that failed when CARLA crashed: assoc-deepsort + dam4sam.
# (assoc-motion already done -> paper_ext_assoc_motion.json.) Sequential (single CARLA).
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}; NET=cyh-carla-net; CONFIG=train/config_v5.yaml
EPISODES=20; SEED_BASE=91000; STEPS=900; TOWN=Town05; S2=6
OUT_CTR=/workspace/rollout/eval_out

run_ps () {  # gpu flags -> starts acot-policy-server, waits for listening
  local gpu=$1 flags=$2
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
  docker run -d --gpus "\"device=$gpu\"" --network "$NET" --name acot-policy-server --shm-size=16g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
      python carla_uav_tracking/rollout/policy_server.py --config '$CONFIG' \
      --ckpt runs/stage2_v5/stage2_best.pt --language-mode neutral --s2-period '$S2' $flags --port 5555" >/dev/null
  for _ in $(seq 1 120); do
    docker logs acot-policy-server 2>&1 | grep -q "listening" && return 0
    docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "server died:"; docker logs acot-policy-server|tail -15; return 1; }
    sleep 3
  done
}
rollout () {  # out
  docker exec cyh-carla python /workspace/rollout/run_rollout.py --policy-host acot-policy-server \
    --policy-port 5555 --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$1" \
    || echo "FAILED $1"
}

echo "=== [assoc_deepsort] rerun ==="
run_ps 2 "--ex-source track --tid-head assoc --assoc-mode deepsort --conf-tau 0.05" && \
  rollout "$OUT_CTR/paper_ext_assoc_deepsort.json"
docker rm -f acot-policy-server >/dev/null 2>&1 || true

echo "=== [dam4sam] rerun ==="
run_ps 4 "--ex-source track --tid-head dam4sam --dam4sam-host dam4sam-svc --dam4sam-port 5601 --conf-tau 0.05" && \
  rollout "$OUT_CTR/paper_ext_dam4sam.json"
docker rm -f acot-policy-server >/dev/null 2>&1 || true
echo "=== DONE rerun ==="
