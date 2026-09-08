#!/usr/bin/env bash
# path-B closed-loop: tid_head=tah (learned TAHRel) vs the heuristic stack (l5 / borrow#1). Same track
# control. GSEED=gt (oracle-seeded memory, comparable to l5/borrow#1 gallery_seed=gt) | committed (deployable).
#   GSEED=gt EPISODES=20 SEED_BASE=91000 STEPS=900 GPU=2 bash carla_uav_tracking/rollout/eval_tah.sh
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}; NET=cyh-carla-net; CONFIG=train/config_v5.yaml
EPISODES=${EPISODES:-20}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-2}; GSEED=${GSEED:-gt}; CONF=${CONF:-0.5}
TAH_CKPT=${TAH_CKPT:-runs/tah_rel.pt}; OUT_NAME=${OUT_NAME:-paper_ext_tah_${GSEED}}
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT=/workspace/rollout/eval_out/${OUT_NAME}.json
docker rm -f acot-policy-server >/dev/null 2>&1 || true
docker run -d $GPUFLAG --network "$NET" --name acot-policy-server --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
  acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
    python carla_uav_tracking/rollout/policy_server.py --config '$CONFIG' \
    --ckpt runs/stage2_v5/stage2_best.pt --language-mode neutral --s2-period '$S2' \
    --ex-source track --tid-head tah --tah-ckpt '$TAH_CKPT' --gallery-seed '$GSEED' \
    --conf-tau '$CONF' --port 5555" >/dev/null
for _ in $(seq 1 120); do
  docker logs acot-policy-server 2>&1 | grep -q "listening" && break
  docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "server died:"; docker logs acot-policy-server|tail -15; exit 1; }
  sleep 3
done
echo "=== [tah GSEED=$GSEED CONF=$CONF] $EPISODES ep ==="
docker exec cyh-carla python /workspace/rollout/run_rollout.py --policy-host acot-policy-server \
  --policy-port 5555 --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$OUT" || echo FAILED
docker rm -f acot-policy-server >/dev/null 2>&1 || true
echo "=== DONE tah. compare l5 / borrow#1(m0.3) / tah: ==="
docker exec cyh-carla python /workspace/rollout/continuous_metrics.py \
  /workspace/rollout/eval_out/paper_l5_dino_conf_tavg.json \
  /workspace/rollout/eval_out/paper_ext_reidmotion03.json "$OUT" || true
