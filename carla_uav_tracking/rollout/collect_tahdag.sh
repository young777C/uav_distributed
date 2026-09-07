#!/usr/bin/env bash
# path-B DAgger collection: run tah gt-seeded on COLLECTION seeds (92000+, disjoint from eval 91000-19),
# log deployment-distribution TAH frames to TAH_DAGGER_DIR for on-policy retraining.
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}; NET=cyh-carla-net
EPISODES=${EPISODES:-24}; SEED_BASE=${SEED_BASE:-92000}; STEPS=${STEPS:-900}; GPU=${GPU:-2}
docker rm -f acot-policy-server >/dev/null 2>&1 || true
docker run -d --gpus "\"device=$GPU\"" --network "$NET" --name acot-policy-server --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e TAH_DAGGER_DIR=runs/tahdag \
  -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
  acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
    python carla_uav_tracking/rollout/policy_server.py --config train/config_v5.yaml \
    --ckpt runs/stage2_v5/stage2_best.pt --language-mode neutral --s2-period 6 \
    --ex-source track --tid-head tah --tah-ckpt runs/tah_rel.pt --gallery-seed gt --conf-tau 0.5 --port 5555" >/dev/null
for _ in $(seq 1 120); do docker logs acot-policy-server 2>&1 | grep -q listening && break
  docker ps --format '{{.Names}}'|grep -q acot-policy-server || { echo died; docker logs acot-policy-server|tail; exit 1; }; sleep 3; done
echo "=== collect $EPISODES ep from seed $SEED_BASE ==="
docker exec cyh-carla python /workspace/rollout/run_rollout.py --policy-host acot-policy-server \
  --policy-port 5555 --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town Town05 \
  --out /workspace/rollout/eval_out/tahdag_collect.json || echo FAILED
docker rm -f acot-policy-server >/dev/null 2>&1 || true
echo "=== DONE collect. tahdag files: $(ls $REPO/runs/tahdag/*.pt 2>/dev/null | wc -l) ==="
