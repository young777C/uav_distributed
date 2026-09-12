#!/usr/bin/env bash
# rebuild the pruned acot-uav-train:cu118 image, then run the deployable (committed + lang cold-start +
# reanchor) TAHRelM eval. Default bridge is 192.168.x (not 172.x) → build respects the net constraint.
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; cd "$REPO"
say(){ echo "[deploy $(date +%H:%M:%S)] $*"; }
if ! docker image inspect acot-uav-train:cu118 >/dev/null 2>&1; then
  say "image missing — rebuilding (base ~5GB pull + pip)..."
  bash train/docker/build.sh 2>&1 | tail -25
fi
docker image inspect acot-uav-train:cu118 >/dev/null 2>&1 || { say "BUILD FAILED"; exit 1; }
say "image ready — launching deployable eval (committed + reanchor=lang)"
docker rm -f acot-policy-server >/dev/null 2>&1 || true
TAH_CKPT=runs/tah_relm_dag.pt OUT_NAME=paper_ext_tahrelm_deploy \
  GSEED=committed REANCHOR=lang RPAT=5 CONF=0.5 \
  EPISODES=20 SEED_BASE=91000 STEPS=900 GPU=2 \
  bash carla_uav_tracking/rollout/eval_tah.sh 2>&1 | tail -6
say "=== compare gt / committed(no-lang) / deploy(committed+lang+reanchor) ==="
docker exec cyh-carla python /workspace/rollout/continuous_metrics.py \
  /workspace/rollout/eval_out/paper_ext_tahrelm_gt.json \
  /workspace/rollout/eval_out/paper_ext_tahrelm_committed.json \
  /workspace/rollout/eval_out/paper_ext_tahrelm_deploy.json 2>&1 | tail -16
say "=== DEPLOY-EVAL DONE ==="
