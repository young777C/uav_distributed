#!/usr/bin/env bash
# Orchestrate H1/H2: wait for the two ablation trainings (detached docker) to finish,
# then run the closed-loop eval for w/o-EAR (H1) and w/o-IAR (H2), reusing the H0
# full-model run (eval_out/m0.json) as the paired baseline. Verbose to stdout.
#
#   nohup bash carla_uav_tracking/rollout/run_h1h2_pipeline.sh > /tmp/h1h2_pipeline.log 2>&1 &
set -uo pipefail
cd /nvidia/hque/code/cyhe/uav-acot-track
GPU=${GPU:-2}

echo "[pipeline] $(date -u +%H:%M:%S) waiting for ablation trainings to finish ..."
for _ in $(seq 1 240); do        # up to ~2h
  running=0
  docker ps --format '{{.Names}}' | grep -qx acot-train-noear && running=1
  docker ps --format '{{.Names}}' | grep -qx acot-train-noiar && running=1
  [ $running -eq 0 ] && break
  sleep 30
done
echo "[pipeline] $(date -u +%H:%M:%S) trainings no longer running. checkpoints:"
ls -la runs/stage2_v5_noear/stage2_best.pt runs/stage2_v5_noiar/stage2_best.pt 2>&1 || true

for abl in ear iar; do
  ck=runs/stage2_v5_no${abl}/stage2_best.pt
  if [ ! -f "$ck" ]; then
    echo "[pipeline] MISSING $ck — no${abl} training failed; skipping H eval"
    docker logs acot-train-no${abl} 2>&1 | tail -15
    continue
  fi
  echo "[pipeline] ===== H eval: full vs w/o-${abl} ====="
  GPU=$GPU ABLATE=$abl EPISODES=8 STEPS=900 S2_PERIOD=6 TOWN=Town05 SEED_BASE=90000 \
    bash carla_uav_tracking/rollout/eval_ablation.sh || echo "[pipeline] eval no${abl} errored"
done
echo "[pipeline] $(date -u +%H:%M:%S) DONE"
