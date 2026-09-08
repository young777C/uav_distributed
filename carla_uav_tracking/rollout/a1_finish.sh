#!/usr/bin/env bash
# path-A: retrain TAHRelM (TAHRel + learned motion-consensus) on the SAME offline+DAgger cache (OFAT: only
# arch differs vs tahdag) → closed-loop gt-seeded eval → compare vs tahdag(0.158)/borrow#1(0.263)/l5.
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; cd "$REPO"
say(){ echo "[a1 $(date +%H:%M:%S)] $*"; }
say "retrain TAHRelM on offline+DAgger (GPU3, 60ep --motion --aug) — OFAT vs tahdag"
docker run --rm --gpus '"device=3"' -v "$REPO":"$REPO" -w "$REPO" acot-uav-train:cu118 \
  python train/tah_train.py --cache runs/tah_cache_dag.pt --out runs/tah_relm_dag.pt \
  --motion --aug --epochs 60 --wd 3e-4 --device cuda:0 2>&1 | grep -E 'params|arch=|ep60/60|done. best' | tail -4
[ -f runs/tah_relm_dag.pt ] || { say "RETRAIN FAILED (no ckpt)"; exit 1; }
say "closed-loop gt-seeded eval TAHRelM (GPU2, seed 91000-19)"
docker rm -f acot-policy-server >/dev/null 2>&1 || true
TAH_CKPT=runs/tah_relm_dag.pt OUT_NAME=paper_ext_tahrelm_gt GSEED=gt CONF=0.5 \
  EPISODES=20 SEED_BASE=91000 STEPS=900 GPU=2 bash carla_uav_tracking/rollout/eval_tah.sh 2>&1 | tail -4
say "=== 4-way compare: l5 / borrow#1 / TAH+DAgger / TAHRelM+DAgger ==="
docker exec cyh-carla python /workspace/rollout/continuous_metrics.py \
  /workspace/rollout/eval_out/paper_l5_dino_conf_tavg.json \
  /workspace/rollout/eval_out/paper_ext_reidmotion03.json \
  /workspace/rollout/eval_out/paper_ext_tahdag_gt.json \
  /workspace/rollout/eval_out/paper_ext_tahrelm_gt.json 2>&1 | tail -18
say "=== A1 DONE ==="
