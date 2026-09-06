#!/usr/bin/env bash
# ONE batch for the paper's two tables, all 20-ep same-seed (91000-19), same ckpt where noted,
# live CARLA → fully comparable (eliminates cross-batch DiT-sampling run-variance).
#
# MAIN (WHICH ladder, fully-deployable track mode, ckpt=stage2_v5, vary the identity mechanism):
#   l1_xattn        single-frame language grounding
#   l2_vlmpool      + appearance re-ID (coarse VLM-pool feature)
#   l3_dino         + crop-DINOv2 feature + K2 gallery
#   l4_dino_conf    + confidence-gated commitment
#   l5_dino_conf_tavg  + temporal-EMA re-ID (tavg=0.4)   <- final system
# NEGATIVES (same config as l4 = track+reid-dino-K2+conf05, vary the CKPT = control-side fine-tune):
#   neg_bc04        BC-centering (gentlest blend 0.4)   -> pairs vs l4
#   neg_rle0        Stage-3 RL E0 (SVG centering)       -> pairs vs l4
#
#   EPISODES=20 SEED_BASE=91000 STEPS=900 GPU=2 bash carla_uav_tracking/rollout/eval_paper_batch.sh
set -uo pipefail   # NOT -e: one arm dying shouldn't abort the batch
REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}
NET=cyh-carla-net
CONFIG=${CONFIG:-train/config_v5.yaml}
EPISODES=${EPISODES:-20}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-2}
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT_CTR=/workspace/rollout/eval_out
mkdir -p "$REPO/carla_uav_tracking/rollout/eval_out"

REID="--tid-head reid --reid-feature dinov2 --reid-bank 2 --reid-topm 2"
# name | ckpt | policy flags
ARMS=(
  "l1_xattn|runs/stage2_v5/stage2_best.pt|--ex-source track --tid-head xattn"
  "l2_vlmpool|runs/stage2_v5/stage2_best.pt|--ex-source track --tid-head reid --reid-feature vlmpool --reid-bank 2 --reid-topm 2"
  "l3_dino|runs/stage2_v5/stage2_best.pt|--ex-source track $REID"
  "l4_dino_conf|runs/stage2_v5/stage2_best.pt|--ex-source track $REID --conf-tau 0.05"
  "l5_dino_conf_tavg|runs/stage2_v5/stage2_best.pt|--ex-source track $REID --conf-tau 0.05 --reid-tavg 0.4"
  "neg_bc04|runs/stage2_bc_center_b0.4.pt|--ex-source track $REID --conf-tau 0.05"
  "neg_rle0|runs/stage2_rl_e0.pt|--ex-source track $REID --conf-tau 0.05"
)

run_arm () {  # name ckpt flags out
  local name=$1 ckpt=$2 flags=$3 out=$4
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
  docker run -d $GPUFLAG --network "$NET" --name acot-policy-server --shm-size=16g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
      python carla_uav_tracking/rollout/policy_server.py \
      --config '$CONFIG' --ckpt '$ckpt' --language-mode neutral --s2-period '$S2' \
      $flags --port 5555" >/dev/null
  for _ in $(seq 1 120); do
    docker logs acot-policy-server 2>&1 | grep -q "listening" && break
    docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "[$name] server died:"; docker logs acot-policy-server | tail -20; return 1; }
    sleep 3
  done
  echo "=== [$name] $EPISODES ep × $STEPS on $TOWN (ckpt=$ckpt) $flags ==="
  docker exec cyh-carla python /workspace/rollout/run_rollout.py \
    --policy-host acot-policy-server --policy-port 5555 \
    --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$out" \
    || echo "[$name] run_rollout FAILED (will need re-run)"
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
}

for spec in "${ARMS[@]}"; do
  IFS='|' read -r name ckpt flags <<< "$spec"
  run_arm "$name" "$ckpt" "$flags" "$OUT_CTR/paper_${name}.json"
done

echo ""
echo "=== DONE. Continuous-metric tables: ==="
echo "MAIN ladder:  python carla_uav_tracking/rollout/continuous_metrics.py \\"
echo "  eval_out/paper_l1_xattn.json eval_out/paper_l2_vlmpool.json eval_out/paper_l3_dino.json eval_out/paper_l4_dino_conf.json eval_out/paper_l5_dino_conf_tavg.json"
echo "NEGATIVES:    python carla_uav_tracking/rollout/continuous_metrics.py \\"
echo "  eval_out/paper_l4_dino_conf.json eval_out/paper_neg_bc04.json eval_out/paper_neg_rle0.json"
