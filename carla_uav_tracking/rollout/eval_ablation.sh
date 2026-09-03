#!/usr/bin/env bash
# H1/H2 closed-loop module ablation: full model (M0) vs w/o-EAR (H1) or w/o-IAR (H2),
# on the SAME held-out seed set. The ablated model was TRAINED with the module's DiT
# feed zeroed (train/config_v5_no{ear,iar}.yaml); the eval passes --ablate so inference
# matches. H1/H2 signal = ACTION quality (track_seconds / SR), NOT mis_follow (tid is
# unchanged). Reuses eval_out/m0.json as the full baseline if present (same settings!).
#
#   ABLATE=ear EPISODES=8 STEPS=900 S2_PERIOD=6 TOWN=Town05 GPU=2 bash carla_uav_tracking/rollout/eval_ablation.sh
set -euo pipefail
ABLATE=${ABLATE:?set ABLATE=ear or iar}

REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}
NET=cyh-carla-net
EPISODES=${EPISODES:-8}; SEED_BASE=${SEED_BASE:-90000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-all}
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT_HOST=$REPO/carla_uav_tracking/rollout/eval_out; OUT_CTR=/workspace/rollout/eval_out
mkdir -p "$OUT_HOST"

run_arm () {  # name config ckpt out [ablate]
  local name=$1 config=$2 ckpt=$3 out=$4 ablate=${5:-}
  local ablflag=""; [ -n "$ablate" ] && ablflag="--ablate $ablate"
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
  docker run -d $GPUFLAG --network "$NET" --name acot-policy-server --shm-size=16g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 python carla_uav_tracking/rollout/policy_server.py \
      --config "$config" --ckpt "$ckpt" --language-mode neutral --s2-period "$S2" $ablflag --port 5555 >/dev/null
  for _ in $(seq 1 80); do
    docker logs acot-policy-server 2>&1 | grep -q "listening" && break
    docker ps --format '{{.Names}}' | grep -q acot-policy-server || { echo "[$name] server died:"; docker logs acot-policy-server; exit 1; }
    sleep 3
  done
  echo "=== [$name] $EPISODES ep × $STEPS on $TOWN (ablate=${ablate:-none}) ==="
  docker exec cyh-carla python /workspace/rollout/run_rollout.py \
    --policy-host acot-policy-server --policy-port 5555 \
    --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$out"
  docker rm -f acot-policy-server >/dev/null 2>&1 || true
}

# Full baseline (reuse the H0 M0 run if it exists with matching settings).
if [ ! -f "$OUT_HOST/m0.json" ]; then
  run_arm M0_full train/config_v5.yaml runs/stage2_v5/stage2_best.pt "$OUT_CTR/m0.json"
else
  echo "=== reusing existing $OUT_HOST/m0.json as full baseline ==="
fi
# Ablated arm.
run_arm "M_no${ABLATE}" "train/config_v5_no${ABLATE}.yaml" "runs/stage2_v5_no${ABLATE}/stage2_best.pt" \
  "$OUT_CTR/no${ABLATE}.json" "$ABLATE"

echo "=== H1/H2 paired comparison (full vs w/o-${ABLATE}) ==="
docker exec cyh-carla python /workspace/rollout/compare_runs.py \
  "$OUT_CTR/m0.json" "$OUT_CTR/no${ABLATE}.json" --label-a M0_full --label-b "no${ABLATE}"
