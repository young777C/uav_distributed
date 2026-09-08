#!/usr/bin/env bash
# DEPLOYABLE identity + LANGUAGE RE-ANCHOR closed-loop PAIRED eval (paper2 甲).
#
# All arms are FULLY GT-FREE on WHICH: --gallery-seed committed stores the crop of the identity we
# currently BELIEVE (prior commitment; language pick at cold-start), never the GT target. So the
# appearance gallery can drift onto a look-alike (realistic) — which is exactly where a drift-free
# LANGUAGE re-anchor earns its keep. Same ckpt, same held-out seeds, same deployable reid config
# (track + reid-dinov2 + K2 + topm2 + conf-tau0.05 + reid-tavg0.4). SINGLE VARIABLE = --reanchor:
#   reid_deploy      : committed gallery, reanchor off   → honest deployable re-ID baseline
#   reanchor_lang    : + language re-anchor on sustained collapse  → 兑现甲
#   reanchor_oracle  : + GT re-anchor  → headroom upper bound (how much a PERFECT re-anchor buys)
# Also pairs against the legacy GT-seeded gallery run (reid_tavg0.4.json) to quantify the GT crutch.
#
# GATE: reanchor_lang beats reid_deploy on track_correct_frac ↑ / max_wrong_run_s ↓ / lock_wrong ↓
#       (paired, same seeds) → language is load-bearing as fallback arbitration.
#
#   EPISODES=18 SEED_BASE=91000 STEPS=900 GPU=2 bash carla_uav_tracking/rollout/eval_reanchor.sh
set -euo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
HF=${HF_CACHE:-$HOME/.cache/huggingface}
NET=cyh-carla-net
SRV=${SRV:-acot-policy-server-reanchor}   # UNIQUE name — never collide with a concurrent acot-policy-server run
CKPT=${CKPT:-runs/stage2_v5/stage2_best.pt}
CONFIG=${CONFIG:-train/config_v5.yaml}
EPISODES=${EPISODES:-18}; SEED_BASE=${SEED_BASE:-91000}; STEPS=${STEPS:-900}
TOWN=${TOWN:-Town05}; S2=${S2_PERIOD:-6}; GPU=${GPU:-2}; TAVG=${TAVG:-0.4}
PATIENCE=${PATIENCE:-5}; RTAVG=${RTAVG:-0.5}   # RTAVG=temporal-language re-anchor EMA (0=single-frame)
[ "$GPU" = "all" ] && GPUFLAG="--gpus all" || GPUFLAG="--gpus device=$GPU"
OUT_CTR=/workspace/rollout/eval_out
mkdir -p "$REPO/carla_uav_tracking/rollout/eval_out"

run_arm () {  # name gallery_seed reanchor rtavg out
  local name=$1 gseed=$2 reanch=$3 rtavg=$4 out=$5
  docker rm -f $SRV >/dev/null 2>&1 || true
  docker run -d $GPUFLAG --network "$NET" --name $SRV --shm-size=16g \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF":/root/.cache/huggingface -w "$REPO" \
    acot-uav-train:cu118 bash -c "pip install -q timm==1.0.29 2>/dev/null; \
      python carla_uav_tracking/rollout/policy_server.py \
      --config '$CONFIG' --ckpt '$CKPT' --language-mode neutral --s2-period '$S2' \
      --ex-source track --tid-head reid --reid-feature dinov2 --reid-bank 2 --reid-topm 2 \
      --conf-tau 0.05 --reid-tavg '$TAVG' \
      --gallery-seed '$gseed' --reanchor '$reanch' --reanchor-patience '$PATIENCE' \
      --reanchor-tavg '$rtavg' --port 5555" >/dev/null
  for _ in $(seq 1 100); do
    docker logs $SRV 2>&1 | grep -q "listening" && break
    docker ps --format '{{.Names}}' | grep -q $SRV || { echo "[$name] server died:"; docker logs $SRV; exit 1; }
    sleep 3
  done
  echo "=== [$name] $EPISODES ep × $STEPS on $TOWN (gallery=$gseed reanchor=$reanch) ==="
  docker exec cyh-carla python /workspace/rollout/run_rollout.py \
    --policy-host $SRV --policy-port 5555 \
    --episodes "$EPISODES" --seed-base "$SEED_BASE" --steps "$STEPS" --town "$TOWN" --out "$out"
  docker rm -f $SRV >/dev/null 2>&1 || true
}

# ARMS = space-separated subset of: deploy lang langtavg oracle   (default: the salvage set)
for arm in ${ARMS:-langtavg}; do
  case "$arm" in
    deploy)   run_arm reid_deploy     committed off    0       "$OUT_CTR/reid_deploy.json" ;;
    lang)     run_arm reanchor_lang   committed lang   0       "$OUT_CTR/reanchor_lang.json" ;;
    langtavg) run_arm reanchor_langtavg committed lang "$RTAVG" "$OUT_CTR/reanchor_langtavg.json" ;;   # ★ temporal salvage
    oracle)   run_arm reanchor_oracle committed oracle 0       "$OUT_CTR/reanchor_oracle.json" ;;
  esac
done

echo ""
echo "=== PAIRED continuous metrics (same seeds) ==="
echo "--- crutch: legacy GT-gallery (reid_tavg0.4) vs deployable committed-gallery (reid_deploy) ---"
docker exec cyh-carla python /workspace/rollout/continuous_metrics.py \
  "$OUT_CTR/reid_tavg0.4.json" "$OUT_CTR/reid_deploy.json" || true
echo "--- 甲 salvage: reid_deploy vs single-frame-lang vs TEMPORAL-lang vs oracle (upper bound) ---"
docker exec cyh-carla python /workspace/rollout/continuous_metrics.py \
  "$OUT_CTR/reid_deploy.json" "$OUT_CTR/reanchor_lang.json" \
  "$OUT_CTR/reanchor_langtavg.json" "$OUT_CTR/reanchor_oracle.json"

