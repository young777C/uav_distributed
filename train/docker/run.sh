#!/usr/bin/env bash
# Run a command inside the training container on the A40s.
#
#   bash train/docker/run.sh                                   # GPU sanity check
#   bash train/docker/run.sh python -m train.stage1_ear --config train/config.yaml
#
# Mounts repo + data at their SAME absolute paths so configs work unchanged.
# DATA is mounted READ-ONLY so training can never disturb data generation.
set -euo pipefail

REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
IMAGE=${IMAGE:-acot-uav-train:cu118}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
mkdir -p "$HF_CACHE"

# --gpus all needs nvidia-container-toolkit; fall back to --runtime=nvidia if set.
GPUFLAG="--gpus all"
if ! docker info 2>/dev/null | grep -qi "nvidia"; then
  if [ -e /usr/bin/nvidia-container-runtime ]; then GPUFLAG="--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all"; fi
fi

# Optional stable container name (NAME=acot-train-run) so long runs are monitorable
# with `docker logs -f <NAME>` / `docker exec -it <NAME> ...`.
NAMEFLAG=""
if [ -n "${NAME:-}" ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  NAMEFLAG="--name $NAME"
fi

# DETACH=1 -> run in background (survives shell/session teardown); follow with `docker logs -f <NAME>`.
DETACHFLAG=""
if [ "${DETACH:-0}" = "1" ]; then DETACHFLAG="-d"; fi

exec docker run --rm $DETACHFLAG $NAMEFLAG $GPUFLAG \
  --shm-size=16g \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$REPO":"$REPO" \
  -v "$DATA":"$DATA":ro \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -w "$REPO" \
  "$IMAGE" "${@:-python -c 'import torch;print(\"torch\",torch.__version__,\"cuda_avail\",torch.cuda.is_available(),\"gpus\",torch.cuda.device_count())'}"
