#!/usr/bin/env bash
# Logged, preflighted launcher — the rigor guardrail for long/background runs.
# Replaces ad-hoc `docker run ... >/dev/null` (which lost logs TWICE, see memory
# acot-uav-experiment-rigor). Guarantees:
#   * output ALWAYS goes to a MOUNTED host file (survives --rm; never /dev/null)
#   * bigger --shm-size (DataLoader workers share tensors via /dev/shm — the shm
#     blow-up that crashed grounding_train came from too-small shm)
#   * a preflight print (image/gpu/shm/log/cmd) + a smoke reminder
#
# Usage (foreground, tees live + to file):
#   LOG=runs/foo.log GPU=2 bash train/docker/run_logged.sh python -m train.xxx ...
# Background (still logs to the mounted file; follow with tail -f $LOG):
#   DETACH=1 LOG=runs/foo.log GPU=2 bash train/docker/run_logged.sh python -m train.xxx ...
set -euo pipefail

REPO=/nvidia/hque/code/cyhe/uav-acot-track
DATA=/nvidia/hque/data/carla_data
IMAGE=${IMAGE:-acot-uav-train:cu118}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
SHM=${SHM:-32g}
# ALWAYS --gpus all so host GPU index == container index -> the inner `--device cuda:N`
# selects host GPU N (matches docker/run.sh + every working train run). Do NOT use
# `--gpus device=N`: that exposes ONE gpu as cuda:0, making `--device cuda:N` an
# "invalid device ordinal" (the mismatch that just bit the guardrail test). Pick the
# GPU via the command's own --device flag, NOT a GPU= env here.
GPUFLAG="--gpus all"
mkdir -p "$HF_CACHE" "$REPO/runs"

# log path: default timestamped; force under the mounted repo so it survives --rm
LOG=${LOG:-runs/run_$(date +%Y%m%d_%H%M%S).log}
case "$LOG" in /*) ;; *) LOG="$REPO/$LOG";; esac
NAME=${NAME:-acot-run-$(date +%s)}

echo "[run_logged] preflight: image=$IMAGE gpus=all(pick via --device cuda:N) shm=$SHM name=$NAME"
echo "[run_logged] log  -> $LOG   (mounted; survives --rm)"
echo "[run_logged] cmd  -> $*"
echo "[run_logged] TIP: smoke 1-2 units WITH this exact shm/batch/workers before any full fan-out."

docker rm -f "$NAME" >/dev/null 2>&1 || true
DOCKER=(docker run --rm --name "$NAME" $GPUFLAG --shm-size="$SHM"
  -e HF_TOKEN="${HF_TOKEN:-}" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
  -v "$REPO":"$REPO" -v "$DATA":"$DATA":ro -v "$HF_CACHE":/root/.cache/huggingface
  -w "$REPO" "$IMAGE" "$@")

if [ "${DETACH:-0}" = "1" ]; then
  nohup "${DOCKER[@]}" > "$LOG" 2>&1 &
  echo "[run_logged] detached pid=$!  follow: tail -f $LOG   (docker stop $NAME to kill)"
else
  "${DOCKER[@]}" 2>&1 | tee "$LOG"
fi
