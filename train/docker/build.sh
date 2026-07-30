#!/usr/bin/env bash
# Build the training image. Does NOT touch host CUDA/driver — the image ships its
# own cu118 torch which runs on the host R470 driver via minor-version compat.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
docker build -t acot-uav-train:cu118 -f "$HERE/Dockerfile" "$HERE"
echo "Built acot-uav-train:cu118. Verify GPU access with:"
echo "  bash $HERE/run.sh   # prints torch/cuda/gpu status (expect cuda_avail True)"
