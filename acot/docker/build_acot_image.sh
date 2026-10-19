#!/usr/bin/env bash
# Build acot-uav-train:cu118-acot (see Dockerfile.acot). The wheel is downloaded on the HOST (containers do no
# downloads); the build itself runs with --network none.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
pip download timm==1.0.29 --no-deps -d "$ctx" -q
echo "4266486e355c9beb9432e689de580d09e2efe7fa92a9a79e76c03e7763eac724  $ctx/timm-1.0.29-py3-none-any.whl" | sha256sum -c -
cp "$here/Dockerfile.acot" "$ctx/Dockerfile"
docker image inspect acot-uav-train:cu118 >/dev/null || { echo "base image acot-uav-train:cu118 missing: bash train/docker/build.sh"; exit 2; }
docker build --network none -t acot-uav-train:cu118-acot "$ctx"
docker image inspect -f '{{.Id}}' acot-uav-train:cu118-acot
