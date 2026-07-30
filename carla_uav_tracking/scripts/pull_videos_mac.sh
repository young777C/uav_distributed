#!/usr/bin/env bash
# Pull CARLA inspection videos from the generation server to THIS Mac and open them.
#
# Runs on the Mac (not in the container). Uses rsync over ssh (resumable, skips
# already-downloaded files) then opens the folder + clips in QuickTime.
#
# Usage:
#   ./pull_videos_mac.sh <user@server> [dataset] [remote_root]
#     dataset      dataset dir name under the data root (default: occ_check)
#                  e.g.  occ_check   (336x336)   |   occ_check_1080   (1080x1080)
#     remote_root  server data root (default: /nvidia/hque/data/carla_data)
#
# Examples:
#   ./pull_videos_mac.sh hque@10.0.0.5                     # 336² inspection set
#   ./pull_videos_mac.sh hque@10.0.0.5 occ_check_1080      # 1080² set
set -euo pipefail

SERVER="${1:?usage: $0 <user@server> [dataset] [remote_root]}"
DATASET="${2:-occ_check}"
REMOTE_ROOT="${3:-/nvidia/hque/data/carla_data}"

REMOTE_DIR="${REMOTE_ROOT}/${DATASET}/videos"
LOCAL_DIR="${HOME}/carla_videos/${DATASET}"

mkdir -p "$LOCAL_DIR"

echo "==> Pulling ${SERVER}:${REMOTE_DIR}/*.mp4"
echo "    -> ${LOCAL_DIR}"
if ! rsync -avh --progress "${SERVER}:${REMOTE_DIR}/"*.mp4 "$LOCAL_DIR/"; then
  echo "!! No videos found (are they rendered yet?). List what's on the server with:"
  echo "   ssh ${SERVER} 'ls -la ${REMOTE_DIR}'"
  exit 1
fi

echo "==> Opening ${LOCAL_DIR}"
open "$LOCAL_DIR"            # reveal folder in Finder
open "$LOCAL_DIR"/*.mp4     # open clips in QuickTime
