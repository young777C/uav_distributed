#!/usr/bin/env bash
# One-shot build: main.tex -> main.pdf using the bundled Tectonic binary.
# Intermediates (.aux/.bbl) are kept so citation numbers resolve cleanly.
set -euo pipefail
cd "$(dirname "$0")"
export HOME="${HOME:-/home/hque}"
TECTONIC="./.tools/tectonic"
if [ ! -x "$TECTONIC" ]; then
  echo "ERROR: $TECTONIC not found."
  echo "Reinstall it (see README.md > 'PDF build & auto-update')."
  exit 1
fi
exec "$TECTONIC" -X compile main.tex --keep-logs --keep-intermediates
