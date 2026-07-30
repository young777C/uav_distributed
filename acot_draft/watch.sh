#!/usr/bin/env bash
# Continuous auto-rebuild. Watches main.tex, references.bib, and figures/,
# and rebuilds main.pdf whenever any of them changes. Ctrl-C to stop.
#
# Dependency-free: polls file contents once per second (no inotify/entr needed),
# so it works over NFS and inside plain shells. main.pdf and build intermediates
# are intentionally NOT watched, to avoid a rebuild loop.
set -uo pipefail
cd "$(dirname "$0")"
export HOME="${HOME:-/home/hque}"

sig() {
  { cat main.tex references.bib 2>/dev/null; ls -l figures/ 2>/dev/null; } | cksum
}

ts() { date '+%H:%M:%S' 2>/dev/null || echo now; }

echo "[watch $(ts)] initial build ..."
if ./build.sh >/dev/null 2>&1; then
  echo "[watch $(ts)] OK -> main.pdf"
else
  echo "[watch $(ts)] initial build FAILED -> run ./build.sh to see the error"
fi

last="$(sig)"
echo "[watch $(ts)] watching main.tex, references.bib, figures/  (Ctrl-C to stop)"
while true; do
  cur="$(sig)"
  if [ "$cur" != "$last" ]; then
    last="$cur"
    echo "[watch $(ts)] change detected -> rebuilding ..."
    if ./build.sh >/dev/null 2>&1; then
      echo "[watch $(ts)] OK -> main.pdf updated"
    else
      echo "[watch $(ts)] BUILD FAILED -> run ./build.sh to see the error"
    fi
  fi
  sleep 1
done
