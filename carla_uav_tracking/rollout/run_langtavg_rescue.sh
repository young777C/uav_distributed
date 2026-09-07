#!/usr/bin/env bash
# Temporal LANGUAGE re-anchor rescue (救甲): apply per-actor temporal-EMA to the language re-anchor
# vote (accumulate over the re-appearance window, commit on integrated evidence) instead of the
# single-frame force-commit that falsified 甲 (0.363->0.225). Working tree (--reanchor-tavg, uncommitted).
# Smoke (fire>0) -> full 18ep langtavg arm. Single CARLA — caller confirmed FREE/NO_BG.
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; cd "$REPO"
RTAVG=${RTAVG:-0.5}
say(){ echo "[langtavg $(date +%H:%M:%S)] $*"; }
carla_alive(){ docker exec cyh-carla bash -c "python -c \"import socket,sys;s=socket.socket();s.settimeout(2);sys.exit(0 if s.connect_ex(('127.0.0.1',2012))==0 else 1)\"" 2>/dev/null; }
carla_restart(){ say "CARLA dead — restarting"; docker exec cyh-carla bash -c "pkill -9 -f CarlaUE4 2>/dev/null; sleep 2";
  docker exec -d cyh-carla bash -c 'cd /home/carla && CUDA_VISIBLE_DEVICES=1 nohup ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2012 -quality-level=Epic -nosound -graphicsadapter=1 >/tmp/carla_server_2012.log 2>&1 &';
  until carla_alive; do sleep 4; done; sleep 5; say "CARLA back"; }

carla_alive || carla_restart
# ---- Step 1 smoke ----
say "=== STEP 1 smoke (ARMS=langtavg RTAVG=$RTAVG, 2ep×300) ==="
: > runs/langtavg_smoke_srvlog.txt
( until docker ps --format '{{.Names}}' | grep -q '^acot-policy-server-reanchor$'; do sleep 3; done
  docker logs -f acot-policy-server-reanchor >> runs/langtavg_smoke_srvlog.txt 2>&1 ) &
CAP=$!
EPISODES=2 STEPS=300 ARMS=langtavg RTAVG=$RTAVG GPU=2 bash carla_uav_tracking/rollout/eval_reanchor.sh >> runs/langtavg_smoke.log 2>&1 || true
kill $CAP 2>/dev/null || true; docker rm -f acot-policy-server-reanchor >/dev/null 2>&1 || true
FIRED=$(grep -oE 'fired [0-9]+ re-anchors' runs/langtavg_smoke_srvlog.txt 2>/dev/null | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')
JSON=carla_uav_tracking/rollout/eval_out/reanchor_langtavg.json
if [ -f "$JSON" ] && [ "${FIRED:-0}" -gt 0 ]; then
  say "smoke PASS: reanchor_langtavg.json written, re-anchors fired=$FIRED (>0)"
else
  say "smoke FAILED: json=$([ -f "$JSON" ] && echo yes || echo NO) fired=$FIRED (need>0). NOT running full. check --reanchor-tavg/patience."; exit 1
fi
# ---- Step 2 full ----
carla_alive || carla_restart
say "=== STEP 2 full (18ep×900, ARMS=langtavg RTAVG=$RTAVG) ==="
EPISODES=18 SEED_BASE=91000 STEPS=900 RTAVG=$RTAVG GPU=2 ARMS=langtavg bash carla_uav_tracking/rollout/eval_reanchor.sh >> runs/langtavg_full.log 2>&1 || true
say "=== langtavg FULL done (RTAVG=$RTAVG). json: $([ -f "$JSON" ] && echo yes || echo NO) ==="
