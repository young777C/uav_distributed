#!/usr/bin/env bash
# A-HANDOFF: when the external-baseline rerun finishes, run the reanchor experiment (paper2 甲).
# Waits for '=== DONE rerun ===', frees my resources, verifies the single CARLA world is FREE+alive
# (restarts CARLA if it crashed), runs Step-1 smoke (must fire re-anchors >0), then Step-2 full 18ep.
# Runs from the WORKING TREE (uncommitted policy.py/policy_server.py/eval_reanchor.sh).
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track
cd "$REPO"
say(){ echo "[handoff $(date +%H:%M:%S)] $*"; }

carla_alive(){ docker exec cyh-carla bash -c "python -c \"import socket,sys;s=socket.socket();s.settimeout(2);sys.exit(0 if s.connect_ex(('127.0.0.1',2012))==0 else 1)\"" 2>/dev/null; }
carla_restart(){
  say "CARLA dead — restarting"
  docker exec cyh-carla bash -c "pkill -9 -f CarlaUE4 2>/dev/null; sleep 2"
  docker exec -d cyh-carla bash -c 'cd /home/carla && CUDA_VISIBLE_DEVICES=1 nohup ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2012 -quality-level=Epic -nosound -graphicsadapter=1 >/tmp/carla_server_2012.log 2>&1 &'
  until carla_alive; do sleep 4; done; sleep 5; say "CARLA back up"
}

say "waiting for external eval (=== DONE rerun ===) ..."
until grep -q "=== DONE rerun ===" runs/ext_rerun.log 2>/dev/null; do sleep 60; done
say "external eval done. freeing my resources (acot-policy-server, dam4sam-svc → GPU2/3/4)"
docker rm -f acot-policy-server dam4sam-svc >/dev/null 2>&1 || true
sleep 5

# hard prereq: no other rollout on the shared CARLA
until [ "$(docker exec cyh-carla pgrep -f run_rollout.py 2>/dev/null | wc -l)" = "0" ]; do say "waiting: a rollout still on CARLA"; sleep 15; done
carla_alive || carla_restart

# ---------- Step 1: smoke (ARMS=lang, 2ep) — must fire re-anchors > 0 ----------
say "=== STEP 1 smoke (ARMS=lang, 2ep×300) ==="
: > runs/reanchor_smoke_srvlog.txt
( until docker ps --format '{{.Names}}' | grep -q '^acot-policy-server-reanchor$'; do sleep 3; done
  docker logs -f acot-policy-server-reanchor >> runs/reanchor_smoke_srvlog.txt 2>&1 ) &
CAP=$!
EPISODES=2 STEPS=300 ARMS=lang GPU=2 bash carla_uav_tracking/rollout/eval_reanchor.sh >> runs/reanchor_smoke.log 2>&1 || true
kill $CAP 2>/dev/null || true
docker rm -f acot-policy-server-reanchor >/dev/null 2>&1 || true

FIRED=$(grep -oE 'fired [0-9]+ re-anchors' runs/reanchor_smoke_srvlog.txt 2>/dev/null | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}')
JSON=carla_uav_tracking/rollout/eval_out/reanchor_lang.json
if [ -f "$JSON" ] && [ "${FIRED:-0}" -gt 0 ]; then
  say "smoke PASS: reanchor_lang.json written, re-anchors fired total=$FIRED (>0)"
else
  say "smoke FAILED: json=$([ -f "$JSON" ] && echo yes || echo NO), re-anchors fired=$FIRED (need >0). NOT running full — check --reanchor-patience / trigger logic. Aborting."
  exit 1
fi

# ---------- Step 2: full 18ep 3-arm ----------
carla_alive || carla_restart
say "=== STEP 2 full (18ep×900, 3 arms: deploy/lang/oracle) ==="
EPISODES=18 SEED_BASE=91000 STEPS=900 GPU=2 bash carla_uav_tracking/rollout/eval_reanchor.sh >> runs/reanchor_full.log 2>&1 || true
say "=== reanchor FULL done. arms: $(ls carla_uav_tracking/rollout/eval_out/{reid_deploy,reanchor_lang,reanchor_oracle}.json 2>/dev/null | wc -l)/3 ==="
