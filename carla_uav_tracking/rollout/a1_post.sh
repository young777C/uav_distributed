#!/usr/bin/env bash
# fires after a1_finish: regenerate all data-driven analyses+figures now that TAHRelM json exists.
set -uo pipefail
REPO=/nvidia/hque/code/cyhe/uav-acot-track; cd "$REPO"
say(){ echo "[a1post $(date +%H:%M:%S)] $*"; }
say "waiting for A1 (=== A1 DONE + tahrelm json)..."
until grep -q "=== A1 DONE ===" runs/a1_finish.log 2>/dev/null \
      && [ -f carla_uav_tracking/rollout/eval_out/paper_ext_tahrelm_gt.json ]; do sleep 60; done
say "A1 done — regenerating analyses + figures with TAHRelM arm"
echo "===== STABILITY (control_stability.py) ====="
python carla_uav_tracking/rollout/control_stability.py 2>&1 | sed 's/^/  /'
echo "===== SCENARIO (scenario_split.py) ====="
python carla_uav_tracking/rollout/scenario_split.py 2>&1 | sed 's/^/  /'
echo "===== FIGURES (regen_arm_figs.py) ====="
python acot_note/figs/regen_arm_figs.py 2>&1 | sed 's/^/  /'
say "=== A1-POST DONE (3 figures + tables regenerated; awaiting interpretation) ==="
