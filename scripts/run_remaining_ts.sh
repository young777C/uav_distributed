#!/bin/bash
# Run remaining time-scale experiments using direct runner calls.
# Handles combined config creation and directory setup.
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
OUT_ROOT="$PROJECT_ROOT/results/ts_sensitivity/parallel"
PYTHONPATH="$PROJECT_ROOT/src"
EPISODES=3
SEEDS="0 1 2"

mkdir -p "$OUT_ROOT"

# Helper: run a single experiment
run_one() {
    local case=$1
    local system_cfg=$2
    local ts=$3
    local seed=$4
    local system_name
    system_name=$(basename "$system_cfg" .yaml)
    local exp_name="${case}__${system_name}"
    local run_dir="$OUT_ROOT/$exp_name/Ts${ts}/seed${seed}"
    local metrics_path="$run_dir/metrics.jsonl"
    local combined_cfg="$run_dir/combined_config.yaml"

    # Skip if already done
    if [ -f "$metrics_path" ] && [ "$(wc -l < "$metrics_path")" -ge "$EPISODES" ]; then
        echo "  SKIP $exp_name Ts=$ts seed=$seed (already done)"
        return 0
    fi

    mkdir -p "$run_dir"

    # Write combined config using Python
    PYTHONPATH="$PYTHONPATH" python3 -c "
import yaml, sys
from pathlib import Path
ROOT = Path('$PROJECT_ROOT')
payload = {
    'extends': [
        str(ROOT / 'configs/experiments/paper1/cases/${case}.yaml'),
        str(ROOT / '$system_cfg'),
    ],
    'experiment': {'id': '$exp_name'},
}
with open('$combined_cfg', 'w') as f:
    yaml.dump(payload, f)
"

    echo "  RUN $exp_name Ts=$ts seed=$seed"
    PYTHONPATH="$PYTHONPATH" python3 -m uavlab.paper1.runner.run_vis \
        --config "$combined_cfg" \
        --episodes "$EPISODES" \
        --seed "$seed" \
        --slow_interval_steps "$ts" \
        --run_dir "$run_dir" \
        --metrics_jsonl "$metrics_path" 2>&1 | tail -1

    echo "  DONE $exp_name Ts=$ts seed=$seed"
}

echo "=== Stage 1: C2 FDLC Ts=80 (seeds 1,2) ==="
for s in 1 2; do
    run_one "c2_g2_m2" "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml" 80 $s
done

echo ""
echo "=== Stage 2: C2 Periodic Goal Ts=80 (all seeds) ==="
for s in $SEEDS; do
    run_one "c2_g2_m2" "configs/experiments/paper1/system/ts_sweep_periodic_goal.yaml" 80 $s
done

echo ""
echo "=== Stage 3: C2 Periodic Goal Ts=40 (all seeds) ==="
for s in $SEEDS; do
    run_one "c2_g2_m2" "configs/experiments/paper1/system/ts_sweep_periodic_goal.yaml" 40 $s
done

echo ""
echo "=== Stage 4: C2 FDLC Ts=20 (all seeds) ==="
for s in $SEEDS; do
    run_one "c2_g2_m2" "configs/experiments/paper1/system/struct_full_dual_loop_distributed.yaml" 20 $s
done

echo ""
echo "=== All remaining experiments completed ==="
