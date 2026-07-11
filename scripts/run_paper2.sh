#!/usr/bin/env bash
# Paper 2 experiment runner (convenience wrapper)
# Usage:
#   ./scripts/run_paper2.sh --config configs/experiments/paper2/c1_g2_m2_d_epa_rhp.yaml --experiment d_epa_rhp
#   ./scripts/run_paper2.sh --config configs/experiments/paper2/ablation_voi.yaml --experiment event_dual_rhp
#   ./scripts/run_paper2.sh --sweep configs/experiments/paper2/sweep_full.yaml

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$1" = "--sweep" ]; then
    shift
    PYTHONPATH="$PROJECT_DIR/src" python3 "$PROJECT_DIR/scripts/paper2_sweep.py" --config "$@"
else
    PYTHONPATH="$PROJECT_DIR/src" python3 -m uavlab.paper2.runner.run "$@"
fi
