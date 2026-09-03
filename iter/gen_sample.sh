#!/bin/bash
# DATA side of the review loop: generate ONE small sample batch for training-side
# review. Writes to /data/mvp_review_r<ROUND> (never /data/mvp) on GPU!=0, then
# logs a data_version row to the ledger. Safe to launch in the background.
#
#   bash iter/gen_sample.sh <round> [n_ep=12] [gpu=1] [port=2001] [seed]
#
# After it finishes, the training-reviewer agent runs:
#   python -m iter.review_sample --dir <hostdir> --out iter/reviews/r<round>.json --round <round>
set -eu
ROUND="${1:?round number required}"
N_EP="${2:-12}"
GPU="${3:-1}"
PORT="${4:-2001}"
SEED="${5:-$((42 + ROUND * 100))}"
CFG="config/scenarios/mvp.yaml"
HERE="$(cd "$(dirname "$0")/.." && pwd)"      # repo root (uav-acot-track)
CDIR="/data/mvp_review_r${ROUND}"             # container path
HDIR="/nvidia/hque/data/carla_data/mvp_review_r${ROUND}"   # host path (same volume)

if [ "$GPU" = "0" ]; then echo "refuse GPU0 (training)"; exit 1; fi
echo "[gen] round=$ROUND n=$N_EP gpu=$GPU port=$PORT seed=$SEED -> $CDIR"

docker exec cyh-carla bash -lc \
  "cd /workspace && bash scripts/run_resilient.sh '$CFG' '$CDIR' '$N_EP' '$SEED' 60 '$PORT' '$GPU'"

have=$(ls "$HDIR"/episode_*.h5 2>/dev/null | wc -l)
echo "[gen] done: $have/$N_EP episodes in $HDIR"

python "$HERE/iter/log_iter.py" --type data_version --round "$ROUND" --agent data \
  --version "mvp_review_r${ROUND}" \
  --config "gpu=$GPU,port=$PORT,seed=$SEED,cars_only=1,band=10,max_lost=15" \
  --metrics "episodes=$have" \
  --gates "review=pending" \
  --note "sample for training-side review (round $ROUND)" \
  --artifact "dir=$HDIR"
echo "[gen] logged data_version mvp_review_r${ROUND}; ready for review"
