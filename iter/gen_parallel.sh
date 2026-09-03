#!/bin/bash
# Parallel batch data generation across multiple GPUs. Each worker = its own CARLA
# server (own -graphicsadapter GPU + own rpc-port + own CARLA_TM_PORT) writing a
# disjoint shard, then shards are merged (renumbered) into one flat dir.
#
#   bash iter/gen_parallel.sh <total_episodes> <out_name> [gpus] [per_gpu]
#   e.g. bash iter/gen_parallel.sh 60 mvp_full "1 2 3 4 5 6 7" 1
#
# Never uses GPU0 (reserved for training) and never writes /data/mvp (stable).
set -u
N="${1:?total episodes required}"
OUT="${2:?out name required, e.g. mvp_full}"
GPUS="${3:-1 2 3 4 5 6 7}"
PER_GPU="${4:-1}"
CFG="config/scenarios/mvp.yaml"
CONT="cyh-carla"
HOSTROOT="/nvidia/hque/data/carla_data/${OUT}"    # host
CROOT="/data/${OUT}"                              # container

# -- build the worker list (gpu, sub-index) --
WORKERS=()
for g in $GPUS; do
  [ "$g" = "0" ] && { echo "refusing GPU0 (training)"; exit 1; }
  for ((w=0; w<PER_GPU; w++)); do WORKERS+=("$g:$w"); done
done
W=${#WORKERS[@]}
echo "[parallel] $N episodes across $W workers (gpus: $GPUS x$PER_GPU/gpu) -> $HOSTROOT"

# -- even split: first (N%W) workers get one extra --
base=$((N / W)); extra=$((N % W))

# -- launch each worker detached --
pids=()
for ((k=0; k<W; k++)); do
  gpu="${WORKERS[$k]%%:*}"; sub="${WORKERS[$k]##*:}"
  cnt=$base; [ "$k" -lt "$extra" ] && cnt=$((base + 1))
  [ "$cnt" -le 0 ] && continue
  rpc=$((2000 + gpu*10 + sub*3))
  tm=$((8000 + gpu*10 + sub))
  shard="${CROOT}/shard_${k}"
  echo "  worker $k: gpu=$gpu rpc=$rpc tm=$tm n=$cnt -> $shard"
  docker exec "$CONT" bash -lc \
    "export CARLA_TM_PORT=$tm; cd /workspace && bash scripts/run_resilient.sh '$CFG' '$shard' '$cnt' 42 20 '$rpc' '$gpu'" \
    > "/tmp/genpar_${OUT}_w${k}.log" 2>&1 &
  pids+=($!)
  sleep 3   # stagger server starts (avoid map-load thundering herd)
done

echo "[parallel] launched ${#pids[@]} workers; waiting..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[parallel] all workers done (failures: $fail)"

# -- merge shards -> flat dir with globally-unique renumbered ids --
mkdir -p "$HOSTROOT"
gid=0
for ((k=0; k<W; k++)); do
  sd="${HOSTROOT}/shard_${k}"
  [ -d "$sd" ] || continue
  for f in $(ls "$sd"/episode_*.h5 2>/dev/null | sort); do
    printf -v newname "episode_%06d.h5" "$gid"
    mv "$f" "${HOSTROOT}/${newname}"
    gid=$((gid + 1))
  done
  rm -rf "$sd"
done
echo "[parallel] merged $gid episodes into $HOSTROOT"
ls "$HOSTROOT"/episode_*.h5 2>/dev/null | wc -l
