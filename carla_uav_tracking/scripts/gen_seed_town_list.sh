#!/bin/bash
# Generate an explicit list of episodes "seed:town" (inside the cyh-carla container), one generate_batch call per episode
# with --start-id = seed - SEED_BASE (so seed = SEED_BASE + episode id, exactly as a sequential run) and --town pinned
# (reproduces the sequential run's block-of-10 town schedule regardless of how the list is sharded). Keeps a CARLA server
# alive (Epic, -graphicsadapter); skips episodes already written; one retry per episode after a server restart.
#   bash scripts/gen_seed_town_list.sh <config> <output_dir> <port> <gpu> <seed_base> "s1:T1 s2:T2 ..."
set -u
CFG="$1"; OUT="$2"; PORT="$3"; GPU="$4"; SB="$5"; LIST="$6"
mkdir -p "$OUT"
port_up() { python -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null; }
ensure_server() {
    if port_up; then return 0; fi
    echo "  [server] port $PORT down — launching CARLA (gpu $GPU, Epic)..."
    (cd "${CARLA_ROOT:-/home/carla}" && CUDA_VISIBLE_DEVICES="$GPU" nohup ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port="$PORT" \
        -quality-level=Epic -nosound -graphicsadapter="$GPU" >/tmp/carla_server_${PORT}.log 2>&1 &)
    for _ in $(seq 1 45); do if port_up; then echo "  [server] up on $PORT"; sleep 5; return 0; fi; sleep 4; done
    echo "  [server] FAILED to come up on $PORT"; return 1
}
for item in $LIST; do
    seed="${item%%:*}"; town="${item##*:}"; ep=$((seed - SB))
    f=$(printf "%s/episode_%06d.h5" "$OUT" "$ep")
    if [ -f "$f" ]; then echo "=== seed $seed (ep $ep) exists — skip"; continue; fi
    for try in 1 2; do
        ensure_server || exit 1
        echo "=== seed $seed ep $ep town $town (try $try) ==="
        python scripts/generate_batch.py --num-episodes 1 --start-id "$ep" --seed "$SB" --town "$town" \
            --config "$CFG" --output "$OUT" --port "$PORT" --max-size-gb 60 && break
        echo "  [warn] generate_batch failed for seed $seed (try $try); restarting server"
        pkill -9 -f "carla-rpc-port=$PORT( |$)"; sleep 8
    done
done
echo "=== list done ==="
