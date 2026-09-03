#!/bin/bash
# Resilient batch runner: keeps a CARLA server alive AND resumes generate_batch
# past any episode that crashes the client, until N episodes are attempted.
#
# Usage (inside the cyh-carla container):
#   bash scripts/run_resilient.sh <config> <output_dir> <N> [seed] [max_gb] [port] [gpu]
set -u
CFG="$1"; OUT="$2"; N="$3"; SEED="${4:-42}"; MAXGB="${5:-60}"
PORT="${6:-2000}"; GPU="${7:-0}"
mkdir -p "$OUT"

port_up() {
    python -c "import socket,sys;s=socket.socket();s.settimeout(1);\
sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null
}

ensure_server() {
    if port_up; then return 0; fi
    echo "  [server] port $PORT down — launching CARLA (gpu $GPU, Epic)..."
    # -graphicsadapter selects the Vulkan render GPU. CARLA 0.9.15 renders via
    # Vulkan, which IGNORES CUDA_VISIBLE_DEVICES — without -graphicsadapter it
    # always renders on physical GPU0 (the training GPU). Pin it to $GPU.
    (cd "${CARLA_ROOT:-/home/carla}" && CUDA_VISIBLE_DEVICES="$GPU" nohup \
        ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port="$PORT" -quality-level=Epic \
        -nosound -graphicsadapter="$GPU" \
        >/tmp/carla_server_${PORT}.log 2>&1 &)
    for _ in $(seq 1 45); do
        if port_up; then echo "  [server] up on $PORT"; sleep 2; return 0; fi
        sleep 4
    done
    echo "  [server] FAILED to come up on $PORT"; return 1
}

for attempt in $(seq 1 200); do
    start=0
    if [ -f "$OUT/.attempted" ]; then start=$(( $(cat "$OUT/.attempted") + 1 )); fi
    remaining=$(( N - start ))
    if [ "$remaining" -le 0 ]; then
        echo "=== all $N episodes attempted; done ==="
        break
    fi
    ensure_server || { echo "cannot start server; aborting"; exit 1; }
    echo "=== attempt $attempt: start-id=$start remaining=$remaining ==="
    python scripts/generate_batch.py \
        --num-episodes "$remaining" --start-id "$start" \
        --config "$CFG" --output "$OUT" --seed "$SEED" --port "$PORT" --max-size-gb "$MAXGB"
    code=$?
    if [ "$code" = 0 ]; then
        echo "=== batch finished cleanly ==="
        break
    fi
    echo "=== batch exited $code (client/server crash); recovering in 3s ==="
    sleep 3
done
