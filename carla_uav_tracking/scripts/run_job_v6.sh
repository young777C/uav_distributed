#!/bin/bash
# v6 data-generation job (acot_note/1003_data-regen-v6-spec.md). Inside the cyh-carla container.
# Resilient like run_resilient.sh, but PARALLEL-SAFE: per-job progress file + summary jsonl, one town,
# dedicated CARLA port / Vulkan GPU / TrafficManager port.
#   bash scripts/run_job_v6.sh <scenario.yaml> <out_dir> <subset> <town> <start_id> <n> <seed_base> <port> <gpu> <tm_port> <provenance.json>
set -u
CFG="$1"; OUT="$2"; SUBSET="$3"; TOWN="$4"; START="$5"; N="$6"; SEED="$7"; PORT="$8"; GPU="$9"; TM="${10}"; PROV="${11}"
JOB="${SUBSET}_${TOWN}_${START}"
PROG="$OUT/.progress/$JOB"; SUM="$OUT/.summaries/$JOB.jsonl"; mkdir -p "$OUT/.progress" "$OUT/.summaries"
export CARLA_TM_PORT="$TM"
port_up() { python -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null; }
ensure_server() {
    if port_up; then return 0; fi
    echo "  [server] port $PORT down — launching CARLA (gpu $GPU)"
    (cd "${CARLA_ROOT:-/home/carla}" && nohup ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port="$PORT" \
        -quality-level=Epic -nosound -graphicsadapter="$GPU" >/tmp/carla_server_${PORT}.log 2>&1 &)
    for _ in $(seq 1 60); do if port_up; then echo "  [server] up on $PORT"; sleep 20; return 0; fi; sleep 4; done
    echo "  [server] FAILED on $PORT"; return 1
}
END=$((START + N))
for attempt in $(seq 1 100); do
    next=$START
    if [ -f "$PROG" ]; then next=$(( $(cat "$PROG") + 1 )); fi
    remaining=$(( END - next ))
    if [ "$remaining" -le 0 ]; then echo "=== [$JOB] all attempted ==="; break; fi
    ensure_server || exit 1
    echo "=== [$JOB] attempt $attempt: start-id=$next remaining=$remaining ==="
    python scripts/generate_batch.py --num-episodes "$remaining" --start-id "$next" --config "$CFG" \
        --output "$OUT" --seed "$SEED" --port "$PORT" --max-size-gb 100000 --town "$TOWN" --subset "$SUBSET" \
        --progress-file "$PROG" --summary-jsonl "$SUM" --provenance "$PROV"
    code=$?
    if [ "$code" = 0 ]; then echo "=== [$JOB] finished cleanly ==="; break; fi
    echo "=== [$JOB] exited $code; recovering in 5s ==="; sleep 5
done
# Top-up: skipped / QC-rejected episodes are replaced from a private id block TB+j (j < 2N, TB disjoint per chunk),
# so every job delivers N recorded episodes without colliding with the next chunk's ids.
count_ok() { n=0; for id in "$@"; do [ -f "$OUT/episode_$(printf %06d $id).h5" ] && n=$((n+1)); done; echo $n; }
ids_main=$(seq $START $((END-1)))
# disjoint top-up block per chunk: town_base + 500 + 2*(offset of the chunk in its town block)
TB=$(( START - START % 1000 + 500 + 2 * (START % 1000) ))
for round in $(seq 1 4); do
    ids_top=$(seq $TB $((TB+2*N-1)))
    have=$(count_ok $ids_main $ids_top)
    [ "$have" -ge "$N" ] && break
    tstart=$TB; TPROG="$OUT/.progress/${JOB}_topup"
    tnext=$tstart; [ -f "$TPROG" ] && tnext=$(( $(cat "$TPROG") + 1 ))
    need=$(( N - have )); tend=$(( TB + 2*N ))
    [ $((tnext + need)) -gt $tend ] && need=$(( tend - tnext ))
    [ "$need" -le 0 ] && break
    ensure_server || exit 1
    echo "=== [$JOB] top-up round $round: have=$have need=$need from id $tnext ==="
    python scripts/generate_batch.py --num-episodes "$need" --start-id "$tnext" --config "$CFG" \
        --output "$OUT" --seed "$SEED" --port "$PORT" --max-size-gb 100000 --town "$TOWN" --subset "$SUBSET" \
        --progress-file "$TPROG" --summary-jsonl "$SUM" --provenance "$PROV"
done
echo "=== [$JOB] delivered $(count_ok $ids_main $(seq $TB $((TB+2*N-1))))/$N ==="
