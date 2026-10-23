#!/bin/bash
# probe.sh CARD TAG: card.sh in the background; once vLLM's stats line shows at least $RUNNING
# (64) requests running, and $DELAY s (60) later, 60 py-spy dumps each of the machine process
# and vLLM's engine core 0.5 s apart (the server's API process too with SERVE_PORT), top's CPU%
# per thread, then the per-thread aggregates (threads.py). Needs GBC_VLM_STATS=1. $SPY: the
# py-spy command (attaching may need sudo).
set -u
CARD=$1; TAG=$2
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT=${OUT_ROOT:-$HERE/../../../runs/bench/machine}
SPY=${SPY:-py-spy}
OUT=$OUT_ROOT/$TAG.probe; mkdir -p "$OUT"
bash "$HERE/card.sh" "$CARD" "$TAG" > "$OUT/run.out" 2>&1 &
LOG=$OUT_ROOT/$TAG.log
[ -n "${SERVE_PORT:-}" ] && LOG=$OUT_ROOT/$TAG.server.log
until grep -qE "Running: ([0-9]{3,}|[6-9][0-9]) reqs" "$LOG" 2> /dev/null; do sleep 5; done
sleep "${DELAY:-60}"
MAIN=$(pgrep -f "run.py $OUT_ROOT/$TAG " | head -1)
API=
[ -n "${SERVE_PORT:-}" ] && API=$(pgrep -f "vllm serve .*--port $SERVE_PORT" | head -1)
CORE=$(ps -eo pid,ppid,comm | awk -v p="${API:-$MAIN}" '$2==p && $3 ~ /VLLM::Engine/ {print $1}')
echo "main $MAIN api $API core $CORE" > "$OUT/pids"
top -H -b -n 3 -d 2 -p "$MAIN,$CORE${API:+,$API}" > "$OUT/top.txt"
for _ in $(seq 1 60); do
  timeout 10 $SPY dump -p "$MAIN" --nonblocking >> "$OUT/main.txt" 2>&1; echo "=====" >> "$OUT/main.txt"
  timeout 10 $SPY dump -p "$CORE" --nonblocking >> "$OUT/core.txt" 2>&1; echo "=====" >> "$OUT/core.txt"
  if [ -n "$API" ]; then
    timeout 10 $SPY dump -p "$API" --nonblocking >> "$OUT/api.txt" 2>&1; echo "=====" >> "$OUT/api.txt"
  fi
  sleep 0.5
done
for part in main core api; do
  [ -f "$OUT/$part.txt" ] && ${PY:-python} "$HERE/threads.py" "$OUT/$part.txt" > "$OUT/$part.agg"
done
wait
echo "PROBE_DONE $TAG"
