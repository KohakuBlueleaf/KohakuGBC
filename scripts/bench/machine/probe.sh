#!/bin/bash
# probe.sh CARD TAG: card.sh in the background; from $DELAY s in (300), 60 py-spy dumps each of
# the machine process and vLLM's engine core 0.5 s apart, top's CPU% per thread, then the
# per-thread aggregates (threads.py). $SPY: the py-spy command (attaching may need sudo).
set -u
CARD=$1; TAG=$2
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT=${OUT_ROOT:-$HERE/../../../runs/bench/machine}
SPY=${SPY:-py-spy}
OUT=$OUT_ROOT/$TAG.probe; mkdir -p "$OUT"
bash "$HERE/card.sh" "$CARD" "$TAG" > "$OUT/run.out" 2>&1 &
sleep "${DELAY:-300}"
MAIN=$(pgrep -f "run.py $OUT_ROOT/$TAG " | head -1)
CORE=$(ps -eo pid,ppid,comm | awk -v p="$MAIN" '$2==p && $3 ~ /VLLM::Engine/ {print $1}')
echo "main $MAIN core $CORE" > "$OUT/pids"
top -H -b -n 3 -d 2 -p "$MAIN,$CORE" > "$OUT/top.txt"
for _ in $(seq 1 60); do
  timeout 10 $SPY dump -p "$MAIN" --nonblocking >> "$OUT/main.txt" 2>&1; echo "=====" >> "$OUT/main.txt"
  timeout 10 $SPY dump -p "$CORE" --nonblocking >> "$OUT/core.txt" 2>&1; echo "=====" >> "$OUT/core.txt"
  sleep 0.5
done
${PY:-python} "$HERE/threads.py" "$OUT/main.txt" > "$OUT/main.agg"
${PY:-python} "$HERE/threads.py" "$OUT/core.txt" > "$OUT/core.agg"
wait
echo "PROBE_DONE $TAG"
